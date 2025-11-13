# ==============================================
# 03_query_pipeline_fixed.py
# 안정형 버전 (Streaming + Sync 통합, 전역 캐싱)
# ==============================================
# pip install "tritonclient[grpc]" qdrant-client transformers numpy llama-index tqdm orjson rapidfuzz

import os, json, time, threading, math, re
import numpy as np
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from transformers import AutoTokenizer
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from rapidfuzz import fuzz
import logging
logger = logging.getLogger("uvicorn.error")  # uvicorn 콘솔에 바로 찍힘
HANGUL_INNER_SPACE_RE = re.compile(r'(?<=[\uAC00-\uD7AF]) (?=[\uAC00-\uD7AF])')
# ---------------------------
# 환경 설정
# ---------------------------
QDRANT_HOST  = os.getenv("QDRANT_HOST", "211.241.177.73")
QDRANT_URL   = os.getenv("QDRANT_URL", "http://211.241.177.73:6333")
COLLECTION   = os.getenv("QDRANT_COLLECTION", "peS2o_rag")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
TRITON_URL   = os.getenv("TRITON_URL", "211.241.177.73:8001")
MODEL_NAME   = os.getenv("TRITON_MODEL", "gemma_vllm_0")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "./data/")

TOP_K_BASE = 20
TOP_K_RETURN = 20
MAX_TOKENS    = 512
TEMPERATURE   = 0.6
TOP_P         = 0.9

SCORE_THRESHOLD = 0.15
FUZZ_MIN        = 40

CTX_TOKEN_BUDGET = 2200
SNIPPET_MAX_CHARS = 1800

# ---------------------------
# 전역 리소스 캐시
# ---------------------------
_qdr, _emb, _retriever = None, None, None

# tokenizer는 로컬 디렉토리/모델명 모두 지원
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
except Exception:
    # 폴백: 모델 이름이 올바르지 않을 때 기본 토크나이저
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ---------------------------
# Triton LLM (Streaming / Sync 겸용)
# ---------------------------

def _make_inputs(prompt: str, *, max_tokens: int, stop: List[str] | None, temperature: float, top_p: float):
    text = InferInput("text_input", [1], "BYTES")
    text.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    sparams = InferInput("sampling_parameters", [1], "BYTES")
    params = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if stop:
        params["stop"] = list(stop)
    sparams.set_data_from_numpy(np.array([json.dumps(params).encode("utf-8")], dtype=object))

    return text, sparams


def _triton_stream_generator(
        prompt,
        text,
        sparams,
        first_token_timeout=20,   # 첫 토큰까지 최대 20초
        idle_timeout=5,           # 토큰 사이 idle 은 5초
):
    cli = InferenceServerClient(url=TRITON_URL, verbose=False)

    stream_flag = InferInput("stream", [1], "BOOL")
    stream_flag.set_data_from_numpy(np.array([True], dtype=bool))
    outs = [InferRequestedOutput("text_output")]

    q, done = [], threading.Event()

    def on_resp(result, error):
        if result is None:
            done.set()
            return
        if error:
            logger.info(f"[ERR] Triton 오류: {error}")
            done.set()
            return
        arr = result.as_numpy("text_output")
        if arr is not None and len(arr) > 0:
            txt = arr[0].decode("utf-8")
            if txt:
                q.append(txt)

    cli.start_stream(callback=on_resp)
    cli.async_stream_infer(MODEL_NAME, inputs=[text, sparams, stream_flag], outputs=outs)

    try:
        start = time.time()
        last = start
        got_first = False

        while not done.is_set() or q:
            if q:
                chunk = q.pop(0)
                got_first = True
                last = time.time()
                yield chunk
            else:
                now = time.time()
                # 첫 토큰 기다리는 중
                if not got_first and now - start > first_token_timeout:
                    logger.info("[WARN] first token timeout")
                    break
                # 첫 토큰 이후 idle
                if got_first and now - last > idle_timeout:
                    logger.info("[WARN] idle timeout after first token")
                    break
                time.sleep(0.01)
    finally:
        cli.stop_stream()


def triton_infer(
        prompt: str,
        *,
        stream: bool = True,
        max_tokens: int = MAX_TOKENS,
        stop: List[str] | None = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        timeout_first=20, timeout_idle=5
):
    """decoupled 모델 호환: stream=True만 사용하고, sync는 스트림을 모아 문자열 반환."""
    logger.info(f"\n[DEBUG] 🚀 triton_infer 호출 - stream={stream}, max_tokens={max_tokens}, len={len(prompt)}")

    text, sparams = _make_inputs(
        prompt,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
    )

    gen = _triton_stream_generator(
        prompt,
        text,
        sparams,
        first_token_timeout=timeout_first,
        idle_timeout=timeout_idle,
    )
    if stream:
        # ❗여기서 절대 yield / yield from 쓰지 말 것
        return gen                 # 제너레이터 '객체'를 반환 (함수 자체는 일반 함수)
    else:
        # pseudo-sync: 스트림 결과 모아서 문자열 반환
        return "".join(list(gen))



# ---------------------------
# RAG 빌드
# ---------------------------

def build_rag_objects():
    global _qdr, _emb, _retriever
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    _qdr = QdrantClient(host=QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
    _emb = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda", embed_batch_size=32, trust_remote_code=True)
    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=_emb)
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)
    return _qdr, _emb, _retriever


# ---------------------------
# 질의 확장
# ---------------------------

def dynamic_expand_query_llm(query: str) -> List[str]:
    prompt = f"""
You are a scientific keyword generator for academic search.
Respond ONLY with a JSON array of 8 concise English keywords.
Do NOT include explanations, examples, or formatting outside the array.

Input: {query}
Output:
""".strip()

    resp = triton_infer(
        prompt,
        stream=False,       # pseudo-sync
        max_tokens=64,      # 살짝 여유
        # stop=["]"],       # 굳이 안 써도 됨. 원하면 이 정도만.
        temperature=0.3,
        top_p=0.9,
        timeout_first=30, timeout_idle=10
    ) or ""

    resp = _ensure_text(resp).strip()

    match = re.search(r"\[[^\]]*\]", resp, re.S)
    if match:
        json_text = match.group(0)
        try:
            kws = json.loads(json_text)
            return [k.strip() for k in kws if isinstance(k, str) and k.strip()][:10]
        except json.JSONDecodeError:
            pass

    parts = re.split(r"[,;/\n]", resp)
    kws = [re.sub(r"[^A-Za-z0-9\s\-]", "", p).strip() for p in parts]
    kws = [k for k in kws if 2 <= len(k) <= 40 and re.search(r"[A-Za-z]", k)]
    return sorted(set(kws))[:10]


def expand_query_kor(query: str) -> Tuple[str, List[str]]:
    terms = dynamic_expand_query_llm(query)
    expanded = " OR ".join(sorted(set(terms or [query])))
    return expanded, sorted(set(terms))


# ---------------------------
# 검색 + 재랭킹
# ---------------------------

def _safe_query_embedding(emb, text: str):
    try:
        vec = emb.get_query_embedding(text)
    except Exception:
        vec = emb.get_text_embedding(text)
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).tolist()


def dense_retrieve_hybrid(client: QdrantClient, emb, expanded_text: str, keywords: List[str], top_k=TOP_K_BASE):
    q_vec = _safe_query_embedding(emb, expanded_text)
    hits = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True,
        timeout=3  # 1초 제한
    ).points
    return hits


def expand_variants(keywords: List[str]) -> List[str]:
    variants = set()
    for k in keywords:
        if not k:
            continue
        variants.add(k)
        if not k.endswith("s"):
            variants.add(k + "s")
        if k.endswith("y") and len(k) > 1:
            variants.add(k[:-1] + "ies")
    return sorted(variants)


# ---------------------------
# 부스팅 + 재랭킹
# ---------------------------

def _payload_texts(payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    _node_text : 항상 '순수 텍스트'
    _node_content : JSON 구조 (선택적)
    """
    body = ""
    title = payload.get("_title", "") or ""

    # 1) _node_text 우선 (순수 텍스트로 저장된다고 가정)
    if isinstance(payload.get("_node_text"), str):
        body = payload["_node_text"].strip()

    # 2) _node_content 가 JSON일 경우, 필요 시 merge
    node_content = payload.get("_node_content")
    if node_content:
        try:
            node = json.loads(node_content) if isinstance(node_content, str) else node_content
            text2 = node.get("text", "")
            if text2 and len(text2) > len(body):
                body = text2.strip()

            # title fallback
            meta_title = node.get("metadata", {}).get("title")
            if not title and meta_title:
                title = meta_title.strip()
        except Exception:
            pass  # node_content가 깨져 있으면 무시

    # 3) title이 끝까지 없다면 body 앞 60자로 fallback
    if not title:
        title = (body[:60] + "...") if body else "Untitled"

    return body, title


def _keyword_score_for_hit(payload: Dict[str, Any], keywords: List[str]) -> float:
    body, title = _payload_texts(payload)
    if not body and not title:
        return 0.0
    best = 0.0
    for kw in keywords:
        if not kw:
            continue
        if title:
            s = fuzz.partial_ratio(kw, title)
            if s >= FUZZ_MIN:
                best = max(best, s * 1.2)
        if body:
            s = fuzz.partial_ratio(kw, body)
            if s >= FUZZ_MIN:
                best = max(best, s)
    return best / 200.0


def keyword_boost(hits, keywords: List[str]) -> Dict[str, float]:
    boost = {}
    for h in hits:
        try:
            b = _keyword_score_for_hit(h.payload or {}, keywords)
        except Exception:
            b = 0.0
        boost[h.id] = b
    return boost


def rrf_rerank(hits, boost_map: Dict[str, float], k=60):
    scored, id2hit = {}, {}
    for rank, h in enumerate(hits, start=1):
        id2hit[h.id] = h
        base = 1.0 / (k + rank)
        qdr = float(getattr(h, "score", 0.0) or 0.0)
        boost = boost_map.get(h.id, 0.0)
        scored[h.id] = scored.get(h.id, 0.0) + base + (qdr * 0.15) + boost
    reranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [id2hit[i] for i, _ in reranked]


def dedup_by_doc(hits, max_k=TOP_K_RETURN):
    seen, out = set(), []
    for h in hits:
        payload = h.payload or {}
        doc_id = payload.get("paper_id") or payload.get("doc_id") or payload.get("document_id") or payload.get("ref_doc_id")
        if not doc_id:
            doc_id = h.id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(h)
        if len(out) >= max_k:
            break
    return out


# ---------------------------
# 컨텍스트 + 프롬프트
# ---------------------------

def clamp_text(s, max_chars=SNIPPET_MAX_CHARS):
    # float, int, None, dict 등 전부 string으로 강제
    if not isinstance(s, str):
        s = str(s)

    s = re.sub(r"\s+", " ", s).strip()
    return s[:int(max_chars)]

def build_context_and_refs(hits) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    RAG context builder
    _node_text: 항상 순수 텍스트
    _node_content: JSON (optional)
    """
    items, refs = [], []

    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}

        # 1) ID 추출
        pid = payload.get("paper_id") or payload.get("doc_id") or payload.get("document_id") or payload.get("ref_doc_id") or "unknown"

        # 2) 텍스트/제목 통합
        body, title = _payload_texts(payload)

        # 3) 본문 클램프
        body = clamp_text(body, SNIPPET_MAX_CHARS)

        # 4) title은 _payload_texts에서 이미 생성됨
        items.append(f"[{i}] {title}\n{body}")
        refs.append((i, title.strip(), str(pid)))

    return "\n\n".join(items), refs


def token_len(s: str) -> int:
    try:
        return len(tokenizer.encode(s))
    except Exception:
        return math.ceil(len((s or "")) / 3)


def trim_context_to_budget(ctx: str, budget=CTX_TOKEN_BUDGET) -> str:
    if token_len(ctx) <= budget:
        return ctx
    paras = ctx.split("\n\n")
    kept, total = [], 0
    for p in paras:
        t = token_len(p) + 2
        if total + t > budget:
            break
        kept.append(p)
        total += t
    return "\n\n".join(kept)


def build_rag_prompt(context_text, query, refs):
    if refs:
        ref_lines = "\n".join([f"[{n}] {title} (ID={pid})" for n, title, pid in refs])
    else:
        ref_lines = "N/A"

    system_msg = (
        "당신은 사용자를 보조하는 LLM입니다. 반드시 제공된 컨텍스트에서만 근거를 사용하세요. "
        "컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다'라고 말하세요."
    )
    user_msg = f"""다음은 관련 문서 발췌입니다:

{context_text}

출처:
{ref_lines}

질문: {query}

요구사항:
- 문장 내 [1], [2] 형태의 근거 각주 달기
- 근거 외 지식 사용 금지
- 마지막 줄에 '참고문헌: [1] 제목A, [2] 제목B'
"""
    try:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{system_msg}\n</s>\n<|user|>\n{user_msg}\n</s>\n<|assistant|>\n"


def build_chat_prompt(query: str) -> str:
    sys = "당신은 친절하고 간결한 어시스턴트입니다."
    try:
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": query},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{sys}\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"


# ---------------------------
# 게이트 판단
# ---------------------------

def should_use_rag(query: str, hits, kw_list: List[str]) -> bool:
    if not hits:
        return False
    max_score = max([float(getattr(h, "score", 0.0) or 0.0) for h in hits])
    if max_score < SCORE_THRESHOLD:
        return False
    casual_patterns = ["날씨", "기분", "안녕", "좋아", "이름", "몇 시", "누구", "심심", "ㅎ", "사랑", "고마워"]
    if any(re.search(re.escape(p), query) for p in casual_patterns):
        return False
    english_ratio = sum(1 for k in kw_list if re.search(r"[A-Za-z]", k)) / (len(kw_list) or 1)
    if english_ratio < 0.4:
        return False
    return True


def _ensure_text(x):
    # Triton streaming generator → 수집 후 string
    if isinstance(x, str):
        return x
    if hasattr(x, "__iter__") and not isinstance(x, (bytes, dict, list, tuple, np.ndarray)):
        try:
            return "".join(list(x))
        except Exception:
            return str(x)
    return str(x)

def decide_rag_needed(query: str) -> bool:
    prompt = (
        "Classify the query.\n"
        "Return only ONE word: RAG or CHAT.\n\n"
        f"Query: {query}\n"
        "Answer:"
    )

    resp = triton_infer(
        prompt,
        stream=False,          # 내부 decoupled → stream 기반 join
        max_tokens=16,
        stop=None,             # ❗ 중단 조건 절대 쓰지 않기
        temperature=0.0,
        top_p=1.0,
        timeout_first=30, timeout_idle=10
    )

    resp = _ensure_text(resp).strip()
    logger.info("========== decide_rag_needed() RAW RESPONSE ==========")
    logger.info(repr(resp))
    logger.info("=======================================================")

    text = resp.strip().lower()

    if "rag" in text[:5]:
        return True
    if "chat" in text[:5]:
        return False

    return False

def rag_gate_decision(query: str, hits, kw_list: List[str], need_rag: bool) -> Tuple[bool, str]:
    if not hits:
        return False, "❌ 검색 결과 없음 → Chat 전환."
    elif max(float(h.score or 0.0) for h in hits) < SCORE_THRESHOLD:
        return False, "⚠️ 검색 스코어 낮음 → Chat 전환."
    gate_ok = should_use_rag(query, hits, kw_list)
    if not (need_rag and gate_ok):
        return False, "🤖 게이트 판단 결과: 일반 대화 유지."
    return True, "✅ 게이트 판단 결과: RAG 수행."


# ---------------------------
# 콘솔 테스트 루프
# ---------------------------

def main():
    qdr, emb, retriever = build_rag_objects()
    logger.info("✅ RAG pipeline ready\n")

    while True:
        try:
            query = input("질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"exit", "quit"}:
            break

        need_rag = decide_rag_needed(query)
        logger.info(f"🧭 판단 결과: {'RAG' if need_rag else 'CHAT'}")

        expanded_text, kw_list = expand_query_kor(query)
        keywords = expand_variants(kw_list)
        hits = dense_retrieve_hybrid(qdr, emb, expanded_text, keywords)

        ok, msg = rag_gate_decision(query, hits, kw_list, need_rag)
        logger.info(msg)

        if not ok:
            prompt = build_chat_prompt(query)
            logger.info("\n📘 답변:")
            for chunk in triton_infer(prompt, stream=True, max_tokens=MAX_TOKENS):
                logger.info(chunk.rstrip("\n"))
            logger.info("\n" + "-" * 80)
            continue

        boost_map = keyword_boost(hits, kw_list)
        reranked = rrf_rerank(hits, boost_map)
        final_hits = dedup_by_doc(reranked)
        ctx, refs = build_context_and_refs(final_hits)
        ctx = trim_context_to_budget(ctx, budget=CTX_TOKEN_BUDGET)

        prompt = build_rag_prompt(ctx, query, refs)
        logger.info("\n📚 RAG 답변:")
        for chunk in triton_infer(prompt, stream=True, max_tokens=MAX_TOKENS):
            logger.info(chunk.rstrip("\n"))
        logger.info("\n참고문헌:", ", ".join([f"[{n}] {title}" for n, title, _ in refs]))
        logger.info("\n" + "-" * 80)


if __name__ == "__main__":
    main()
