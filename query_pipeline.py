# 03_query_pipeline.py
# pip install "tritonclient[grpc]" qdrant-client transformers numpy llama-index tqdm orjson rapidfuzz

import os, json, time, threading, math, re
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from transformers import AutoTokenizer
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

from rapidfuzz import fuzz

# ---------------------------
# 환경 설정
# ---------------------------
QDRANT_HOST  = os.getenv("QDRANT_HOST", "211.241.177.73")  # 권장: host + grpc_port
QDRANT_URL   = os.getenv("QDRANT_URL", "http://211.241.177.73:6333")  # (예비)
COLLECTION   = os.getenv("QDRANT_COLLECTION", "peS2o_rag")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 1024-d
TRITON_URL   = os.getenv("TRITON_URL", "211.241.177.73:8001")
MODEL_NAME   = os.getenv("TRITON_MODEL", "gemma_vllm_0")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "./")

TOP_K_BASE = 20
TOP_K_RETURN = 20
MAX_TOKENS    = 1024
TEMPERATURE   = 0.6
TOP_P         = 0.9

SCORE_THRESHOLD = 0.15
FUZZ_MIN        = 40

CTX_TOKEN_BUDGET = 2200
SNIPPET_MAX_CHARS = 1800

# ---------------------------
# Triton LLM
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

def triton_infer(prompt: str, stream=False) -> str:
    cli = InferenceServerClient(url=TRITON_URL, verbose=False)
    if not cli.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Triton model not ready: {MODEL_NAME}")

    text = InferInput("text_input", [1], "BYTES")
    text.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    sparams = InferInput("sampling_parameters", [1], "BYTES")
    sparams.set_data_from_numpy(np.array([
        json.dumps({"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS}).encode("utf-8")
    ], dtype=object))

    stream_flag = InferInput("stream", [1], "BOOL")
    stream_flag.set_data_from_numpy(np.array([stream], dtype=bool))
    outs = [InferRequestedOutput("text_output")]

    done = threading.Event()
    acc_text = []

    def on_resp(result, error):
        if error:
            print("[ERR]", error)
            done.set(); return
        txt = result.as_numpy("text_output")[0].decode("utf-8")
        acc_text.append(txt)
        done.set()

    cli.start_stream(callback=on_resp)
    cli.async_stream_infer(MODEL_NAME, inputs=[text, sparams, stream_flag], outputs=outs)
    done.wait(timeout=180)
    cli.stop_stream()
    return "".join(acc_text).strip()

# ---------------------------
# RAG 빌드
# ---------------------------
def build_rag_objects():
    qdr = QdrantClient(host=QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
    emb = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda", embed_batch_size=128, trust_remote_code=True)
    vstore = QdrantVectorStore(client=qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=emb)
    retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)
    return qdr, emb, retriever

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
    """

    resp = triton_infer(prompt).strip()

    #JSON 배열 부분만 추출 (lazy match)
    match = re.search(r"\[[^\]]*\]", resp, re.S)
    if match:
        json_text = match.group(0)
        try:
            kws = json.loads(json_text)
            return [k.strip() for k in kws if isinstance(k, str) and k.strip()][:10]
        except json.JSONDecodeError:
            pass

    # fallback: 쉼표 기반 파싱
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
    ).points

    return hits

def expand_variants(keywords: List[str]) -> List[str]:
    variants = set()
    for k in keywords:
        variants.add(k)
        if not k.endswith("s"):
            variants.add(k + "s")
        if k.endswith("y"):
            variants.add(k[:-1] + "ies")
    return sorted(variants)

# ---------------------------
# 부스팅 + 재랭킹
# ---------------------------
def _payload_texts(payload: Dict[str, Any]) -> Tuple[str, str]:
    node_json = payload.get("_node_text")
    body, title = "", payload.get("_title", "")
    if node_json:
        try:
            node = json.loads(node_json)
            body = node.get("text", "") or ""
            if not title:
                title = node.get("metadata", {}).get("title", "") or ""
        except Exception:
            pass
    body2 = payload.get("_node_text", "")
    if body2 and len(body2) > len(body):
        body = body2
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
        # ✅ paper_id를 기본 식별자로 사용
        doc_id = payload.get("paper_id") or payload.get("doc_id") or payload.get("document_id") or payload.get("ref_doc_id")
        if not doc_id:
            doc_id = h.id  # fallback
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(h)
        if len(out) >= max_k:
            break
    return out

# ---------------------------
# 컨텍스트 구성
# ---------------------------
def clamp_text(s: str, max_chars=SNIPPET_MAX_CHARS) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]

def build_context_and_refs(hits) -> Tuple[str, List[Tuple[int, str, str]]]:
    items, refs = [], []
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        text = ""
        title = payload.get("_title", "")
        pid = payload.get("paper_id") or payload.get("doc_id") or "unknown"

        # ✅ 1️⃣ 기본: _node_text를 우선 사용
        if "_node_text" in payload and payload["_node_text"]:
            text = payload["_node_text"]

        # ✅ 2️⃣ 예외적으로 _node_content가 존재하는 경우 (JSON 구조 지원)
        elif "_node_content" in payload:
            try:
                node = json.loads(payload["_node_content"])
                text = node.get("text", "")
                if not title:
                    title = node.get("metadata", {}).get("title", "") or ""
            except Exception:
                pass

        # ✅ 3️⃣ 정리 및 클램프
        text = clamp_text(text, SNIPPET_MAX_CHARS)
        if not title:
            title = text[:50] + "..."

        items.append(f"[{i}] {title}\n{text}")
        refs.append((i, title.strip(), str(pid)))

    return "\n\n".join(items), refs

def token_len(s: str) -> int:
    try:
        return len(tokenizer.encode(s))
    except Exception:
        return math.ceil(len(s) / 3)

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

# ---------------------------
# 프롬프트 (RAG/대화 모드 분리)
# ---------------------------
def build_rag_prompt(context_text, query, refs):
    """
    refs: (번호, 제목, paper_id)
    """
    if refs:
        ref_lines = "\n".join([f"[{n}] {title} (ID={pid})" for n, title, pid in refs])
    else:
        ref_lines = "N/A"

    system_msg = (
        "당신은 사용자를 보조하는 LLM입니다. 반드시 제공된 컨텍스트에서만 근거를 사용하세요. "
        "컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다'라고만 말하고, 추측하지 마세요. "
        "가능하면 문장 내에 근거 번호 각주를 표시하세요."
    )
    user_msg = f"""다음은 관련 문서 발췌입니다(번호=출처):

{context_text}

(출처 번호 매핑)
{ref_lines}

질문: {query}

요구사항:
- 문장 내에 [1], [2] 형태로 근거 번호를 달아주세요.
- 컨텍스트에 없는 내용은 쓰지 마세요(추가 지식 금지).
- 마지막 줄에 '참고문헌:' 뒤에 논문 제목을 함께 나열하세요. 예시: 참고문헌: [1] 제목A, [2] 제목B
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
    sys = "당신은 친절하고 간결한 어시스턴트입니다. 사용자의 일상 질문에 자연스럽게 답하세요."
    try:
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": query},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{sys}\n</s>\n<|user|>\n{query}\n</s>\n<|assistant|>\n"

def should_use_rag(query: str, hits, kw_list: List[str]) -> bool:
    if not hits:
        return False

    # 점수 체크
    max_score = max([float(getattr(h, "score", 0.0) or 0.0) for h in hits])
    if max_score < SCORE_THRESHOLD:
        return False

    # 질의 유형 체크 (단순 대화 배제)
    casual_patterns = [
        "날씨", "기분", "안녕", "좋아", "이름", "몇 시", "누구", "심심", "배고파",
        "오늘", "어때", "ㅋㅋ", "ㅎ", "??", "잘자", "사랑", "고마워", "ㅎㅇ"
    ]

    if any(re.search(re.escape(p), query) for p in casual_patterns):
        return False

    # 키워드 품질 체크 (영문 학술 키워드 비율)
    english_ratio = sum(1 for k in kw_list if re.search(r"[A-Za-z]", k)) / (len(kw_list) or 1)
    if english_ratio < 0.4:
        return False

    return True

def decide_rag_needed(query: str) -> bool:
    prompt = f"""
    You are a controller that decides whether to use RAG (vector database search).
    If the user asks for factual, technical, or academic information, return "RAG".
    If the user asks for casual talk or opinion, return "CHAT".
    Only output one word: RAG or CHAT.

    User query: {query}
    Output:
    """
    resp = triton_infer(prompt).strip().upper()
    return "RAG" in resp

def rag_gate_decision(query: str, hits, kw_list: List[str], need_rag: bool) -> tuple[bool, str]:
    # 휴리스틱 필터 (스코어, 일반 대화 감지 등)
    gate_ok = should_use_rag(query, hits, kw_list)
    msg = ""

    # 결과 로그
    if not hits:
        msg = "❌ RAG 시도했으나 결과 없음 → fallback to chat."
        print(msg)
        return False, msg
    elif max(float(h.score or 0.0) for h in hits) < SCORE_THRESHOLD:
        msg = "⚠️ 검색 스코어 낮음 → fallback to chat."
        print(msg)
        return False, msg
    elif not (need_rag and gate_ok):
        msg = "🤖 게이트 판단 결과: 일반 대화 모드 유지."
        print(msg)
        return False, msg

    msg = "✅ 게이트 판단 결과: RAG 검색/응답 수행."
    print(msg)
    return True, msg

# ---------------------------
# 메인 루프
# ---------------------------
def main():
    qdr, emb, retriever = build_rag_objects()
    print("✅ LLM-decides-RAG pipeline ready\n")

    while True:
        query = input("질문 > ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        # LLM이 판단
        need_rag = decide_rag_needed(query)
        print(f"🧭 LLM 판단 결과: {'RAG 검색 수행' if need_rag else '일반 대화'}")

        # RAG 검색
        expanded_text, kw_list = expand_query_kor(query)
        keywords = expand_variants(kw_list)
        print(keywords);
        hits = dense_retrieve_hybrid(qdr, emb, expanded_text, keywords, top_k=TOP_K_BASE)

        #  게이트 판단
        if not rag_gate_decision(query, hits, kw_list, need_rag):
            chat_prompt = build_chat_prompt(query)
            answer = triton_infer(chat_prompt, stream=True)
            print("\n📘 답변:"); print(answer.strip()); print("-" * 80)
            continue

        # 통과 시 RAG 수행
        boost_map = keyword_boost(hits, kw_list)
        reranked = rrf_rerank(hits, boost_map, k=60)
        final_hits = dedup_by_doc(reranked, max_k=TOP_K_RETURN)

        ctx, refs = build_context_and_refs(final_hits)
        ctx = trim_context_to_budget(ctx, budget=CTX_TOKEN_BUDGET)

        print(f"\n🔎 Retrieved top-{len(final_hits)} (after RRF+dedup).")
        rag_prompt = build_rag_prompt(ctx, query, refs)
        print("⚡ LLM generating response...")
        answer = triton_infer(rag_prompt, stream=True)
        print("\n📘 답변:"); print(answer.strip()); print("-" * 80)

if __name__ == "__main__":
    main()
