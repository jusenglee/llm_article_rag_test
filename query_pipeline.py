# ==============================================
# 03_query_pipeline_fixed_final.py
# 최적화 버전 (Client Reuse + Dual RAG Stack)
# ==============================================
# pip install "tritonclient[grpc]" qdrant-client transformers numpy llama-index tqdm orjson rapidfuzz

import os
import json
import time
import threading
import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
# llama_index 관련 임포트
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from transformers import AutoTokenizer
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from rapidfuzz import fuzz
from dataclasses import dataclass
# 로깅 설정
logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAG_Pipeline")

# ---------------------------
# 환경 설정
# ---------------------------
QDRANT_HOST  = os.getenv("QDRANT_HOST", "211.241.177.73")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6334))  # GRPC Port
COLLECTION   = os.getenv("QDRANT_COLLECTION", "peS2o_rag")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

TRITON_URL        = os.getenv("TRITON_URL", "211.241.177.73:8001")
DEFAULT_MODEL_NAME = os.getenv("TRITON_MODEL", "gpt_oss_0")  # 기본 LLM 이름
TOKENIZER_MAP = {
    "gpt_oss_0": "openai/gpt-oss-20b",      # gpt-oss용 경로/HF ID
    "gemma_vllm_0": "./data/gemma3",    # 지금 사용하던 Gemma3 토커나이저
}

# === B 스택용 설정 (두 번째 임베딩 + 벡터DB) ===
QDRANT_HOST_B  = os.getenv("QDRANT_HOST_B", QDRANT_HOST)
QDRANT_PORT_B  = int(os.getenv("QDRANT_PORT_B", QDRANT_PORT))
COLLECTION_B   = os.getenv("QDRANT_COLLECTION_B", "e5_rag")
EMBED_MODEL_B  = os.getenv("EMBEDDING_MODEL_B", "intfloat/multilingual-e5-large")

# 하이퍼파라미터
TOP_K_BASE      = 20    # 검색 후보군
TOP_K_RETURN    = 20    # 최종 반환 개수
MAX_TOKENS      = 8192
TEMPERATURE     = 0.6
TOP_P           = 0.9

SCORE_THRESHOLD = 0.15
FUZZ_MIN        = 40

CTX_TOKEN_BUDGET = 4096
SNIPPET_MAX_CHARS = 4096


@dataclass
class RagResult:
    stack: str            # "A" or "B"
    expanded_query: str
    keywords: List[str]
    hits: Any             # Qdrant 결과 (points)
    reranked_hits: Any
    context: str
    refs: List[str]
    timings: Dict[str, float]
    llm_answer: str | None = None

# ---------------------------
# 전역 리소스 캐시
# ---------------------------
_qdr, _emb, _retriever = None, None, None        # A 스택
_qdr2, _emb2, _retriever2 = None, None, None     # B 스택
_triton_client = None

_tokenizers: Dict[str, AutoTokenizer] = {}

def get_tokenizer_for_model(model_name: str) -> AutoTokenizer:
    if model_name not in _tokenizers:
        tok_id = TOKENIZER_MAP[model_name]
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(
            tok_id, trust_remote_code=True
        )
    return _tokenizers[model_name]

ASSISTANT_FINAL_MARKER = "assistantfinal"
def extract_final_answer(raw: str) -> str:
    """gpt-oss가 analysis/assistantfinal 포맷으로 뱉을 때, 최종 답변만 추출."""
    if not raw:
        return ""

    text = str(raw).strip()
    idx = text.rfind(ASSISTANT_FINAL_MARKER)
    if idx == -1:
        # 마커 없으면 그냥 원본 반환 (다른 모델 대비 안전장치)
        return text

    final = text[idx + len(ASSISTANT_FINAL_MARKER):]
    # 콜론/공백 정리
    final = final.lstrip(" :\n\t")
    logger.info(final)
    return final.strip()
# ---------------------------
# Triton Client & LLM Core
# ---------------------------

def get_triton_client() -> InferenceServerClient:
    """Triton Client Singleton"""
    global _triton_client
    if _triton_client is None:
        try:
            _triton_client = InferenceServerClient(url=TRITON_URL, verbose=False)
            logger.info(f"✅ Triton Client connected to {TRITON_URL}")
        except Exception as e:
            logger.error(f"❌ Triton Client connection failed: {e}")
            raise e
    return _triton_client


def _make_inputs(
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
):
    text = InferInput("text_input", [1], "BYTES")
    text.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    # vLLM-backend 권장 명칭: sampling_parameters
    sparams = InferInput("sampling_parameters", [1], "BYTES")

    # vLLM Python backend 쪽 구현이 문자열 기반 파싱을 사용하는 경우가 많음
    params = {
        "temperature": str(float(temperature)),
        "top_p": str(float(top_p)),
        "max_tokens": str(int(max_tokens)),
        # stream 플래그는 별도 BOOL 인풋으로 전달 중
    }

    sparams.set_data_from_numpy(
        np.array([json.dumps(params).encode("utf-8")], dtype=object)
    )
    return text, sparams


def _triton_stream_generator(
        model_name: str,
        prompt: str,
        text: InferInput,
        sparams: InferInput,
        first_token_timeout: int = 10,
        idle_timeout: int = 20,
):
    cli = get_triton_client()  # Triton client 재사용

    stream_flag = InferInput("stream", [1], "BOOL")
    stream_flag.set_data_from_numpy(np.array([True], dtype=bool))
    outs = [InferRequestedOutput("text_output")]

    q: List[str] = []
    done = threading.Event()

    def on_resp(result, error):
        if error:
            logger.error(f"[ERR] Triton Callback Error: {error}")
            done.set()
            return

        if result is None:
            done.set()
            return

        arr = result.as_numpy("text_output")
        if arr is not None and len(arr) > 0:
            raw = arr[0]
            chunk = raw.decode("utf-8", errors="ignore")
            q.append(chunk)

        is_final = False
        try:
            resp = result.get_response()
            params = getattr(resp, "parameters", None)
            if params:
                flag = params.get("triton_final_response")
                if flag and getattr(flag, "bool_param", False):
                    is_final = True
        except Exception as e:
            logger.debug(f"[STREAM] triton_final_response check failed: {e}")

        if is_final:
            done.set()
            return

    # 스트림 시작
    cli.start_stream(callback=on_resp)
    cli.async_stream_infer(model_name, inputs=[text, sparams, stream_flag], outputs=outs)

    try:
        start_time = time.time()
        last_yield_time = start_time
        got_first = False

        while not done.is_set() or q:
            if q:
                chunk = q.pop(0)
                got_first = True
                last_yield_time = time.time()
                yield chunk
            else:
                now = time.time()
                if not got_first and (now - start_time > first_token_timeout):
                    logger.warning("[WARN] First token timeout")
                    break
                if got_first and (now - last_yield_time > idle_timeout):
                    logger.warning("[WARN] Idle timeout after response started")
                    break
                time.sleep(0.005)
    finally:
        cli.stop_stream()


def _triton_infer_sync(
        model_name: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
) -> str:
    text, sparams = _make_inputs(
        prompt, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, stream=True
    )

    accumulated_text = ""
    for chunk in _triton_stream_generator(model_name, prompt, text, sparams, 10, 20):
        accumulated_text += chunk

    return accumulated_text.strip()


def triton_infer(
        model_name: str,
        prompt: str,
        *,
        stream: bool = True,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        timeout_first: int = 20,
        timeout_idle: int = 120,
):
    logger.info(f"[TRITON] infer start - model={model_name}, len={len(prompt)}")

    if stream:
        text, sparams = _make_inputs(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, stream=True
        )
        return _triton_stream_generator(
            model_name, prompt, text, sparams,
            timeout_first, timeout_idle
        )

    # Sync Path
    return _triton_infer_sync(
        model_name,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

def ensure_single_model_loaded(target_model: str, timeout: float = 120.0) -> None:
    """
    1) 레포지토리 인덱스를 보고 target 이외 모델은 모두 unload
    2) target 모델이 READY 상태가 아니면 load + READY 될 때까지 대기
    """
    cli = get_triton_client()

    # 1. 모델 레포지토리 인덱스 조회
    try:
        repo = cli.get_model_repository_index()
    except Exception as e:
        logger.error(f"[TRITON] get_model_repository_index failed: {e}")
        raise

    # 2. target 외 모델 unload
    for m in getattr(repo, "models", []):
        name = getattr(m, "name", None)
        if not name or name == target_model:
            continue
        try:
            if cli.is_model_ready(name):
                logger.info(f"[TRITON] unloading other model: {name}")
                cli.unload_model(name)
        except Exception as e:
            logger.warning(f"[TRITON] unload_model({name}) failed: {e}")

    # 3. target 이 이미 READY면 바로 리턴
    try:
        if cli.is_model_ready(target_model):
            logger.info(f"[TRITON] target model {target_model} already READY")
            return
    except Exception as e:
        logger.warning(f"[TRITON] is_model_ready({target_model}) error: {e}")

    # 4. target load
    logger.info(f"[TRITON] loading model: {target_model}")
    cli.load_model(target_model)

    # 5. READY 될 때까지 polling
    start = time.time()
    while True:
        try:
            if cli.is_model_ready(target_model):
                logger.info(f"[TRITON] model {target_model} READY")
                return
        except Exception as e:
            logger.warning(f"[TRITON] is_model_ready({target_model}) check failed: {e}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout while waiting for model {target_model} to be READY")

        time.sleep(0.5)


def unload_model_safe(model_name: str) -> None:
    cli = get_triton_client()
    try:
        if cli.is_model_ready(model_name):
            logger.info(f"[TRITON] unloading model: {model_name}")
            cli.unload_model(model_name)
    except Exception as e:
        logger.warning(f"[TRITON] unload_model({model_name}) failed: {e}")
# ---------------------------
# RAG Utils & Objects
# ---------------------------

def build_rag_objects():
    """
    A 스택만 초기화 (기존 단일 스택용).
    듀얼 스택이 필요하면 아래 build_rag_objects_dual() 사용.
    """
    global _qdr, _emb, _retriever
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    # Qdrant Client (A)
    _qdr = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)

    # Embedding Model (A)
    _emb = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True
    )

    # Retriever 설정 (A)
    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=_emb)
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton Client도 미리 준비
    get_triton_client()

    return _qdr, _emb, _retriever


def build_rag_objects_dual():
    """
    A/B 두 개의 RAG 스택을 모두 초기화해서 돌려줌.
    A: (QDRANT_HOST,  COLLECTION,  EMBED_MODEL)
    B: (QDRANT_HOST_B,COLLECTION_B,EMBED_MODEL_B)
    """
    global _qdr, _emb, _retriever, _qdr2, _emb2, _retriever2

    # A 스택 먼저 초기화 (기존 로직 재사용)
    qdr, emb, retriever = build_rag_objects()

    # B 스택이 이미 준비되어 있으면 그대로 반환
    if _qdr2 and _emb2 and _retriever2:
        return qdr, emb, retriever, _qdr2, _emb2, _retriever2

    # Qdrant Client (B)
    _qdr2 = QdrantClient(host=QDRANT_HOST_B, grpc_port=QDRANT_PORT_B, prefer_grpc=True)

    # Embedding Model (B)
    _emb2 = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_B,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True
    )

    # Retriever 설정 (B)
    vstore2 = QdrantVectorStore(client=_qdr2, collection_name=COLLECTION_B)
    sctx2 = StorageContext.from_defaults(vector_store=vstore2)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vstore2, embed_model=_emb2)
    _retriever2 = index2.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton은 공용
    get_triton_client()

    return qdr, emb, retriever, _qdr2, _emb2, _retriever2


def clamp_text(s, max_chars=SNIPPET_MAX_CHARS):
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:int(max_chars)]


def _payload_texts(payload: Dict[str, Any]) -> Tuple[str, str]:
    body = ""
    title = payload.get("_title", "") or ""

    if isinstance(payload.get("_node_text"), str):
        body = payload["_node_text"].strip()

    node_content = payload.get("_node_content")
    if node_content:
        try:
            node = json.loads(node_content) if isinstance(node_content, str) else node_content
            text2 = node.get("text", "")
            if text2 and len(text2) > len(body):
                body = text2.strip()

            meta_title = node.get("metadata", {}).get("title")
            if not title and meta_title:
                title = meta_title.strip()
        except Exception:
            pass

    if not title:
        title = (body[:60] + "...") if body else "Untitled"

    return body, title


# ---------------------------
# Query Expansion & Reranking
# ---------------------------

def dynamic_expand_query_llm(query: str) -> List[str]:
    prompt = f"""You are a scientific keyword generator.
Respond ONLY with a JSON array of 8 concise English keywords.
Input: {query}
Output: """

    raw = triton_infer(
        DEFAULT_MODEL_NAME,
        prompt,
        stream=False,
        max_tokens=64,
        temperature=0.3,
    )

    resp = extract_final_answer(raw)  # 🔴 CoT 제거

    try:
        # JSON 파싱 시도 (배열 찾기)
        match = re.search(r"\[.*?\]", resp, re.S)
        if match:
            return json.loads(match.group(0))[:10]
    except Exception:
        pass

    # Fallback: 쉼표/줄바꿈 분리
    parts = re.split(r"[,;\n]", resp)
    kws = [re.sub(r"[^A-Za-z0-9\s\-]", "", p).strip() for p in parts]
    return [k for k in kws if 2 <= len(k) <= 40][:10]


def expand_query_kor(query: str) -> Tuple[str, List[str]]:
    terms = dynamic_expand_query_llm(query)
    # 원본 쿼리 + 확장어
    all_terms = sorted(set(terms + [query]))
    expanded_query = " ".join(all_terms)
    return expanded_query, all_terms


def dense_retrieve_hybrid(
        client: QdrantClient,
        emb,
        expanded_text: str,
        keywords: List[str],
        collection_name: str,
):
    """
    A/B 어디든 동일하게 사용 가능하도록 collection_name 인자를 추가.
    """
    try:
        q_vec = emb.get_query_embedding(expanded_text)
    except Exception:
        q_vec = emb.get_text_embedding(expanded_text)

    hits = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=TOP_K_BASE,
        with_payload=True,
        timeout=3
    ).points
    return hits


def _keyword_score_for_hit(payload: Dict[str, Any], keywords: List[str]) -> float:
    body, title = _payload_texts(payload)

    # [최적화] Fuzzy matching 전, 텍스트 길이 제한 (속도 향상)
    body = clamp_text(body, max_chars=4096)

    if not body and not title:
        return 0.0

    best = 0.0
    for kw in keywords:
        if not kw:
            continue
        # 제목 가중치
        if title and fuzz.partial_ratio(kw, title) >= FUZZ_MIN:
            best = max(best, 80)  # 단순화된 점수
        # 본문
        if body and fuzz.partial_ratio(kw, body) >= FUZZ_MIN:
            best = max(best, 60)

    return best / 100.0  # 정규화 (0~1 범위)


def rrf_rerank(hits, keywords: List[str], k=60):
    scored: Dict[Any, float] = {}
    id2hit = {h.id: h for h in hits}

    # 키워드 부스팅 점수 미리 계산
    boost_map: Dict[Any, float] = {}
    for h in hits:
        boost_map[h.id] = _keyword_score_for_hit(h.payload or {}, keywords)

    for rank, h in enumerate(hits, start=1):
        rrf_score = 1.0 / (k + rank)
        vec_score = float(h.score) if h.score else 0.0
        boost = boost_map.get(h.id, 0.0)

        # 가중치 합산 (튜닝 포인트)
        final_score = rrf_score + (vec_score * 0.5) + (boost * 0.3)
        scored[h.id] = final_score

    # 정렬
    sorted_ids = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
    return [id2hit[iD] for iD in sorted_ids]


def build_context(hits):
    items, refs = [], []
    seen_ids = set()

    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        doc_id = payload.get("doc_id") or payload.get("paper_id") or str(h.id)

        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        body, title = _payload_texts(payload)
        body = clamp_text(body, SNIPPET_MAX_CHARS)

        # 🔴 디버그: 처음 몇 개만 찍어보자
        if i <= 3:
            logger.info(f"[DEBUG] RAW_TITLE[{i}]: {repr(title[:200])}")
            logger.info(f"[DEBUG] RAW_BODY[{i}]: {repr(body[:200])}")

        items.append(f"[{i}] {title}\n{body}")
        refs.append(f"[{i}] {title}")

        if len(items) >= TOP_K_RETURN:
            break

    return "\n\n".join(items), refs


# ---------------------------
# Main Logic (Gating & Execution)
# ---------------------------

def decide_rag_needed(query: str, model_name: str = DEFAULT_MODEL_NAME) -> bool:
    casual = ["안녕", "날씨", "이름", "뭐해", "반가워"]
    if any(c in query for c in casual):
        return False

    prompt = (
        "You are a classifier.\n"
        "Decide if the user query requires external factual knowledge "
        "such as scientific, technical, or domain-specific information.\n"
        "If YES, answer exactly 'YES'. If NO, answer exactly 'NO'.\n"
        f"Query: {query}\n"
        "Answer (YES or NO only):"
    )

    raw = triton_infer(model_name, prompt, stream=False, max_tokens=4)
    resp = extract_final_answer(raw).strip().upper()
    if "YES" in resp:
        return True
    if "NO" in resp:
        return False

    # 애매하면 보수적으로 RAG 사용
    return True

def run_rag_once(
        query: str,
        stack: str = "A",
        with_llm: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
) -> RagResult:
    """
    하나의 스택(A/B)에 대해:
    - 쿼리 확장
    - 임베딩 + Qdrant 검색
    - 리랭킹
    - 컨텍스트 빌드
    - (옵션) LLM 답변 생성
    전 과정을 실행하고, 타이밍/결과를 모두 담아 반환.
    """
    t_all0 = time.time()
    timings: Dict[str, float] = {}

    # 1) 스택별 객체 준비
    t0 = time.time()
    if stack == "A":
        qdr, emb, retriever, _, _, _ = build_rag_objects_dual()
        collection = COLLECTION
    else:
        qdr, emb, retriever, qdr2, emb2, retriever2 = build_rag_objects_dual()
        qdr, emb, retriever = qdr2, emb2, retriever2
        collection = COLLECTION_B
    timings["stack_init"] = time.time() - t0

    # 2) 쿼리 확장
    t0 = time.time()
    expanded_query, kws = expand_query_kor(query)
    timings["expand_query"] = time.time() - t0

    # 3) dense 검색
    t0 = time.time()
    hits = dense_retrieve_hybrid(
        client=qdr,
        emb=emb,
        expanded_text=expanded_query,
        keywords=kws,
        collection_name=collection,
    )
    timings["dense_search"] = time.time() - t0

    # 4) 리랭킹
    t0 = time.time()
    reranked = rrf_rerank(hits, kws)
    timings["rerank"] = time.time() - t0

    # 5) 컨텍스트 빌드
    t0 = time.time()
    context, refs = build_context(reranked)
    timings["build_context"] = time.time() - t0

    # 6) LLM 답변 생성 (선택)
    llm_answer = None
    if with_llm:
        t0 = time.time()
        prompt = (
            "당신은 과학/기술 문서를 기반으로 답변하는 한국어 LLM입니다.\n"
            "반드시 제공된 컨텍스트에서만 근거를 사용하세요.\n"
            "컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다'라고만 말하고, 추측하지 마세요.\n\n"
            f"[질문]\n{query}\n\n[컨텍스트]\n{context}\n"
        )
        raw = triton_infer(model_name, prompt, stream=False, max_tokens=1024)
        llm_answer = extract_final_answer(raw)
        timings["llm_answer"] = time.time() - t0

    timings["total"] = time.time() - t_all0

    return RagResult(
        stack=stack,
        expanded_query=expanded_query,
        keywords=kws,
        hits=hits,
        reranked_hits=reranked,
        context=context,
        refs=refs,
        timings=timings,
        llm_answer=llm_answer,
    )

def run_rag_ab_compare(
        query: str,
        with_llm: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, RagResult]:
    """
    같은 질문에 대해 A/B 스택을 모두 실행.
    레포트/로그/오프라인 분석에 쓰기 좋게 dict로 리턴.
    """
    res_a = run_rag_once(query, stack="A", with_llm=with_llm, model_name=model_name)
    res_b = run_rag_once(query, stack="B", with_llm=with_llm, model_name=model_name)

    return {"A": res_a, "B": res_b}


import pathlib

LOG_DIR = pathlib.Path(os.getenv("RAG_BENCH_LOG_DIR", "./rag_bench_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_ab_result_to_file(query: str, res_map: Dict[str, RagResult]):
    ts = int(time.time())
    for key, res in res_map.items():
        out = {
            "timestamp": ts,
            "stack": res.stack,           # "A" or "B"
            "embedding_model": EMBED_MODEL if res.stack == "A" else EMBED_MODEL_B,
            "query": query,
            "expanded_query": res.expanded_query,
            "keywords": res.keywords,
            "timings": res.timings,
            "refs": res.refs,
            "llm_answer": res.llm_answer,
            "top_hits": [
                {
                    "id": h.id,
                    "score": float(h.score) if h.score is not None else 0.0,
                    "payload_doc_id": (h.payload or {}).get("doc_id") or (h.payload or {}).get("paper_id"),
                }
                for h in res.reranked_hits[:TOP_K_RETURN]
            ],
        }
        fname = LOG_DIR / f"{ts}_{res.stack}.jsonl"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
