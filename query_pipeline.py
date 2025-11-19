# ==============================================
# 03_query_pipeline_fixed_final.py
# 최적화 버전 (Client Reuse + Robust EOS + Perf fix)
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

# 로깅 설정
logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAG_Pipeline")

# ---------------------------
# 환경 설정
# ---------------------------
QDRANT_HOST  = os.getenv("QDRANT_HOST", "211.241.177.73")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6334)) # GRPC Port
COLLECTION   = os.getenv("QDRANT_COLLECTION", "peS2o_rag")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

TRITON_URL   = os.getenv("TRITON_URL", "211.241.177.73:8001")
MODEL_NAME   = os.getenv("TRITON_MODEL", "gemma_vllm_0")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "./data/") # 로컬 경로 혹은 모델명

# 하이퍼파라미터
TOP_K_BASE = 20        # 검색 후보군 (넉넉하게 잡음)
TOP_K_RETURN = 20       # 최종 반환 개수
MAX_TOKENS    = 8192
TEMPERATURE   = 0.6
TOP_P         = 0.9

SCORE_THRESHOLD = 0.15
FUZZ_MIN        = 40

CTX_TOKEN_BUDGET = 4096
SNIPPET_MAX_CHARS = 4096

# ---------------------------
# 전역 리소스 캐시
# ---------------------------
_qdr, _emb, _retriever = None, None, None
_triton_client = None   # Triton Client 재사용을 위한 전역 변수

# Tokenizer 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
except Exception:
    logger.warning(f"[WARN] 지정된 토크나이저({TOKENIZER_ID}) 로드 실패. gpt2로 대체합니다.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

def log_prompt_token_stats(prompt: str, name: str = "prompt"):
    try:
        tokens = tokenizer.encode(prompt)
        logger.info(f"[TOKENS] {name}: {len(tokens)} tokens")
    except Exception as e:
        logger.warning(f"[TOKENS] {name}: calculation error: {e}")

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

    # ✅ 이름을 decoding_parameters -> sampling_parameters 로 변경
    sparams = InferInput("sampling_parameters", [1], "BYTES")

    # ✅ vLLM 공식 예제처럼 문자열로 보내는 쪽이 가장 안전
    params = {
        "temperature": str(float(temperature)),
        "top_p": str(float(top_p)),
        "max_tokens": str(int(max_tokens)),
        # "stream"는 별도 BOOL 인풋으로 이미 보내고 있으니 굳이 안 넣어도 됨
    }

    sparams.set_data_from_numpy(
        np.array([json.dumps(params).encode("utf-8")], dtype=object)
    )
    return text, sparams

def _triton_stream_generator(
        prompt: str,
        text: InferInput,
        sparams: InferInput,
        first_token_timeout: int = 10,
        idle_timeout: int = 20,
):
    cli = get_triton_client()  # ① Triton client 재사용

    stream_flag = InferInput("stream", [1], "BOOL")
    stream_flag.set_data_from_numpy(np.array([True], dtype=bool))
    outs = [InferRequestedOutput("text_output")]

    q = []
    done = threading.Event()

    def on_resp(result, error):
        if error:
            logger.error(f"[ERR] Triton Callback Error: {error}")
            done.set()
            return

        if result is None:
            done.set()
            return

        # 1) raw bytes 꺼내기
        arr = result.as_numpy("text_output")
        if arr is not None and len(arr) > 0:
            raw = arr[0]
            chunk = raw.decode("utf-8", errors="ignore")
            q.append(chunk)

        # 2) final response 여부 확인
        is_final = False
        try:
            resp = result.get_response()  # ModelInferResponse(proto)
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

    # -------------------------------
    # ③ 스트림 시작
    # -------------------------------
    cli.start_stream(callback=on_resp)
    cli.async_stream_infer(MODEL_NAME, inputs=[text, sparams, stream_flag], outputs=outs)

    # -------------------------------
    # ④ 스트리밍 루프
    # -------------------------------
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
                # 첫 토큰 대기 타임아웃
                if not got_first and (now - start_time > first_token_timeout):
                    logger.warning("[WARN] First token timeout")
                    break
                # 응답 도중 idle timeout
                if got_first and (now - last_yield_time > idle_timeout):
                    logger.warning("[WARN] Idle timeout after response started")
                    break
                time.sleep(0.005)
    finally:
        cli.stop_stream()

def _triton_infer_sync(
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

    # 제너레이터 소비
    for chunk in _triton_stream_generator(prompt, text, sparams, 10, 20):
        accumulated_text += chunk

    return accumulated_text.strip()

def triton_infer(
        prompt: str,
        *,
        stream: bool = True,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        timeout_first: int = 20,
        timeout_idle: int = 120,
):


    if stream:
        text, sparams = _make_inputs(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, stream=True
        )
        return _triton_stream_generator(prompt, text, sparams, timeout_first, timeout_idle)

    # Sync Path
    return _triton_infer_sync(
        prompt, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p
    )

# ---------------------------
# RAG Utils & Objects
# ---------------------------

def build_rag_objects():
    global _qdr, _emb, _retriever
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    # Qdrant Client
    _qdr = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)

    # Embedding Model
    _emb = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True
    )

    # Retriever 설정
    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=_emb)
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton Client도 미리 준비
    get_triton_client()

    return _qdr, _emb, _retriever

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

    resp = triton_infer(
        prompt, stream=False, max_tokens=64, temperature=0.3
    )

    try:
        # JSON 파싱 시도 (배열 찾기)
        match = re.search(r"\[.*?\]", resp, re.S)
        if match:
            return json.loads(match.group(0))[:10]
    except:
        pass

    # Fallback: 쉼표/줄바꿈 분리
    parts = re.split(r"[,;\n]", resp)
    kws = [re.sub(r"[^A-Za-z0-9\s\-]", "", p).strip() for p in parts]
    return [k for k in kws if 2 <= len(k) <= 40][:10]

def expand_query_kor(query: str) -> Tuple[str, List[str]]:
    terms = dynamic_expand_query_llm(query)
    # 원본 쿼리 + 확장어
    all_terms = sorted(set(terms + [query]))
    expanded_query = " ".join(all_terms) # OR 검색보다는 Embedding에는 나열이 나을 수 있음
    return expanded_query, all_terms

def dense_retrieve_hybrid(client: QdrantClient, emb, expanded_text: str, keywords: List[str]):
    try:
        q_vec = emb.get_query_embedding(expanded_text)
    except Exception:
        q_vec = emb.get_text_embedding(expanded_text)

    hits = client.query_points(
        collection_name=COLLECTION,
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
        if not kw: continue
        # 제목 가중치
        if title and fuzz.partial_ratio(kw, title) >= FUZZ_MIN:
            best = max(best, 80) # 단순화된 점수
        # 본문
        if body and fuzz.partial_ratio(kw, body) >= FUZZ_MIN:
            best = max(best, 60)

    return best / 100.0 # 정규화 (0~1 범위 유도)

def rrf_rerank(hits, keywords: List[str], k=60):
    scored = {}
    id2hit = {h.id: h for h in hits}

    # 키워드 부스팅 점수 미리 계산
    boost_map = {}
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

        if doc_id in seen_ids: continue
        seen_ids.add(doc_id)

        body, title = _payload_texts(payload)
        body = clamp_text(body, SNIPPET_MAX_CHARS)

        items.append(f"[{i}] {title}\n{body}")
        refs.append(f"[{i}] {title}")

        if len(items) >= TOP_K_RETURN: break

    return "\n\n".join(items), refs

# ---------------------------
# Main Logic (Gating & Execution)
# ---------------------------

def decide_rag_needed(query: str) -> bool:
    # 간단한 규칙 기반 필터링 우선
    casual = ["안녕", "날씨", "이름", "뭐해", "반가워"]
    if any(c in query for c in casual):
        return False

    # LLM 판단
    prompt = f"Is this query asking for factual knowledge? (YES/NO)\nQuery: {query}\nAnswer:"
    resp = triton_infer(prompt, stream=False, max_tokens=10)
    return "yes" in resp.lower()
