# rag_pipeline/retrieval.py
import json
import re
import time
from typing import Any, Dict, List, Tuple
from qdrant_client import QdrantClient
from rapidfuzz import fuzz
from sympy.benchmarks.bench_meijerint import timings

from settings import (
    FUZZ_MIN, TOP_K_BASE, TOP_K_RETURN, SNIPPET_MAX_CHARS, logger,
    DEFAULT_MODEL_NAME,
)
from triton_client import triton_infer, extract_final_answer

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

def dynamic_expand_query_llm(query: str) -> List[str]:
    prompt = f"""
You are a scientific keyword generator for academic search.
Respond ONLY with a JSON array of 8 concise English keywords.
Do NOT include explanations, examples, or formatting outside the array.

Input: {query}
Output:
""".strip()
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
        *,
        is_e5: bool = False,
        timings: Dict[str, float] | None = None,
):
    """
    A/B 어디든 동일하게 사용 가능하도록 collection_name 인자를 추가.
    timings가 주어지면,
      - embed_query: 쿼리 임베딩 시간
      - qdrant_search: Qdrant 검색 시간
    을 기록한다.
    """
    text_for_query = expanded_text
    if is_e5:
        # e5 스타일 쿼리 프리픽스
        text_for_query = "query: " + expanded_text

    # 1) 임베딩 시간 측정
    t0 = time.time()
    try:
        q_vec = emb.get_query_embedding(text_for_query)
    except Exception:
        q_vec = emb.get_text_embedding(text_for_query)
    t1 = time.time()

    if timings is not None:
        timings["embed_query"] = t1 - t0

    # 2) Qdrant 검색 시간 측정
    t2 = time.time()
    hits = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=TOP_K_BASE,
        with_payload=True,
        timeout=3
    ).points
    t3 = time.time()

    if timings is not None:
        timings["qdrant_search"] = t3 - t2

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

        if i <= 3:
            logger.info(f"[DEBUG] RAW_TITLE[{i}]: {repr(title)}")
            logger.info(f"[DEBUG] RAW_BODY[{i}]: {repr(body)}")

        items.append(f"[{i}] {title}\n{body}")
        refs.append(f"[{i}] {title}")

        if len(items) >= TOP_K_RETURN:
            break

    return "\n\n".join(items), refs
