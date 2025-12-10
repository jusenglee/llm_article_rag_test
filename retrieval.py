"""
논문 기반 RAG 검색용 retrieval 모듈.

주요 기능:
- LLM 기반 쿼리 확장 (한글 질의 → 영문 키워드 리스트)
- dense 벡터 검색 + Qdrant MatchText를 이용한 hybrid 검색
- fuzzy 키워드 부스팅을 활용한 RRF 기반 재정렬
- 최종 LLM용 컨텍스트 문자열/참조 리스트 생성
"""

import json
import re
import time
from typing import Any, Dict, List, Tuple

from typing import List, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchText

from qdrant_client import QdrantClient
from rapidfuzz import fuzz

from settings import (
    FUZZ_MIN,             # fuzzy 매칭 임계값 (0~100)
    TOP_K_BASE,           # 1차 검색 시 가져올 후보 개수
    TOP_K_RETURN,         # 최종 컨텍스트에 넣을 문서 개수
    SNIPPET_MAX_CHARS,    # 컨텍스트 snippet 길이 제한
    logger,
    DEFAULT_MODEL_NAME,
)
from triton_client import triton_infer, extract_final_answer


# ============================================
# 0. 검색 꼬리 패턴 정의
# ============================================

SEARCH_TAIL_PATTERNS = [
    r"데이터베이스에서\s*검색해줘",
    r"데이터베이스에서\s*검색해 줘",
    r"검색해줘",
    r"검색해 줘",
    r"찾아줘",
    r"찾아 줘",
    r"알려줘",
    r"알려 줘",
    r"설명해줘",
    r"설명해 줘",
]


# ============================================
# 1. 텍스트 유틸 / payload 처리
# ============================================

def clamp_text(s: Any, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """
    긴 텍스트를 한 줄로 정규화하고, max_chars 기준으로 잘라냄.

    - 공백/줄바꿈을 모두 공백 하나로 치환
    - 좌우 trim
    - 지정 길이 초과분은 잘라서 snippet으로 사용
    """
    if not isinstance(s, str):
        s = str(s)
    # 연속 공백/줄바꿈 → 공백 하나
    s = re.sub(r"\s+", " ", s).strip()
    return s[: int(max_chars)]


def _payload_texts(payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Qdrant payload에서 본문(body)와 제목(title)을 추출.

    우선순위:
    1) _node_text 필드 (llama_index 기본 필드)
    2) _node_content(JSON) 안의 text / metadata.title
    3) 제목이 없으면 본문 앞부분을 잘라서 임시 제목 생성
    """
    body = ""
    title = payload.get("_title", "") or ""

    # 1) 기본 node 텍스트
    if isinstance(payload.get("_node_text"), str):
        body = payload["_node_text"].strip()

    # 2) _node_content 안에 더 풍부한 텍스트가 있으면 교체
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
            # payload 구조가 예상과 다르거나 JSON 파싱 실패 시, 조용히 무시
            pass

    # 3) 그래도 제목이 비어 있으면 본문 앞부분으로 대체
    if not title:
        title = (body[:60] + "...") if body else "Untitled"

    return body, title


def normalize_query_for_expand(query: str) -> str:
    """
    키워드 확장용으로, '검색해줘/알려줘/찾아줘' 같은
    액션 꼬리를 제거해 주제만 남긴다.
    도메인 정보는 전혀 하드코딩하지 않는다.
    """
    q = re.sub(r"\s+", " ", (query or "")).strip()
    for pat in SEARCH_TAIL_PATTERNS:
        q = re.sub(pat + r"\s*$", "", q)
    return q.strip()


# ============================================
# 2. LLM 기반 키워드 확장
# ============================================

def dynamic_expand_query_llm(query: str, model_name: str) -> Tuple[str, List[str]]:
    """
    한글/영문 질의를 받아 학술 검색용 확장 쿼리 + 키워드 리스트를 생성.

    반환:
      expanded_query : dense 임베딩에 넣을 '짧은 영어 문장'
      keywords       : lexical / fuzzy 부스팅에 쓸 '짧은 영어 키워드 리스트'
    """
    core_query = normalize_query_for_expand(query)
    if not core_query:
        core_query = (query or "").strip()

    logger.info(f"[EXPAND] core_query={core_query!r}")

    prompt = f"""
Rules:
- keywords:
  - 3 to 8 short English phrases (1–3 words each)
  - Focus on main concepts (time, place, field, key terms).
- Do NOT output anything outside the JSON.
- Do NOT add comments or explanations.
    
Original question: {core_query}
""".strip()

    raw = triton_infer(
        DEFAULT_MODEL_NAME,
        prompt,
        stream=False,
        max_tokens=256,
        temperature=0.0,
    )
    logger.debug(f"[EXPAND] raw (head)={repr(str(raw)[:300])}")

    # gpt-oss 포맷 대비 (assistantfinal 이후만 추출)
    resp = extract_final_answer(raw)
    resp = resp.strip()
    logger.debug(f"[EXPAND] resp (head)={repr(resp[:300])}")

    expanded = ""
    keywords: List[str] = []

    # 1) 가장 먼저, 전체를 JSON으로 시도
    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        # 2) 그 다음, { ... } 부분만 잘라서 다시 시도 (최소 관용 파서)
        m = re.search(r"\{.*\}", resp, re.S)
        if not m:
            logger.warning(f"[EXPAND] JSON parse failed: no JSON object in resp={resp[:200]!r}")
            data = None
        else:
            try:
                data = json.loads(m.group(0))
            except Exception as e:
                logger.warning(f"[EXPAND] JSON parse failed(2): {e!r}, resp={resp[:200]!r}")
                data = None

    if isinstance(data, dict):
        ks = data.get("keywords") or []
        if not isinstance(ks, list):
            ks = [ks]
        for k in ks:
            s = str(k).strip()
            if not s:
                continue
            if len(s) > 40:
                continue
            keywords.append(s)

    # 3) JSON 이 완전히 깨졌다면, 최소 fallback:
    expanded = core_query or query
    if not keywords:
        keywords = [core_query or query]

    # 중복 제거 + 순서 유지
    keywords = dedup_keep_order(keywords)

    logger.info(f"[EXPAND] expanded_query={expanded!r}")
    logger.info(f"[EXPAND] keywords={keywords}")

    return expanded, keywords


def dedup_keep_order(xs: List[Any]) -> List[Any]:
    """
    순서를 유지하면서 중복을 제거한다.
    """
    seen = set()
    out: List[Any] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_query_kor(query: str, model_name: str = DEFAULT_MODEL_NAME) -> Tuple[str, List[str]]:
    """
    - 질의를 normalize(검색 꼬리 제거)한 후,
    - LLM 기반 확장 쿼리 + 키워드 리스트를 생성하고,
    - all_terms 에는 키워드 + 한글 정규화 질의 + 원 질의를 모두 포함.

    반환:
      expanded_query: dense 임베딩용 확장 문자열 (주로 영어 한 줄)
      all_terms:     키워드 + 정규화 질의 + 원 질의 리스트
    """
    core_q = normalize_query_for_expand(query)
    expanded, kws = dynamic_expand_query_llm(query, model_name=model_name)

    # all_terms 는 키워드 + 한글 질의까지 포함해서 lexical / fuzzy에 사용
    all_terms = dedup_keep_order(kws + ([core_q] if core_q else []) + [query])

    logger.info(f"[EXPAND] expanded_query (head)={expanded[:200]!r}")
    logger.info(f"[EXPAND] keywords(all_terms)={all_terms}")

    return expanded, all_terms



# ============================================
# 3. 하이브리드 검색 (dense + lexical)
# ============================================

def build_lexical_filter_any(keywords: List[str]) -> Optional[Filter]:
    """
    키워드 리스트를 받아서,
    '_node_text' 필드에 대해 '키워드 중 하나라도 매칭되면 OK'인 Filter 생성.

    Qdrant 쿼리 의미:
      should = [ MatchText(kw1), MatchText(kw2), ... ]
      → kw1 OR kw2 OR ...
    """
    kw_clean = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    if not kw_clean:
        return None

    # 너무 많은 키워드는 성능/노이즈 때문에 잘라주는 것도 옵션 (예: 상위 8개만)
    MAX_LEXICAL_KEYWORDS = 8
    kw_clean = kw_clean[:MAX_LEXICAL_KEYWORDS]

    conds = [
        FieldCondition(
            key="_node_text",
            match=MatchText(text=kw),
        )
        for kw in kw_clean
    ]

    return Filter(should=conds)

def dense_retrieve_hybrid(
        client: QdrantClient,
        emb: Any,
        expanded_text: str,
        keywords: List[str],
        collection_name: str,
        *,
        timings: Dict[str, float] | None = None,
):
    """
    하이브리드 검색 엔트리 포인트.

    1) dense vector 검색
       - E5 / instruct 등 HuggingFaceEmbedding을 이용한 벡터 질의
       - query_instruction / text_instruction 은 emb 내부 설정에 위임

    2) Qdrant MatchText 기반 lexical 검색
       - payload의 `_node_text` 필드에 대해 full-text 매칭 수행
       - LLM에서 확장된 키워드 리스트를 하나로 합쳐 사용

    3) 두 결과(dense, lexical)를 ID 기준으로 병합하여 반환
       - 실제 점수 조합 및 정렬은 rrf_rerank()에서 수행한다.

    timings가 주어지면:
      - "embed_query": 쿼리 임베딩 시간
      - "qdrant_dense": 벡터 검색 시간
      - "qdrant_lexical": 키워드 검색 시간
    을 기록한다.
    """
    # 0) 질의 텍스트 정리 (None 방어 + 공백 trim)
    text_for_query = (expanded_text or "").strip()
    logger.info(
        f"[HYBRID] collection={collection_name}, "
        f"text_for_query_len={len(text_for_query)}, "
        f"#keywords={len(keywords)}"
    )

    # ------------------------
    # 1) dense vector 검색
    # ------------------------
    t0 = time.time()
    try:
        # HuggingFaceEmbedding.get_query_embedding:
        # - 내부에서 query_instruction("query:", instruct 등)을 알아서 붙여줌
        q_vec = emb.get_query_embedding(text_for_query)
    except Exception:
        # 일부 구현체는 get_query_embedding이 없을 수 있으므로 fallback
        q_vec = emb.get_text_embedding(text_for_query)
    t1 = time.time()

    if timings is not None:
        timings["embed_query"] = t1 - t0

    t2 = time.time()
    try:
        dense_hits = client.query_points(
            collection_name=collection_name,
            query=q_vec,
            limit=TOP_K_BASE,
            with_payload=True,
            timeout=3,
        ).points
    except Exception as e:
        logger.warning(f"[HYBRID] dense search failed: {e}")
        dense_hits = []
    t3 = time.time()

    if timings is not None:
        timings["qdrant_dense"] = t3 - t2

    logger.info(f"[HYBRID] dense_hits={len(dense_hits)}")

    # ------------------------
    # 2) 키워드 기반 lexical 검색
    # ------------------------
    lexical_hits = []
    if keywords:
        flt = build_lexical_filter_any(keywords)
        logger.debug(f"[HYBRID] lexical keywords={keywords!r}, filter={flt!r}")

        if flt is not None:
            try:
                t4 = time.time()
                lexical_hits = client.query_points(
                    collection_name=collection_name,
                    query=None,              # 쿼리 벡터 없음 → 필터 기반 full-text
                    query_filter=flt,
                    limit=TOP_K_BASE,
                    with_payload=True,
                    timeout=3,
                ).points
                t5 = time.time()
                if timings is not None:
                    timings["qdrant_lexical"] = t5 - t4
            except Exception as e:
                logger.warning(f"[HYBRID] lexical search failed: {e}")
                lexical_hits = []
        else:
            logger.info("[HYBRID] lexical filter is None (no valid keywords)")

    logger.info(
        f"[HYBRID] lexical_hits={len(lexical_hits)}, "
        f"dense_hits={len(dense_hits)}"
    )

    # ------------------------
    # 3) dense + lexical 결과 병합
    # ------------------------
    merged: Dict[Any, Any] = {}
    for h in dense_hits:
        merged[h.id] = h

    for h in lexical_hits:
        # 이미 dense에서 나온 id면 dense 결과를 우선 사용
        if h.id not in merged:
            merged[h.id] = h

    hits = list(merged.values())
    logger.info(
        f"[HYBRID] merged_hits={len(hits)} "
        f"(dense={len(dense_hits)}, lexical={len(lexical_hits)})"
    )

    return hits


# ============================================
# 4. 키워드 부스트 + RRF 리랭킹
# ============================================

def _keyword_score_for_hit(payload: Dict[str, Any], keywords: List[str]) -> float:
    """
    한 개의 hit(payload)에 대해 키워드 매칭 기반 부스팅 스코어를 계산.

    - 제목/본문에 대해 fuzzy partial_ratio 사용
    - 제목 매칭 시 더 높은 점수(예: 0.8), 본문은 그보다 낮게(예: 0.6)
    - 최종적으로 0.0 ~ 1.0 범위로 정규화된 부스트 값을 반환
    """
    body, title = _payload_texts(payload)

    # Fuzzy matching 전에 텍스트 길이를 제한해 속도 확보(최적화용)
    body = clamp_text(body, max_chars=4096)

    if not body and not title:
        return 0.0

    best = 0.0
    for kw in keywords:
        if not kw:
            continue

        # 제목 가중치 (타이틀과의 부분 일치)
        if title and fuzz.partial_ratio(kw, title) >= FUZZ_MIN:
            best = max(best, 80)  # 0~100 스케일에서 우선 80 부여

        # 본문 가중치
        if body and fuzz.partial_ratio(kw, body) >= FUZZ_MIN:
            best = max(best, 60)

    # 0~100 점수 → 0.0~1.0 범위로 정규화
    return best / 100.0


def rrf_rerank(hits, keywords: List[str], k: int = 60):
    """
    RRF(Reciprocal Rank Fusion) + 벡터 점수 + 키워드 부스트를
    섞어서 최종 정렬을 수행.

    - RRF: 1 / (k + rank)
    - vec_score: Qdrant 벡터 점수 (distance/score 설정에 따라 해석)
    - boost: _keyword_score_for_hit에서 계산한 키워드 기반 부스트

    final_score = rrf_score + (vec_score * 0.5) + (boost * 0.3)
    """
    scored: Dict[Any, float] = {}
    id2hit = {h.id: h for h in hits}

    # 키워드 부스팅 점수 미리 계산
    boost_map: Dict[Any, float] = {}
    for h in hits:
        boost_map[h.id] = _keyword_score_for_hit(h.payload or {}, keywords)

    for rank, h in enumerate(hits, start=1):
        # rank가 작을수록(RRF 상위일수록) 큰 값을 가짐
        rrf_score = 1.0 / (k + rank)

        # Qdrant의 score (None일 수 있으므로 방어)
        vec_score = float(h.score) if getattr(h, "score", None) is not None else 0.0

        # 위에서 계산한 키워드 부스트
        boost = boost_map.get(h.id, 0.0)

        # 가중치 합산 (튜닝 포인트)
        final_score = rrf_score + (vec_score * 0.5) + (boost * 0.3)
        scored[h.id] = final_score

    # 스코어 내림차순 정렬
    sorted_ids = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
    reranked = [id2hit[iD] for iD in sorted_ids]

    # 상위 몇 개 결과는 디버깅 로그로 남김
    top_dbg = []
    for h in reranked[:3]:
        top_dbg.append(
            f"id={h.id}, vec={getattr(h, 'score', None)}, "
            f"boost={boost_map.get(h.id, 0.0):.3f}, "
            f"title={repr((_payload_texts(h.payload or {})[1]))[:80]}"
        )
    logger.info(f"[RRF] top3 after rerank:\n  " + "\n  ".join(top_dbg))

    return reranked


# ============================================
# 5. 컨텍스트 빌드
# ============================================

def build_context(hits):
    """
    상위 hits를 바탕으로 LLM에게 넘길 컨텍스트 문자열과
    출처(refs) 리스트를 생성.

    - 동일 doc_id(doc_id/paper_id)가 중복 등장하면 첫 번째만 사용
    - 각 아이템은 "[i] 제목\\n본문" 형태의 문자열로 구성
    - refs는 "[i] 제목" 리스트로, UI/로그 표시에 사용 가능
    """
    items: List[str] = []
    refs: List[str] = []
    seen_ids = set()

    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        # 논문/문서 고유 ID 추출 (없으면 point id 사용)
        doc_id = payload.get("doc_id") or payload.get("paper_id") or str(h.id)

        # 동일 doc_id는 한 번만 사용
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        body, title = _payload_texts(payload)
        body = clamp_text(body, SNIPPET_MAX_CHARS)

        # 상위 몇 개는 raw 텍스트를 로그로 남겨 디버깅에 활용
        if i <= TOP_K_RETURN:
            logger.info(f"[DEBUG] RAW_TITLE[{i}]: {repr(title)}")

        items.append(f"[{i}] {title}\n{body}")
        refs.append(f"[{i}] {title}")

        if len(items) >= TOP_K_RETURN:
            break

    logger.info(f"[CTX] built context items={len(items)}, unique_docs={len(seen_ids)}")
    # LLM에 넘길 컨텍스트는 큰 문자열 하나로 합침
    return "\n\n".join(items), refs
