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
from qdrant_client import QdrantClient
from rapidfuzz import fuzz
from sympy.benchmarks.bench_meijerint import timings
from qdrant_client.http.models import Filter, FieldCondition, MatchText

from settings import (
    FUZZ_MIN,             # fuzzy 매칭 임계값 (0~100)
    TOP_K_BASE,           # 1차 검색 시 가져올 후보 개수
    TOP_K_RETURN,         # 최종 컨텍스트에 넣을 문서 개수
    SNIPPET_MAX_CHARS,    # 컨텍스트 snippet 길이 제한
    logger,
    DEFAULT_MODEL_NAME,
)
from triton_client import triton_infer, extract_final_answer

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

def dynamic_expand_query_llm(query: str) -> List[str]:
    """
    한글 사용자의 질의를 입력으로 받아,
    LLM을 이용해 '학술 검색용 영문 키워드 리스트'를 생성.

    - LLM에게 JSON array 형태로만 응답하도록 강하게 제한
    - 실패 시 쉼표/줄바꿈 기준 Fallback 파싱
    """
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

    # gpt-oss가 analysis/assistantfinal 포맷을 쓸 수 있으므로 최종 답변만 추출
    resp = extract_final_answer(raw)

    try:
        # JSON 배열 패턴만 추출하여 파싱
        match = re.search(r"\[.*?\]", resp, re.S)
        if match:
            return json.loads(match.group(0))[:10]
    except Exception:
        # JSON 파싱 실패 시 조용히 Fallback으로 전환
        pass

    # Fallback: 쉼표/줄바꿈 등으로 쪼개서 영문 키워드 후보 생성
    parts = re.split(r"[,;\n]", resp)
    kws = [re.sub(r"[^A-Za-z0-9\s\-]", "", p).strip() for p in parts]
    # 너무 짧거나 긴 토큰은 제외
    return [k for k in kws if 2 <= len(k) <= 40][:10]

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

def expand_query_kor(query: str) -> Tuple[str, List[str]]:
    """
    한글 사용자 질의를 받아:
      1) LLM으로부터 영문 키워드 리스트를 받고
      2) 원 질의(query)를 포함해 dedup 후
      3) '확장 질의 문자열(expanded_query)'와 '키워드 리스트'를 반환.

    - expanded_query: "kw1 kw2 ... kwN {원질문}" 형식의 하나의 긴 문자열
    - keywords: LLM 키워드 + 원질문까지 포함된 리스트
    """
    # 1) LLM이 반환한 영문 키워드들
    terms = dynamic_expand_query_llm(query)

    # 2) LLM 키워드 + 원문 질의를 순서 유지 + 중복 제거
    all_terms = dedup_keep_order(terms + [query])

    # 3) dense 검색용 확장 텍스트
    expanded_query = " ".join(all_terms)
    print(expanded_query)
    return expanded_query, all_terms

def dense_retrieve_hybrid(
        client: QdrantClient,
        emb,
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
        logger.warning(f"[dense_retrieve_hybrid] dense search failed: {e}")
        dense_hits = []
    t3 = time.time()

    if timings is not None:
        timings["qdrant_dense"] = t3 - t2

    # ------------------------
    # 2) 키워드 기반 lexical 검색
    # ------------------------
    lexical_hits = []
    if keywords:
        # 키워드들을 하나의 문장으로 합쳐 full-text match에 사용
        text_for_lexical = " ".join([k for k in keywords if k]).strip()
        if text_for_lexical:
            try:
                # _node_text 필드를 대상으로 MatchText 수행
                flt = Filter(
                    must=[
                        FieldCondition(
                            key="_node_text",
                            match=MatchText(text=text_for_lexical),
                        )
                    ]
                )
                t4 = time.time()
                lexical_hits = client.query_points(
                    collection_name=collection_name,
                    query=None,   # 순수 텍스트 필터 기반 검색
                    filter=flt,
                    limit=TOP_K_BASE,
                    with_payload=True,
                    timeout=3,
                ).points
                t5 = time.time()
                if timings is not None:
                    timings["qdrant_lexical"] = t5 - t4
            except Exception as e:
                logger.warning(f"[dense_retrieve_hybrid] lexical search failed: {e}")
                lexical_hits = []

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
    return hits


def _keyword_score_for_hit(payload: Dict[str, Any], keywords: List[str]) -> float:
    """
    한 개의 hit(payload)에 대해 키워드 매칭 기반 부스팅 스코어를 계산.

    - 제목/본문에 대해 fuzzy partial_ratio 사용
    - 제목 매칭 시 더 높은 점수(예: 0.8), 본문은 그보다 낮게(예: 0.6)
    - 최종적으로 0.0 ~ 1.0 범위로 정규화된 부스트 값을 반환
    """
    body, title = _payload_texts(payload)

    #  Fuzzy matching 전에 텍스트 길이를 제한해 속도 확보(최적화용)
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
    return [id2hit[iD] for iD in sorted_ids]


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
        if i <= 3:
            logger.info(f"[DEBUG] RAW_TITLE[{i}]: {repr(title)}")
            logger.info(f"[DEBUG] RAW_BODY[{i}]: {repr(body)}")

        items.append(f"[{i}] {title}\n{body}")
        refs.append(f"[{i}] {title}")

        if len(items) >= TOP_K_RETURN:
            break

    # LLM에 넘길 컨텍스트는 큰 문자열 하나로 합침
    return "\n\n".join(items), refs