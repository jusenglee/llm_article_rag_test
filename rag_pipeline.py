# rag_pipeline/rag_pipeline.py
"""
RAG 파이프라인 진입점 모듈.

기능 요약:
- decide_rag_needed: 질의가 RAG(외부 지식)를 필요로 하는지 여부를 판정
- run_rag_once: 단일 스택(A or B)에 대해 RAG 파이프라인 1회 실행
- run_rag_ab_compare: 동일 질의에 대해 A/B 스택을 모두 실행하여 비교용 결과 반환

A 스택: intfloat/multilingual-e5-large-instruct 기반 (instruct 스타일 쿼리 임베딩)
B 스택: intfloat/multilingual-e5-large 기반 (클래식 E5 query/passage 프리픽스)
"""

import time
from typing import Dict

from settings import (
    DEFAULT_MODEL_NAME,
    COLLECTION,
    COLLECTION_B,
    EMBED_MODEL,
    EMBED_MODEL_B,
    logger,
)
from rag_types import RagResult
from triton_client import triton_infer, extract_final_answer
from rag_store import build_rag_objects_dual
from retrieval import (
    expand_query_kor,
    dense_retrieve_hybrid,
    rrf_rerank,
    build_context,
)


def decide_rag_needed(query: str, model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """
    사용자의 질의가 외부 지식(RAG)을 필요로 하는지 판단.

    판단 순서:
      1) 캐주얼/잡담 키워드 기반 cheap heuristic
      2) '논문/연구/정의/법' 등 명백한 지식/사실성 키워드 포함 시 → 바로 RAG 사용
      3) 위로도 애매하면 LLM 분류기로 YES/NO 판정
      4) LLM 호출 실패/예상 밖 응답 → 보수적으로 RAG 사용(True)

    반환:
      - True  → RAG를 사용해 답변하는 것이 좋음
      - False → 일반 Chat 모드로 처리해도 됨
    """
    q = (query or "").strip()
    if not q:
        # 완전히 비어 있는 질의는 굳이 RAG까지 호출할 필요 없음
        return False

    # 1) 캐주얼 / 스몰톡 후보 (substring 기준)
    casual_keywords = [
        "날씨", "기분", "안녕", "잘 지내", "잘지내", "하이", "ㅎㅇ",
        "좋아", "고마워", "감사", "사랑", "잘자", "잘 자",
        "심심", "배고파", "ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "??",
        "이름 뭐", "이름이 뭐야", "너 누구", "넌 누구",
        "몇 시야", "몇시야", "몇 시 인지", "몇시 인지",
    ]

    # 아주 짧은 질의 + 캐주얼 토큰 위주면 그냥 CHAT 으로 보내도 충분
    if len(q) <= 10 and any(ck in q for ck in casual_keywords):
        return False

    # 2) "명백히 지식/논문/사실성" 느낌 나는 키워드 → 바로 RAG 사용
    knowledge_cues = [
        "논문", "연구", "실험", "통계", "근거", "데이터",
        "정의", "개념", "원리", "메커니즘",
        "논리", "이론", "증거",
        "어떻게 동작", "어떻게 작동",
        "제도", "법", "규정", "규칙",
        "성능", "비교", "벤치마크",
        "초록", "abstract", "figure", "table",
    ]

    if any(k in q for k in knowledge_cues):
        return True

    # 3) 길이 기준: 일정 길이 이상이면 "정보/설명 요구"일 확률이 높다고 가정
    #    (챗봇 잡담은 보통 20자 이내가 많다는 경험적 가정)
    if len(q) >= 40:
        return True

    # 4) 최종적으로 LLM 분류기에 위임
    prompt = (
        "You are a classifier.\n"
        "Decide if the user query requires external factual knowledge, "
        "such as academic, scientific, technical, legal, medical, or other "
        "domain-specific information beyond common chit-chat.\n"
        "\n"
        "If the query asks for factual explanations, definitions, comparisons, "
        "statistics, laws, research results, or information that usually comes from "
        "articles, papers, or documentation, answer YES.\n"
        "If the query is casual small talk, greetings, or purely subjective/emotional "
        "chat (like saying hello, talking about feelings without asking for factual "
        "information), answer NO.\n"
        "\n"
        "Answer format rules:\n"
        "- Respond with exactly YES or NO.\n"
        "- Do not add any other words.\n"
        "\n"
        f"Query: {q}\n"
        "Answer (YES or NO only):"
    )

    try:
        raw = triton_infer(model_name, prompt, stream=False, max_tokens=4)
        resp = extract_final_answer(raw).strip().upper()
    except Exception as e:
        logger.warning(f"[decide_rag_needed] LLM gating failed: {e}")
        # 게이팅 자체가 실패하면 보수적으로 RAG 사용
        return True

    # 한국어 '예', '아니오' 같은 변종 응답까지 최소한 처리
    if resp.startswith("YES") or resp.startswith("예"):
        return True
    if resp.startswith("NO") or resp.startswith("아니"):
        return False

    # 애매한 응답이면 보수적으로 RAG 사용
    logger.warning(f"[decide_rag_needed] Unexpected classifier response: {resp!r}")
    return True


def run_rag_once(
        query: str,
        stack: str = "A",
        with_llm: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
) -> RagResult:
    """
    단일 RAG 스택(A 또는 B)에 대해 전체 파이프라인을 1회 실행.

    수행 단계:
      1) 스택별 객체(Qdrant, Embedding, Retriever) 준비
      2) LLM 기반 쿼리 확장 (expand_query_kor)
      3) dense + lexical hybrid 검색 (dense_retrieve_hybrid)
      4) RRF + 키워드 부스팅 기반 재정렬 (rrf_rerank)
      5) LLM 컨텍스트/출처 문자열 생성 (build_context)
      6) (옵션) LLM 호출로 최종 답변 생성

    timings 딕셔너리에는 각 단계별 소요 시간을 기록해 반환한다.
    """
    t_all0 = time.time()
    timings: Dict[str, float] = {}

    # 1) 스택별 객체 준비
    t0 = time.time()
    if stack == "A":
        # A 스택: instruct 기반 임베딩 (multilingual-e5-large-instruct)
        # build_rag_objects_dual()는 A/B 둘 다 초기화하므로,
        # 여기서는 A 관련 객체만 사용한다.
        qdr, emb, retriever, _, _, _ = build_rag_objects_dual()
        collection = COLLECTION
    elif stack == "B":
        # B 스택: classic E5 임베딩 (multilingual-e5-large)
        qdr, emb, retriever, qdr2, emb2, retriever2 = build_rag_objects_dual()
        # B용 객체로 교체
        qdr, emb, retriever = qdr2, emb2, retriever2
        collection = COLLECTION_B
    else:
        # 방어 코드: 정의되지 않은 스택이 들어올 경우 오류
        raise ValueError(f"Unknown RAG stack: {stack}")
    timings["stack_init"] = time.time() - t0

    # 2) 쿼리 확장 (한글 질의 → 영문 키워드 + 원문 포함 확장 문자열)
    t0 = time.time()
    expanded_query, kws = expand_query_kor(query)
    timings["expand_query"] = time.time() - t0

    # 3) hybrid 검색 (dense + lexical)
    t0 = time.time()
    hits = dense_retrieve_hybrid(
        client=qdr,
        emb=emb,
        expanded_text=expanded_query,
        keywords=kws,
        collection_name=collection,
        timings=timings,  # 내부에서 embed_query, qdrant_dense, qdrant_lexical 기록
    )
    timings["dense_search"] = time.time() - t0

    # 4) RRF + 키워드 부스팅 기반 재정렬
    t0 = time.time()
    reranked = rrf_rerank(hits, kws)
    timings["rerank"] = time.time() - t0

    # 5) 컨텍스트/출처 생성 (LLM에 넘길 텍스트)
    t0 = time.time()
    context, refs = build_context(reranked)
    timings["build_context"] = time.time() - t0

    # 6) (옵션) LLM 호출
    llm_answer = None
    if with_llm:
        # TODO: 여기에 컨텍스트 기반 LLM 호출 로직(triton_infer 등)을 구현
        #  - system/user 프롬프트 구성
        #  - 스트리밍/비스트리밍 여부
        #  - extract_final_answer 활용 여부 등
        ...
        timings["llm_answer"] = time.time() - t0

    timings["total"] = time.time() - t_all0

    logger.info(
        f"[PERF][{stack}] expand={timings['expand_query']:.4f}s, "
        f"dense={timings['dense_search']:.4f}s, "
        f"rerank={timings['rerank']:.4f}s, "
        f"context={timings['build_context']:.4f}s, "
        f"llm={timings.get('llm_answer', 0.0):.4f}s, "
        f"total={timings['total']:.4f}s"
    )

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
    동일 질의에 대해 A/B 두 개의 RAG 스택을 모두 실행해 결과를 비교용으로 반환.

    반환 형식:
        {
            "A": RagResult(...),  # instruct 기반 스택 결과
            "B": RagResult(...),  # classic E5 기반 스택 결과
        }

    - 오프라인 실험/레포트/로그 분석에 활용 가능
    - with_llm=False 로 호출하면 검색/리랭크/컨텍스트까지만 비교 가능
    """
    res_a = run_rag_once(query, stack="A", with_llm=with_llm, model_name=model_name)
    res_b = run_rag_once(query, stack="B", with_llm=with_llm, model_name=model_name)

    return {"A": res_a, "B": res_b}
