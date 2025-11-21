# rag_pipeline/rag_pipeline.py
import time
from typing import Dict

from settings import (
    DEFAULT_MODEL_NAME, COLLECTION, COLLECTION_B, EMBED_MODEL, EMBED_MODEL_B,
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
    casual = [
        "날씨", "기분", "안녕", "좋아", "이름", "몇 시", "누구", "심심", "배고파",
        "오늘", "어때", "ㅋㅋ", "ㅎ", "??", "잘자", "사랑", "고마워", "ㅎㅇ"
    ]
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
    t_all0 = time.time()
    timings: Dict[str, float] = {}

    # 1) 스택별 객체 준비
    t0 = time.time()
    if stack == "A":
        qdr, emb, retriever, _, _, _ = build_rag_objects_dual()
        collection = COLLECTION
        is_e5 = False
    else:
        qdr, emb, retriever, qdr2, emb2, retriever2 = build_rag_objects_dual()
        qdr, emb, retriever = qdr2, emb2, retriever2
        collection = COLLECTION_B
        is_e5 = True
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
        is_e5=is_e5,   # 👈 여기 중요 (e5 프리픽스)
    )
    timings["dense_search"] = time.time() - t0

    # 4) 리랭킹
    t0 = time.time()
    reranked = rrf_rerank(hits, kws)
    timings["rerank"] = time.time() - t0

    # 5) 컨텍스트
    t0 = time.time()
    context, refs = build_context(reranked)
    timings["build_context"] = time.time() - t0

    # 6) (옵션) LLM
    llm_answer = None
    if with_llm:
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
    같은 질문에 대해 A/B 스택을 모두 실행.
    레포트/로그/오프라인 분석에 쓰기 좋게 dict로 리턴.
    """
    res_a = run_rag_once(query, stack="A", with_llm=with_llm, model_name=model_name)
    res_b = run_rag_once(query, stack="B", with_llm=with_llm, model_name=model_name)

    return {"A": res_a, "B": res_b}

