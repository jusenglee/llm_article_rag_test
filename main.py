# -*- coding: utf-8 -*-
import logging, nest_asyncio, time, traceback
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from query_pipeline import (
    build_rag_objects, triton_infer, expand_query_kor, expand_variants,
    dense_retrieve_hybrid, rag_gate_decision, build_chat_prompt,
    keyword_boost, rrf_rerank, dedup_by_doc, build_context_and_refs,
    trim_context_to_budget, build_rag_prompt, decide_rag_needed, tokenizer
)

# ---------------------------
# 초기 설정
# ---------------------------
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("uvicorn")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 전역 객체 (멀티워커일 땐 프로세스별 초기화)
qdr = emb = retriever = None

@app.on_event("startup")
async def init_rag():
    global qdr, emb, retriever
    logger.info("🚀 Initializing RAG components...")
    qdr, emb, retriever = build_rag_objects()
    # Warmup: GPU/토크나이저/JIT lazy init 비용 제거
    try:
        from query_pipeline import tokenizer
        _ = tokenizer.encode("warmup")
        _ = emb.get_text_embedding("warmup")
    except Exception as e:
        logger.info(f"Warmup skipped: {e}")
    logger.info("✅ RAG pipeline ready.\n")

# ---------------------------
# 라우트
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_rag(request: Request, question: str = Form(...)):
    logs = []
    t0 = time.time()

    def log(msg):
        logger.info(msg); logs.append(msg)

    try:
        log(f"\n질문 > {question}")

        # --- 게이트 판단 ---
        t1 = time.time()
        need_rag = decide_rag_needed(question)  # 동기(비스트림)
        t2 = time.time()

        log(f"🧭 LLM 판단 결과: {'RAG 검색 수행' if need_rag else '일반 대화'}  "
            f"(gate={t2 - t1:.2f}s)")

        # --- 확장/검색 ---
        expanded_text, kw_list = expand_query_kor(question)
        keywords = expand_variants(kw_list)
        log(f"🔑 확장 키워드: {keywords}")

        t3 = time.time()
        hits = dense_retrieve_hybrid(qdr, emb, expanded_text, keywords)
        t4 = time.time()
        log(f"🔍 검색된 문서 수(max 20): {len(hits)}  (search={t4 - t3:.2f}s)")

        ok, gate_msg = rag_gate_decision(question, hits, kw_list, need_rag)
        log(gate_msg)

        # --- 일반 대화 ---
        if not ok:
            prompt = build_chat_prompt(question)
            t5 = time.time()
            full = triton_infer(prompt, stream=False)  # 문자열(비스트림)
            t6 = time.time()
            log(f"⚡ LLM: {t6 - t5:.2f}s")
            log("\n📘 답변:\n" + full.strip() + "\n" + "-"*80)
            return JSONResponse({"mode": "chat", "answer": full.strip(), "logs": logs})

        # --- RAG ---
        log("✅ 게이트 판단 결과: RAG 검색/응답 수행")
        boost_map = keyword_boost(hits, kw_list)
        reranked = rrf_rerank(hits, boost_map)
        final_hits = dedup_by_doc(reranked)
        ctx, refs = build_context_and_refs(final_hits)
        ctx = trim_context_to_budget(ctx)
        log(f"🔎 Retrieved top-{len(final_hits)} (after RRF+dedup).")

        rag_prompt = build_rag_prompt(ctx, question, refs)
        t7 = time.time()
        full = triton_infer(rag_prompt, stream=False)  # 문자열(비스트림)
        t8 = time.time()
        log(f"⚡ LLM: {t8 - t7:.2f}s")
        log("\n📘 답변:\n" + full.strip() + "\n" + "-"*80)

        return JSONResponse({"mode": "rag", "answer": full.strip(), "refs": refs, "logs": logs})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------------------
# SSE STREAM
# ---------------------------
@app.get("/query/stream")
async def query_stream(question: str):

    async def event_gen():
        try:
            yield "data: [STEP 0] 질문 수신\n\n"

            # STEP 1: 게이트 판단
            t0 = time.time()
            need_rag = decide_rag_needed(question)
            t1 = time.time()
            yield f"data: [STEP 1] 게이트={need_rag} (t={t1 - t0:.2f}s)\n\n"

            # ---------------------------------------------------------
            # ★ 게이트=False → RAG 전체 스킵 (최적화 핵심)
            # ---------------------------------------------------------
            if not need_rag:
                yield "data: [STEP 2] RAG 스킵 → 일반 대화 진행\n\n"
                prompt = build_chat_prompt(question)

                yield "data: [STEP 3] LLM 스트리밍 시작 (chat)\n\n"
                full_bytes = bytearray()
                for chunk in triton_infer(prompt, stream=True):
                    if not chunk:
                        continue

                    # chunk 자체가 bytes임 → 통째로 저장
                    if isinstance(chunk, bytes):
                        full_bytes.extend(chunk)
                    else:
                        full_bytes.extend(chunk.encode("utf-8"))

                    # 중간에는 placeholder 출력 (원하면)
                    yield f"data: {chunk}\n\n"

                # 최종 디코딩
                decoded = tokenizer.decode(list(full_bytes))

                yield f"data: {decoded}\n\n"
                yield "data: [END]\n\n"

            # ---------------------------------------------------------
            # ★ need_rag == True → RAG 파이프라인 전체 수행
            # ---------------------------------------------------------
            yield "data: [STEP 2] 확장/검색 시작 (RAG)\n\n"

            expanded_text, kw_list = expand_query_kor(question)
            keywords = expand_variants(kw_list)
            yield f"data: [STEP 2] 확장 키워드={keywords}\n\n"

            t2 = time.time()
            hits = dense_retrieve_hybrid(qdr, emb, expanded_text, keywords)
            t3 = time.time()
            yield f"data: [STEP 3] hits={len(hits)} (t={t3 - t2:.2f}s)\n\n"

            ok, _ = rag_gate_decision(question, hits, kw_list, need_rag)

            if not ok:
                yield "data: [STEP 3b] 검색 스코어 낮음 → Chat으로 전환\n\n"
                prompt = build_chat_prompt(question)
                mode = "chat"
            else:
                mode = "rag"
                yield "data: [STEP 4] 문맥 구성 시작\n\n"
                boost_map = keyword_boost(hits, kw_list)
                reranked = rrf_rerank(hits, boost_map)
                final_hits = dedup_by_doc(reranked)
                ctx, refs = build_context_and_refs(final_hits)
                ctx = trim_context_to_budget(ctx)
                prompt = build_rag_prompt(ctx, question, refs)

            yield f"data: [STEP 4] 모드={mode}\n\n"

            # STEP 5: LLM 스트리밍
            yield "data: [STEP 5] LLM 스트리밍 시작\n\n"
            for chunk in triton_infer(prompt, stream=True):
                if chunk:
                    yield f"data: {chunk}\n\n"

            yield "data: [END]\n\n"

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            yield f"data: ⚠️ 오류: {err}\n\n"
            yield "data: [END]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
    )
# ---------------------------
# 로컬 실행
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    # 운영에서는 --workers 1 권장 (전역 커넥션 재사용 및 디버깅 편의)
    uvicorn.run(app, host="0.0.0.0", port=8082, reload=False)
