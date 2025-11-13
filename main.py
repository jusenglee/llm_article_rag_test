import logging, nest_asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from query_pipeline import (
    build_rag_objects, triton_infer, expand_query_kor, expand_variants,
    dense_retrieve_hybrid, rag_gate_decision, build_chat_prompt,
    keyword_boost, rrf_rerank, dedup_by_doc, build_context_and_refs,
    trim_context_to_budget, build_rag_prompt, decide_rag_needed
)

# ---------------------------
# 초기 설정
# ---------------------------
nest_asyncio.apply()  # ✅ JupyterLab 호환을 위해 이벤트 루프 중첩 허용

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("uvicorn")  # uvicorn stdout과 동일한 채널

logger.info("🚀 Initializing RAG components...")
qdr, emb, retriever = build_rag_objects()
logger.info("✅ RAG pipeline ready.\n")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# 라우트
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_rag(request: Request, question: str = Form(...)):
    logs = []  # 사용자에게 보낼 로그 모음

    def log(msg):
        logger.info(msg)
        logs.append(msg)

    log(f"\n질문 > {question}")
    need_rag = decide_rag_needed(question)
    log(f"🧭 LLM 판단 결과: {'RAG 검색 수행' if need_rag else '일반 대화'}")

    expanded_text, kw_list = expand_query_kor(question)
    keywords = expand_variants(kw_list)
    log(f"🔑 확장 키워드: {keywords}")

    hits = dense_retrieve_hybrid(qdr, emb, expanded_text, keywords)
    log(f"🔍 검색된 문서 수(max 20): {len(hits)}")
    ok, gate_msg = rag_gate_decision(question, hits, kw_list, need_rag)
    log(gate_msg)

    if not ok:
        log("🤖 일반 대화 모드로 전환")
        prompt = build_chat_prompt(question)
        answer = triton_infer(prompt, stream=True)
        log(f"\n📘 답변:\n{answer.strip()}\n" + "-"*80)
        return JSONResponse({"mode": "chat", "answer": answer.strip(), "logs": logs})

    log("✅ 게이트 판단 결과: RAG 검색/응답 수행")
    boost_map = keyword_boost(hits, kw_list)
    reranked = rrf_rerank(hits, boost_map)
    final_hits = dedup_by_doc(reranked)
    ctx, refs = build_context_and_refs(final_hits)
    ctx = trim_context_to_budget(ctx)
    log(f"🔎 Retrieved top-{len(final_hits)} (after RRF+dedup).")

    rag_prompt = build_rag_prompt(ctx, question, refs)
    log("⚡ LLM generating response...")
    answer = triton_infer(rag_prompt, stream=True)
    log(f"\n📘 답변:\n{answer.strip()}\n" + "-"*80)

    return JSONResponse({"mode": "rag", "answer": answer.strip(), "refs": refs, "logs": logs})


# ---------------------------
# Jupyter / CLI 실행 모두 지원
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082, reload=False)
