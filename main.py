# -*- coding: utf-8 -*-
import logging, nest_asyncio, time, traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from query_pipeline import (
    build_rag_objects,
    triton_infer,
    expand_query_kor,
    dense_retrieve_hybrid,
    rrf_rerank,
    build_context,
    decide_rag_needed,
    tokenizer,
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
        _ = tokenizer.encode("warmup")
        _ = emb.get_text_embedding("warmup")
    except Exception as e:
        logger.info(f"Warmup skipped: {e}")
    logger.info("✅ RAG pipeline ready.\n")


async def triton_stream_async(prompt: str):
    """
    triton_infer(stream=True) 제너레이터를 비동기 SSE용으로 감싸는 래퍼
    """
    import asyncio

    loop = asyncio.get_event_loop()
    gen = triton_infer(prompt, stream=True)

    while True:
        chunk = await loop.run_in_executor(None, next, gen, None)
        if chunk is None:
            break
        # query_pipeline 쪽에서 이미 str로 줘도 방어적으로 처리
        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
        yield text


# ---------------------------
# 라우트
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------
# SSE STREAM
# ---------------------------
@app.get("/query/stream")
async def query_stream(question: str):

    async def event_gen():
        import asyncio

        try:
            yield "data: [STEP 0] 질문 수신\n\n"
            await asyncio.sleep(0)

            # STEP 1: 게이트 (RAG / Chat 판단)
            t0 = time.time()
            need_rag = decide_rag_needed(question)
            t1 = time.time()
            yield f"data: [STEP 1] 게이트={need_rag} (t={t1 - t0:.2f}s)\n\n"

            context_str = ""
            refs = []

            # --- RAG or Chat 분기 ---
            if not need_rag:
                # 순수 Chat 모드
                yield "data: [STEP 2] RAG 스킵 → 일반 대화 진행\n\n"
            else:
                yield "data: [STEP 2] 확장/검색 시작 (RAG)\n\n"

                # 2-1. 질의 확장
                expanded_text, kw_list = expand_query_kor(question)
                yield f"data: [STEP 2] 확장 키워드={kw_list}\n\n"

                # 2-2. 벡터 검색
                t2 = time.time()
                hits = dense_retrieve_hybrid(qdr, emb, expanded_text, kw_list)
                t3 = time.time()
                yield f"data: [STEP 3] hits={len(hits)} (t={t3 - t2:.2f}s)\n\n"

                if not hits:
                    yield "data: [STEP 3b] 검색 결과 없음 → Chat으로 전환\n\n"
                else:
                    # 2-3. 재랭킹 + 컨텍스트 구성
                    yield "data: [STEP 4] 문맥 구성 시작\n\n"
                    reranked = rrf_rerank(hits, kw_list)
                    context_str, refs = build_context(reranked)

            # 3. 프롬프트 빌드
            # --- RAG 모드 프롬프트 ---
            ref_lines = ""
            if context_str:
                if refs:
                    ref_lines = "\n".join(refs)
                else:
                    ref_lines = "(출처 정보 없음)"

                sys_msg = (
                    "당신은 사용자를 보조하는 LLM입니다. 반드시 제공된 컨텍스트에서만 근거를 사용하세요. "
                    "컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다'라고만 말하고, 추측하지 마세요. "
                    "가능하면 문장 내에 근거 번호 각주를 표시하세요."
                )
                user_msg = f"""
                다음은 관련 문서 발췌입니다(번호=출처):
                {context_str}
                
                (출처 번호 매핑)
                {ref_lines}
                
                질문: {question}
                
                요구사항:
                - 문장 내에 [1], [2] 형태로 근거 번호를 달아주세요.
                - 컨텍스트에 없는 내용은 쓰지 마세요(추가 지식 금지).
                - 마지막 줄에 '참고문헌:' 뒤에 논문 제목을 함께 나열하세요. 예시: 참고문헌: [1] 제목A, [2] 제목B
                """

            # --- 일반 Chat 프롬프트 ---
            else:
                sys_msg = (
                    "너는 친절하고 간결하게 답변하는 AI 어시스턴트야. "
                    "질문이 특별히 다른 언어를 요구하지 않는 한, 기본적으로 **한국어**로 답변해라."
                )
                user_msg = question

            # tokenizer가 있는 경우 chat template 활용
            try:
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # 템플릿 미지원 모델일 때 fallback
                prompt = f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

            # -------------------------------------------------
            # STEP 5 (공통 스트리밍)
            # -------------------------------------------------
            yield "data: [STEP 5] LLM 스트리밍 시작\n\n"

            async for chunk in triton_stream_async(prompt):
                # 여기서는 chunk가 이미 UTF-8 safe str 이라고 가정
                text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                if text.strip():
                    # SSE 형식
                    yield f"data: {text}\n\n"

            # 스트림 종료 신호
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
        },
    )


# ---------------------------
# 로컬 실행
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    # 운영에서는 --workers 1 권장 (전역 커넥션 재사용 및 디버깅 편의)
    uvicorn.run(app, host="0.0.0.0", port=8082, reload=False)
