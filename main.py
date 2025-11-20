# -*- coding: utf-8 -*-
import logging, nest_asyncio, time, traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from query_pipeline import (
    build_rag_objects_dual,
    triton_infer,
    expand_query_kor,
    dense_retrieve_hybrid,
    rrf_rerank,
    build_context,
    decide_rag_needed,
    tokenizer,
    COLLECTION,
    COLLECTION_B,
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
qdr_a = emb_a = retriever_a = None
qdr_b = emb_b = retriever_b = None


@app.on_event("startup")
async def init_rag():
    global qdr_a, emb_a, retriever_a, qdr_b, emb_b, retriever_b
    logger.info("🚀 Initializing RAG components (dual)...")
    qdr_a, emb_a, retriever_a, qdr_b, emb_b, retriever_b = build_rag_objects_dual()

    # Warmup: GPU/토크나이저/JIT lazy init 비용 제거 (A 스택 기준)
    try:
        _ = tokenizer.encode("warmup")
        _ = emb_a.get_text_embedding("warmup")
    except Exception as e:
        logger.info(f"Warmup skipped: {e}")
    logger.info("✅ RAG pipeline (A/B) ready.\n")


async def triton_stream_async(prompt: str):
    """
    triton_infer(stream=True) 제너레이터를 비동기 SSE용으로 감싸는 래퍼
    """
    import asyncio

    loop = asyncio.get_event_loop()
    gen = triton_infer(prompt, stream=True)

    i = 0
    while True:
        chunk = await loop.run_in_executor(None, next, gen, None)
        if chunk is None:
            logger.info("[DEBUG] RAW_CHUNK EOF")
            break

        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")

        # 🔴 여기서 실제 토큰 단위 출력 확인
        logger.info(f"[DEBUG] RAW_CHUNK[{i}]: {repr(text)}")
        i += 1

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

            # 공통: 질의 확장은 한 번만
            expanded_text = None
            kw_list = None

            # RAG 결과 컨텍스트
            context_a = ""
            refs_a = []
            context_b = ""
            refs_b = []

            # --- RAG or Chat 분기 ---
            if not need_rag:
                # 순수 Chat 모드
                yield "data: [STEP 2] RAG 스킵 → 일반 대화 진행\n\n"
            else:
                yield "data: [STEP 2] 확장/검색 시작 (RAG, A/B 비교)\n\n"

                # 2-1. 질의 확장 (한 번만)
                expanded_text, kw_list = expand_query_kor(question)
                yield f"data: [STEP 2] 확장 키워드={kw_list}\n\n"

                # ---------- A 스택 ----------
                t2a = time.time()
                try:
                    hits_a = dense_retrieve_hybrid(qdr_a, emb_a, expanded_text, kw_list, COLLECTION)
                    t3a = time.time()
                    yield f"data: [STEP 3A] A스택 hits={len(hits_a)} (t={t3a - t2a:.2f}s)\n\n"

                    if hits_a:
                        yield "data: [STEP 4A] A스택 문맥 구성 시작\n\n"
                        context_a, refs_a = build_context(hits_a)
                    else:
                        yield "data: [STEP 3A] A스택: 검색 결과 없음\n\n"
                except Exception as e:
                    yield f"data: [STEP 3A] A스택 검색 오류: {e}\n\n"

                # ---------- B 스택 ----------
                t2b = time.time()
                try:
                    hits_b = dense_retrieve_hybrid(qdr_b, emb_b, expanded_text, kw_list, COLLECTION_B)
                    t3b = time.time()
                    yield f"data: [STEP 3B] B스택 hits={len(hits_b)} (t={t3b - t2b:.2f}s)\n\n"

                    if hits_b:
                        yield "data: [STEP 4B] B스택 문맥 구성 시작\n\n"
                        context_b, refs_b = build_context(hits_b)
                    else:
                        yield "data: [STEP 3B] B스택: 검색 결과 없음\n\n"
                except Exception as e:
                    yield f"data: [STEP 3B] B스택 검색 오류: {e}\n\n"

            # -------- 프롬프트 빌드 & 스트리밍 --------

            # RAG가 필요 없거나, 둘 다 컨텍스트가 비었으면: 단일 Chat 모드
            if not need_rag or (not context_a and not context_b):
                sys_msg = (
                    "너는 친절하고 간결하게 답변하는 AI 어시스턴트야. "
                    "질문이 특별히 다른 언어를 요구하지 않는 한, 기본적으로 한국어로 답변해라."
                )
                user_msg = question

                try:
                    messages = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

                yield "data: [STEP 5] LLM 스트리밍 시작 (일반 Chat)\n\n"

                async for chunk in triton_stream_async(prompt):
                    text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    if text.strip():
                        yield f"data: {text}\n\n"

                yield "data: [END]\n\n"
                return

            # 여기부터는 RAG A/B 비교 모드

            # ======================================
            # A 스택 응답
            # ======================================
            if context_a:
                ref_lines_a = "\n".join(refs_a) if refs_a else "(출처 정보 없음)"


                sys_msg_a = (
                    "당신은 과학·기술 논문을 요약해서 한국어로 설명하는 전문 어시스턴트입니다.\n"
                    "- 반드시 제공된 컨텍스트(문서 발췌)에서만 근거를 사용하세요.\n"
                    "- 컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다.'라고만 말하고, 임의로 추측하지 마세요.\n"
                    "- 한국어 문장에서 정상적인 띄어쓰기를 사용하고, 단어들을 공백 없이 붙여 쓰지 마세요.\n"
                    "- 답변은 항상 ① 한 문단 요약 ② 번호가 있는 핵심 정리 목록 ③ '참고문헌' 섹션 순으로 작성하세요.\n"
                )
                user_msg_a = f"""
다음은 [A 스택]에서 검색한 관련 문서 발췌입니다. 각 문단 앞의 번호는 출처 번호입니다.

[컨텍스트 발췌 시작]
{context_a}
[컨텍스트 발췌 끝]

(출처 번호 매핑)
{ref_lines_a}

사용자의 질문:
{question}

답변 형식 가이드라인(아주 중요):
1. 첫 문단에 2~3문장으로 전체 내용을 한국어로 요약합니다.
2. 그 다음에는 "1. 소제목" 형식의 번호 매기기 목록으로 핵심 내용을 정리합니다.
   - 각 항목은 "1. 소제목 [1][3]" 처럼 관련 출처 번호를 대괄호로 표기합니다.
   - 소제목 아래 줄에서 2~4문장 정도로 설명을 덧붙입니다.
3. 문장 중간에 근거를 달 때는 "…라는 점이 보고되었습니다[1][3]."처럼 [1] 형태의 인용 번호를 사용합니다.
4. 한국어 문장 사이에는 일반적인 띄어쓰기를 유지하고,
   '의학기술의최신동향은'처럼 단어를 모두 붙여 쓰지 말고
   '의학 기술의 최신 동향은'처럼 자연스러운 띄어쓰기를 사용하세요.
5. 마지막에는 아래 예시처럼 참고문헌 섹션을 추가합니다.

참고문헌:
[1] 논문 제목A
[2] 논문 제목B
[3] 논문 제목C

위 형식을 최대한 정확하게 지키면서 답변하세요.
"""

                try:
                    messages_a = [
                        {"role": "system", "content": sys_msg_a},
                        {"role": "user", "content": user_msg_a},
                    ]

                    logger.info("===== [DEBUG] PROMPT_A_HEAD =====")
                    logger.info(messages_a[:400])
                    logger.info("===== [DEBUG] PROMPT_A_TAIL =====")
                    logger.info(messages_a[-400:])
                    prompt_a = tokenizer.apply_chat_template(
                        messages_a, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_a = f"<|system|>\n{sys_msg_a}\n<|user|>\n{user_msg_a}\n<|assistant|>\n"

                yield "data: \n\n"
                yield "data: =============================\n\n"
                yield "data: [RAG-A] 임베딩/벡터DB 스택 A 응답\n\n"
                yield "data: =============================\n\n"

                async for chunk in triton_stream_async(prompt_a):
                    text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    if text.strip():
                        yield f"data: [A] {text}\n\n"

            # ======================================
            # B 스택 응답
            # ======================================
            if context_b:
                ref_lines_b = "\n".join(refs_b) if refs_b else "(출처 정보 없음)"

                sys_msg_b = (
                    "당신은 과학/기술 문서를 기반으로 답변하는 한국어 LLM입니다. "
                    "반드시 제공된 컨텍스트에서만 근거를 사용하세요. "
                    "이 응답은 [B 스택] 검색 결과를 기반으로 합니다."
                )
                user_msg_b = f"""
다음은 [B 스택]에서 검색한 관련 문서 발췌입니다(번호=출처):
{context_b}

(출처 번호 매핑)
{ref_lines_b}

질문: {question}

요구사항:
- 문장 내에 [1], [2] 형태로 근거 번호를 달아주세요.
- 컨텍스트에 없는 내용은 쓰지 마세요(추가 지식 금지).
- 마지막 줄에 '참고문헌:' 뒤에 논문 제목을 함께 나열하세요. 예시: 참고문헌: [1] 제목A, [2] 제목B
"""

                try:
                    messages_b = [
                        {"role": "system", "content": sys_msg_b},
                        {"role": "user", "content": user_msg_b},
                    ]

                    logger.info("===== [DEBUG] PROMPT_B_HEAD =====")
                    logger.info(messages_b[:400])
                    logger.info("===== [DEBUG] PROMPT_B_TAIL =====")
                    logger.info(messages_b[-400:])
                    prompt_b = tokenizer.apply_chat_template(
                        messages_b, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_b = f"<|system|>\n{sys_msg_b}\n<|user|>\n{user_msg_b}\n<|assistant|>\n"

                yield "data: \n\n"
                yield "data: =============================\n\n"
                yield "data: [RAG-B] 임베딩/벡터DB 스택 B 응답\n\n"
                yield "data: =============================\n\n"

                async for chunk in triton_stream_async(prompt_b):
                    text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    if text.strip():
                        yield f"data: [B] {text}\n\n"

            # 최종 종료
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
