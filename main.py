# -*- coding: utf-8 -*-
import logging, nest_asyncio, time, traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from rag_pipeline import decide_rag_needed, run_rag_ab_compare
from rag_store import build_rag_objects_dual
from retrieval import expand_query_kor, dense_retrieve_hybrid, rrf_rerank, build_context
from settings import COLLECTION, COLLECTION_B, TRITON_URL
from triton_client import triton_infer, get_tokenizer_for_model, ensure_single_model_loaded, unload_model_safe, \
    get_triton_client

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

# 클라이언트에서 오는 model 키 → Triton 모델 이름 매핑
MODEL_MAP = {
    "gpt": "gpt_oss_0",
    "gemma": "gemma_vllm_0",
    # EXAONE 붙이면 여기
     "exaone": "EXAONE_0",
}


@app.on_event("startup")
async def init_rag():
    global qdr_a, emb_a, retriever_a, qdr_b, emb_b, retriever_b
    logger.info("🚀 Initializing RAG components (dual)...")
    qdr_a, emb_a, retriever_a, qdr_b, emb_b, retriever_b = build_rag_objects_dual()
    logger.info("✅ RAG pipeline (A/B) ready.\n")


async def triton_stream_async(model_name: str, prompt: str):
    """
    triton_infer(stream=True) 제너레이터를 비동기 SSE용으로 감싸는 래퍼
    - 반드시 model_name을 인자로 받아서, 어떤 모델을 쓸지 FastAPI에서 결정
    """
    import asyncio

    loop = asyncio.get_event_loop()
    gen = triton_infer(model_name, prompt, stream=True)

    i = 0
    while True:
        chunk = await loop.run_in_executor(None, next, gen, None)
        if chunk is None:
            logger.info("[DEBUG] RAW_CHUNK EOF")
            break

        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")

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
async def query_stream(question: str, model: str = "gpt"):
    """
    클라이언트에서:
      /query/stream?model=gpt&question=...
    이런 식으로 호출 (HTML에서 select 박스로 model 값을 넘김)

    1) model 키 → Triton 모델 이름 변환
    2) ensure_single_model_loaded(model_name) 호출
    3) RAG / Chat 프롬프트 생성 + triton_stream_async(model_name, ...)
    4) 끝나면 finally에서 unload_model_safe(model_name)
    """

    model_key = model.lower()
    if model_key not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    model_name = MODEL_MAP[model_key]
    logger.info(f"[QUERY] model_key={model_key}, model_name={model_name}, question={question!r}")
    tok = get_tokenizer_for_model(model_name)

    async def event_gen():
        import asyncio

        await asyncio.sleep(0)  # 이벤트 루프 양보용

        # 0. Triton에서 모델 로드 과정을 사용자에게 그대로 노출
        try:
            cli = get_triton_client()

            # 0-1) 레포지토리 인덱스 조회
            yield f"data: [MODEL] Triton 연결 ({TRITON_URL}) 후 모델 로드 시도 중...\n\n"
            repo = cli.get_model_repository_index()
            names = [getattr(m, "name", "?") for m in getattr(repo, "models", [])]
            yield f"data: [MODEL] 현재 등록된 모델: {', '.join(names)}\n\n"

            # 0-2) target 이외 모델 UNLOAD
            for m in getattr(repo, "models", []):
                name = getattr(m, "name", None)
                if not name or name == model_name:
                    continue
                try:
                    if cli.is_model_ready(name):
                        yield f"data: [MODEL] 다른 모델 언로드 요청: {name}\n\n"
                        cli.unload_model(name)
                        yield f"data: [MODEL] 언로드 완료: {name}\n\n"
                except Exception as e:
                    logger.warning(f"[TRITON] unload_model({name}) failed: {e}")
                    yield f"data: [MODEL] 언로드 실패({name}): {type(e).__name__}: {e}\n\n"

            # 0-3) target 모델 상태 확인
            try:
                if cli.is_model_ready(model_name):
                    yield f"data: [MODEL] {model_name} 이미 READY 상태입니다.\n\n"
                else:
                    # 0-4) target 모델 로드 시작
                    yield f"data: [MODEL] {model_name} 로드 시작...\n\n"
                    cli.load_model(model_name)
                    start = time.time()
                    timeout = 120.0

                    # 0-5) READY 될 때까지 polling + 진행 상황 SSE 전송
                    while True:
                        await asyncio.sleep(0.5)
                        elapsed = time.time() - start

                        try:
                            if cli.is_model_ready(model_name):
                                yield f"data: [MODEL] {model_name} READY (t={elapsed:.2f}s)\n\n"
                                break
                        except Exception as e:
                            logger.warning(f"[TRITON] is_model_ready({model_name}) check failed: {e}")
                            yield f"data: [MODEL] 상태 확인 실패: {type(e).__name__}: {e}\n\n"

                        if elapsed > timeout:
                            raise TimeoutError(
                                f"Timeout while waiting for model {model_name} to be READY"
                            )

                        # 진행 중인 상태도 계속 쏴줌
                        yield f"data: [MODEL] {model_name} 로딩 중... (elapsed={elapsed:.1f}s)\n\n"

            except Exception as e:
                raise e

        except Exception as e:
            err = f"Triton 모델 로드 단계에서 오류: {type(e).__name__}: {e}"
            traceback.print_exc()
            yield f"data: ⚠️ {err}\n\n"
            yield "data: [END]\n\n"
            return

        try:
            yield "data: [STEP 0] 질문 수신\n\n"

            # 1: 게이트 (RAG / Chat 판단)
            t0 = time.time()
            need_rag = decide_rag_needed(question, model_name=model_name)
            t1 = time.time()
            yield f"data: [STEP 1] 게이트={need_rag} (t={t1 - t0:.2f}s)\n\n"

            # 질의 확장은 한 번만
            expanded_text = None
            kw_list = None

            # RAG 결과 컨텍스트
            context_a = ""
            refs_a = []
            context_b = ""
            refs_b = []

            # RAG or Chat 분기
            if not need_rag:
                # 순수 Chat 모드
                yield "data: [STEP 2] RAG 스킵 → 일반 대화 진행\n\n"
            else:
                yield "data: [STEP 2] 확장/검색 시작 (RAG, A/B 비교)\n\n"

                # 여기서 전체 RAG A/B 비교 한 번에 수행
                res_map = run_rag_ab_compare(
                    query=question,
                    with_llm=False,          # 여기서는 컨텍스트까지만, LLM은 아래에서 스트리밍
                    model_name=model_name,
                )
                res_a = res_map["A"]
                res_b = res_map["B"]

                # 확장 쿼리 / 키워드 로그
                yield f"data: [EXPAND] 확장 쿼리(A기준) = {res_a.expanded_query}\n\n"
                yield f"data: [EXPAND] 키워드(A기준) = {res_a.keywords}\n\n"

                # 성능 타이밍을 SSE로 전송
                ta = res_a.timings
                tb = res_b.timings

                yield (
                    "data: [PERF-A] "
                    f"확장(expand)={ta.get('expand_query', 0.0):.3f}s, "
                    f"검색(dense_total)={ta.get('dense_search', 0.0):.3f}s, "
                    f"리랭크(rerank)={ta.get('rerank', 0.0):.3f}s, "
                    f"컨텍스트(ctx)={ta.get('build_context', 0.0):.3f}s\n\n"
                )
                yield (
                    "data: [PERF-A] "
                    f"확장(expand)={tb.get('expand_query', 0.0):.3f}s, "
                    f"검색(dense_total)={tb.get('dense_search', 0.0):.3f}s, "
                    f"리랭크(rerank)={tb.get('rerank', 0.0):.3f}s, "
                    f"컨텍스트(ctx)={tb.get('build_context', 0.0):.3f}s\n\n"
                )

                # 상위 문서 목록도 SSE로 전송
                yield "data: [HITS-A] ----- A 스택 상위 문서 목록 -----\n\n"
                for i, h in enumerate(res_a.reranked_hits[:5], start=1):
                    raw = (h.payload or {}).get("_node_text") or ""
                    title = " ".join(str(raw).splitlines())
                    yield f"data: [HITS-A] [{i}] {title}\n\n"

                yield "data: [HITS-B] ----- B 스택 상위 문서 목록 -----\n\n"
                for i, h in enumerate(res_b.reranked_hits[:5], start=1):
                    raw = (h.payload or {}).get("_node_text") or ""
                    title = " ".join(str(raw).splitlines())
                    yield f"data: [HITS-B] [{i}] {title}\n\n"

                context_a, refs_a = res_a.context, res_a.refs
                context_b, refs_b = res_b.context, res_b.refs
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
                    prompt = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

                yield f"data: [STEP 5] LLM 스트리밍 시작 (일반 Chat, model={model_name})\n\n"

                async for chunk in triton_stream_async(model_name, prompt):
                    text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    if text.strip():
                        yield f"data: {text}\n\n"

                yield "data: [END]\n\n"
                return

            # RAG A/B 비교
            # ---------- A 스택 응답 ----------
            if context_a:
                ref_lines_a = "\n".join(refs_a) if refs_a else "(출처 정보 없음)"

                sys_msg_a = (
                    "당신은 과학·기술 논문을 요약해서 한국어로 설명하는 전문 어시스턴트입니다.\n"
                    "- 반드시 제공된 컨텍스트(문서 발췌)에서만 근거를 사용하세요.\n"
                    "- 컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다.'라고만 말하고, 임의로 추측하지 마세요.\n"
                    "- 한국어 문장에서 정상적인 띄어쓰기를 사용하세요.\n"
                    "- 그 안쪽에만 실제 한국어 답변을 작성하고, 그 밖에는 어떤 분석/설명/계획 문장도 쓰지 마세요.\n"
                    "이 응답은 [A 스택] 검색 결과를 기반으로 합니다."
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
2. 그 다음에는 "1. 소제목" 형식의 번호 매기기 목록으로 핵심 내용을 정리합니다. 소제목은 내용을 압축하여 임의로 작성하세요
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
"""

                try:
                    messages_a = [
                        {"role": "system", "content": sys_msg_a},
                        {"role": "user", "content": user_msg_a},
                    ]
                    prompt_a = tok.apply_chat_template(
                        messages_a, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_a = f"<|system|>\n{sys_msg_a}\n<|user|>\n{user_msg_a}\n<|assistant|>\n"

                yield "data: \n\n"
                yield "data: =============================\n\n"
                yield "data: [RAG-A] 임베딩/벡터DB 스택 A 응답\n\n"
                yield "data: =============================\n\n"

                async for chunk in triton_stream_async(model_name, prompt_a):
                    text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    if text.strip():
                        yield f"data: [A] {text}\n\n"

            # ---------- B 스택 응답 ----------
            if context_b:
                ref_lines_b = "\n".join(refs_b) if refs_b else "(출처 정보 없음)"

                sys_msg_b = (
                    "당신은 과학·기술 논문을 요약해서 한국어로 설명하는 전문 어시스턴트입니다.\n"
                    "- 반드시 제공된 컨텍스트(문서 발췌)에서만 근거를 사용하세요.\n"
                    "- 컨텍스트에 없으면 '제공된 문서에서 찾지 못했습니다.'라고만 말하고, 임의로 추측하지 마세요.\n"
                    "- 한국어 문장에서 정상적인 띄어쓰기를 사용하세요.\n"
                    "- 그 안쪽에만 실제 한국어 답변을 작성하고, 그 밖에는 어떤 분석/설명/계획 문장도 쓰지 마세요.\n"
                    "이 응답은 [B 스택] 검색 결과를 기반으로 합니다."
                )
                user_msg_b = f"""
다음은 [B 스택]에서 검색한 관련 문서 발췌입니다. 각 문단 앞의 번호는 출처 번호입니다.

[컨텍스트 발췌 시작]
{context_b}
[컨텍스트 발췌 끝]

(출처 번호 매핑)
{ref_lines_b}

사용자의 질문:
{question}

답변 형식 가이드라인(아주 중요):
1. 첫 문단에 2~3문장으로 전체 내용을 한국어로 요약합니다.
2. 그 다음에는 "1. 소제목" 형식의 번호 매기기 목록으로 핵심 내용을 정리합니다. 소제목은 내용을 압축하여 임의로 작성하세요
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

"""

                try:
                    messages_b = [
                        {"role": "system", "content": sys_msg_b},
                        {"role": "user", "content": user_msg_b},
                    ]
                    prompt_b = tok.apply_chat_template(
                        messages_b, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_b = f"<|system|>\n{sys_msg_b}\n<|user|>\n{user_msg_b}\n<|assistant|>\n"

                yield "data: \n\n"
                yield "data: =============================\n\n"
                yield "data: [RAG-B] 임베딩/벡터DB 스택 B 응답\n\n"
                yield "data: =============================\n\n"

                async for chunk in triton_stream_async(model_name, prompt_b):
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

        finally:
            # 이 요청에서 사용한 모델은 무조건 내려준다 (한 번에 하나 전략)
            try:
                logger.info(f"[TRITON] unload_model_safe({model_name})")
                unload_model_safe(model_name)
            except Exception as e:
                logger.warning(f"[TRITON] unload_model_safe({model_name}) failed: {e}")

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
