# rag_pipeline/triton_client.py
import json
import threading
import time
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from settings import (
    TRITON_URL, TOKENIZER_MAP, MAX_TOKENS, TEMPERATURE, TOP_P,
)
from settings import logger  # 공용 logger

_triton_client: InferenceServerClient | None = None
_tokenizers: Dict[str, AutoTokenizer] = {}

ASSISTANT_FINAL_MARKER = "assistantfinal"


def _is_gpt_oss_model(model_name: str) -> bool:
    """
    gpt-oss 계열 모델 판별용 헬퍼.
    - 이름 규칙은 상황에 따라 다를 수 있으니 필요시 수정.
    """
    name = model_name.lower()
    return ("gpt" in name) and ("oss" in name)

def get_tokenizer_for_model(model_name: str) -> AutoTokenizer:
    if model_name not in _tokenizers:
        tok_id = TOKENIZER_MAP[model_name]
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(
            tok_id, trust_remote_code=True
        )
    return _tokenizers[model_name]


def _get_prompt_tokens(model_name: str, prompt: str) -> int:
    """
    주어진 모델 기준으로 프롬프트 토큰 길이 계산.
    토크나이저 문제 발생 시에는 len(prompt)로 아주 러프하게 fallback.
    """
    try:
        tok = get_tokenizer_for_model(model_name)
        # special token은 이미 시스템 프롬프트에 들어가 있을 가능성 있으니 False
        ids = tok.encode(prompt, add_special_tokens=False)
        return len(ids)
    except Exception as e:
        logger.warning(f"[TRITON] prompt token 계산 실패, fallback 사용: {e}")
        # 완전 비었으면 0, 아니면 글자 수 기준 러프 추정
        return max(1, len(prompt) // 2)


def _compute_max_new_tokens(
        model_name: str,
        prompt: str,
        max_tokens_hint: int | None = None,
) -> int:
    """
    - 토크나이저의 model_max_length(없으면 8192 추정)를 기준으로
      prompt_tokens + max_new_tokens <= max_seq_len - margin 을 만족하도록 조정.
    - max_tokens_hint(= 인자로 받은 max_tokens)는 상한(cap)으로만 사용.
    """
    prompt_tokens = _get_prompt_tokens(model_name, prompt)

    try:
        tok = get_tokenizer_for_model(model_name)
        max_seq_len = getattr(tok, "model_max_length", 8192)
        # HF 쪽에서 종종 엄청 큰 값(1e30 같은) 넣어두는 경우 방어
        if max_seq_len is None or max_seq_len > 100_000:
            max_seq_len = 8192
    except Exception:
        max_seq_len = 8192

    SAFETY_MARGIN = 256      # 여유 버퍼
    MIN_NEW_TOKENS = 64      # 최소 생성 토큰

    # settings.MAX_TOKENS 를 기본 상한으로, 인자로 들어오면 그것으로 override
    cap = int(max_tokens_hint) if max_tokens_hint is not None else int(MAX_TOKENS)

    available = max_seq_len - prompt_tokens - SAFETY_MARGIN
    if available <= 0:
        logger.warning(
            f"[TRITON] prompt가 이미 max_seq_len을 거의 다 쓴 상태입니다: "
            f"prompt_tokens={prompt_tokens}, max_seq_len={max_seq_len}"
        )
        # 그래도 최소한 조금은 생성하도록
        return max(MIN_NEW_TOKENS, min(cap, 128))

    max_new = min(cap, available)
    return max(MIN_NEW_TOKENS, max_new)


def extract_final_answer(raw: str) -> str:
    """gpt-oss가 analysis/assistantfinal 포맷으로 뱉을 때, 최종 답변만 추출."""
    if not raw:
        return ""

    text = str(raw).strip()
    idx = text.rfind(ASSISTANT_FINAL_MARKER)
    if idx == -1:
        # 마커 없으면 그냥 원본 반환 (다른 모델 대비 안전장치)
        return text

    final = text[idx + len(ASSISTANT_FINAL_MARKER):]
    # 콜론/공백 정리
    final = final.lstrip(" :\n\t")
    logger.info(final)
    return final.strip()

def stream_after_assistantfinal(chunks):
    """
    gpt-oss 스트리밍 결과(chunks)를 받아서
    'assistantfinal' 이후 텍스트만 yield하는 제너레이터.
    """
    marker = ASSISTANT_FINAL_MARKER.lower()
    seen = False
    buf = ""

    for chunk in chunks:
        if not chunk:
            continue

        buf += chunk

        if not seen:
            pos = buf.lower().find(marker)
            if pos == -1:
                # 아직 마커 안 나왔으면 계속 버퍼에만 쌓음
                continue

            # 처음으로 마커를 발견한 시점
            seen = True
            start = pos + len(ASSISTANT_FINAL_MARKER)
            # 마커 앞부분은 버리고, 마커 뒤부터 사용
            buf = buf[start:]
            buf = buf.lstrip(" :\n\t")

            if not buf:
                continue

        # 여기부터는 전부 '최종 답변'에 해당
        yield buf
        buf = ""

    # 스트림 종료 후 마무리 처리
    if seen and buf:
        # assistantfinal 이후 남은 찌꺼기
        yield buf
    elif not seen and buf:
        # assistantfinal이 한 번도 안 나온 경우 fallback:
        # 전체 버퍼를 그냥 보내거나, 정책에 따라 버릴 수도 있음.
        logger.warning(
            "[gpt-oss] assistantfinal 마커를 찾지 못했습니다. 전체 버퍼를 그대로 전송합니다."
        )
        yield buf

def get_triton_client() -> InferenceServerClient:
    """Triton Client Singleton"""
    global _triton_client
    if _triton_client is None:
        try:
            _triton_client = InferenceServerClient(url=TRITON_URL, verbose=False)
            logger.info(f"✅ Triton Client connected to {TRITON_URL}")
        except Exception as e:
            logger.error(f"❌ Triton Client connection failed: {e}")
            raise e
    return _triton_client

def _make_inputs(
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
):
    text = InferInput("text_input", [1], "BYTES")
    text.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    # vLLM-backend 권장 명칭: sampling_parameters
    sparams = InferInput("sampling_parameters", [1], "BYTES")

    # vLLM Python backend 쪽 구현이 문자열 기반 파싱을 사용하는 경우가 많음
    params = {
        "temperature": str(float(temperature)),
        "top_p": str(float(top_p)),
        "max_tokens": str(int(max_tokens)),
        # stream 플래그는 별도 BOOL 인풋으로 전달 중
    }

    sparams.set_data_from_numpy(
        np.array([json.dumps(params).encode("utf-8")], dtype=object)
    )
    return text, sparams

def _triton_stream_generator(
        model_name: str,
        prompt: str,
        text: InferInput,
        sparams: InferInput,
        first_token_timeout: int = 10,
        idle_timeout: int = 20,
):
    cli = get_triton_client()  # Triton client 재사용

    stream_flag = InferInput("stream", [1], "BOOL")
    stream_flag.set_data_from_numpy(np.array([True], dtype=bool))
    outs = [InferRequestedOutput("text_output")]

    q: List[str] = []
    done = threading.Event()

    def on_resp(result, error):
        if error:
            logger.error(f"[ERR] Triton Callback Error: {error}")
            done.set()
            return

        if result is None:
            done.set()
            return

        arr = result.as_numpy("text_output")
        if arr is not None and len(arr) > 0:
            raw = arr[0]
            chunk = raw.decode("utf-8", errors="ignore")
            q.append(chunk)

        is_final = False
        try:
            resp = result.get_response()
            params = getattr(resp, "parameters", None)
            if params:
                flag = params.get("triton_final_response")
                if flag and getattr(flag, "bool_param", False):
                    is_final = True
        except Exception as e:
            logger.debug(f"[STREAM] triton_final_response check failed: {e}")

        if is_final:
            done.set()
            return

    # 스트림 시작
    cli.start_stream(callback=on_resp)
    cli.async_stream_infer(model_name, inputs=[text, sparams, stream_flag], outputs=outs)

    try:
        start_time = time.time()
        last_yield_time = start_time
        got_first = False

        while not done.is_set() or q:
            if q:
                chunk = q.pop(0)
                got_first = True
                last_yield_time = time.time()
                yield chunk
            else:
                now = time.time()
                if not got_first and (now - start_time > first_token_timeout):
                    logger.warning("[WARN] First token timeout")
                    break
                if got_first and (now - last_yield_time > idle_timeout):
                    logger.warning("[WARN] Idle timeout after response started")
                    break
                time.sleep(0.005)
    finally:
        cli.stop_stream()

def _triton_infer_sync(
        model_name: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
) -> str:
    # 🔹 여기서 동적으로 max_new_tokens 계산
    dynamic_max_tokens = _compute_max_new_tokens(
        model_name=model_name,
        prompt=prompt,
        max_tokens_hint=max_tokens,
    )

    text, sparams = _make_inputs(
        prompt, max_tokens=dynamic_max_tokens,
        temperature=temperature, top_p=top_p, stream=True
    )

    accumulated_text = ""
    for chunk in _triton_stream_generator(model_name, prompt, text, sparams, 10, 20):
        accumulated_text += chunk

    accumulated_text = accumulated_text.strip()

    # gpt-oss 계열이면 assistantfinal 이후만 추출
    if _is_gpt_oss_model(model_name):
        return extract_final_answer(accumulated_text)

    return accumulated_text

def triton_infer(
        model_name: str,
        prompt: str,
        *,
        stream: bool = True,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        timeout_first: int = 20,
        timeout_idle: int = 120,
):
    logger.info(f"[TRITON] infer start - model={model_name}, len={len(prompt)}")

    if stream:
        text, sparams = _make_inputs(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, stream=True
        )

        base_gen = _triton_stream_generator(
            model_name, prompt, text, sparams,
            timeout_first, timeout_idle
        )

        # gpt-oss 계열이면 assistantfinal 이후만 스트리밍
        if _is_gpt_oss_model(model_name):
            logger.info("[TRITON] gpt-oss 모델 감지 → assistantfinal 이후만 스트리밍")
            return stream_after_assistantfinal(base_gen)

        # 그 외 모델은 raw 스트림 그대로
        return base_gen

    # Sync Path
    return _triton_infer_sync(
        model_name,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

def ensure_single_model_loaded(target_model: str, timeout: float = 120.0) -> None:
    """
    1) 레포지토리 인덱스를 보고 target 이외 모델은 모두 unload
    2) target 모델이 READY 상태가 아니면 load + READY 될 때까지 대기
    """
    cli = get_triton_client()

    # 1. 모델 레포지토리 인덱스 조회
    try:
        repo = cli.get_model_repository_index()
    except Exception as e:
        logger.error(f"[TRITON] get_model_repository_index failed: {e}")
        raise

    # 2. target 외 모델 unload
    for m in getattr(repo, "models", []):
        name = getattr(m, "name", None)
        if not name or name == target_model:
            continue
        try:
            if cli.is_model_ready(name):
                logger.info(f"[TRITON] unloading other model: {name}")
                cli.unload_model(name)
        except Exception as e:
            logger.warning(f"[TRITON] unload_model({name}) failed: {e}")

    # 3. target 이 이미 READY면 바로 리턴
    try:
        if cli.is_model_ready(target_model):
            logger.info(f"[TRITON] target model {target_model} already READY")
            return
    except Exception as e:
        logger.warning(f"[TRITON] is_model_ready({target_model}) error: {e}")

    # 4. target load
    logger.info(f"[TRITON] loading model: {target_model}")
    cli.load_model(target_model)

    # 5. READY 될 때까지 polling
    start = time.time()
    while True:
        try:
            if cli.is_model_ready(target_model):
                logger.info(f"[TRITON] model {target_model} READY")
                return
        except Exception as e:
            logger.warning(f"[TRITON] is_model_ready({target_model}) check failed: {e}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout while waiting for model {target_model} to be READY")

        time.sleep(0.5)

def unload_model_safe(model_name: str) -> None:
    cli = get_triton_client()
    try:
        if cli.is_model_ready(model_name):
            logger.info(f"[TRITON] unloading model: {model_name}")
            cli.unload_model(model_name)
    except Exception as e:
        logger.warning(f"[TRITON] unload_model({model_name}) failed: {e}")
