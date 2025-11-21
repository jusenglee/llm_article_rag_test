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

def get_tokenizer_for_model(model_name: str) -> AutoTokenizer:
    if model_name not in _tokenizers:
        tok_id = TOKENIZER_MAP[model_name]
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(
            tok_id, trust_remote_code=True
        )
    return _tokenizers[model_name]

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
    text, sparams = _make_inputs(
        prompt, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, stream=True
    )

    accumulated_text = ""
    for chunk in _triton_stream_generator(model_name, prompt, text, sparams, 10, 20):
        accumulated_text += chunk

    return accumulated_text.strip()

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
        return _triton_stream_generator(
            model_name, prompt, text, sparams,
            timeout_first, timeout_idle
        )

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
