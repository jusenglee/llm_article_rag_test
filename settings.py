# rag_pipeline/settings.py
import os
import logging
from pathlib import Path

# 로깅
logger = logging.getLogger("RAG_Pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAG_Pipeline")

# Qdrant / Embedding (A)
QDRANT_HOST  = os.getenv("QDRANT_HOST", "211.241.177.73")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6334))
COLLECTION   = os.getenv("QDRANT_COLLECTION", "e5_instruct_rag_100_000")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

# Qdrant / Embedding (B)
QDRANT_HOST_B  = os.getenv("QDRANT_HOST_B", QDRANT_HOST)
QDRANT_PORT_B  = int(os.getenv("QDRANT_PORT_B", QDRANT_PORT))
COLLECTION_B   = os.getenv("QDRANT_COLLECTION_B", "e5_rag_100_000")
EMBED_MODEL_B  = os.getenv("EMBEDDING_MODEL_B", "intfloat/multilingual-e5-large")

# Triton
TRITON_URL         = os.getenv("TRITON_URL", "211.241.177.73:8001")
DEFAULT_MODEL_NAME = os.getenv("TRITON_MODEL", "gpt_oss_0")
TOKENIZER_MAP = {
    "gpt_oss_0": "openai/gpt-oss-20b",
    "gemma_vllm_0": "./data/gemma_vllm_0",
}

# 하이퍼파라미터
TOP_K_BASE       = 50
TOP_K_RETURN     = 20
MAX_TOKENS       = 8192
TEMPERATURE      = 0.6
TOP_P            = 0.9
SCORE_THRESHOLD  = 0.15
FUZZ_MIN         = 40
CTX_TOKEN_BUDGET = 4096
SNIPPET_MAX_CHARS = 4096

# 벤치 로그
LOG_DIR = Path(os.getenv("RAG_BENCH_LOG_DIR", "./rag_bench_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
