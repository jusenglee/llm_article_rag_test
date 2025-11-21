# rag_pipeline/bench_logger.py
import json
import time
from typing import Dict

from .settings import LOG_DIR, EMBED_MODEL, EMBED_MODEL_B
from .rag_types import RagResult

def log_ab_result_to_file(query: str, res_map: Dict[str, RagResult]):
    ts = int(time.time())
    for _, res in res_map.items():
        out = {
            "timestamp": ts,
            "stack": res.stack,
            "embedding_model": EMBED_MODEL if res.stack == "A" else EMBED_MODEL_B,
            "query": query,
            "expanded_query": res.expanded_query,
            "keywords": res.keywords,
            "timings": res.timings,
            "refs": res.refs,
            "llm_answer": res.llm_answer,
            "top_hits": [
                {
                    "id": h.id,
                    "score": float(h.score) if h.score is not None else 0.0,
                    "payload_doc_id": (h.payload or {}).get("doc_id")
                                      or (h.payload or {}).get("paper_id"),
                }
                for h in res.reranked_hits[:20]
            ],
        }
        fname = LOG_DIR / f"{ts}_{res.stack}.jsonl"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
