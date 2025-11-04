# scripts/01_fetch_peS2o_stream.py
import itertools, orjson
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "peS2o_sample.jsonl"

def main():
    # 🚀 스트리밍 모드
    ds = load_dataset(
        "allenai/peS2o",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    sample_iter = itertools.islice(ds, 1000000)

    with OUT_FILE.open("wb") as f:
        for rec in sample_iter:
            f.write(orjson.dumps(rec) + b"\n")

    print(f"✅ wrote 1000000 records → {OUT_FILE}")

if __name__ == "__main__":
    main()
