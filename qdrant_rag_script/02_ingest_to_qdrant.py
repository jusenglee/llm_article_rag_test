import os, json, math, hashlib, time, threading, uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import orjson
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# ---------------------------
# 설정
# ---------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # ← 로컬 테스트 기본값
QDRANT_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_URL  = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION  = os.getenv("QDRANT_COLLECTION", "peS2o_rag")

EMBED_MODEL    = os.getenv("EMBED_MODEL", "BAAI/bge-m3")  # 1024-d dense
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "120"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "512"))
JSON_PATH      = Path(os.getenv("JSON_PATH", "peS2o_sample.jsonl"))
RESUME_FILE    = Path(os.getenv("RESUME_FILE", ".ingest_resume_dualgpu.state"))
VECTOR_DIM     = int(os.getenv("VECTOR_DIM", "1024"))
DISTANCE_ENUM  = Distance.COSINE          # ← 문자열 아님! Enum 사용
USE_CACHE      = os.getenv("USE_CACHE", "1") == "1"

# 컬렉션이 이미 있고 데이터가 있는데 재생성 방지
ALLOW_RECREATE = os.getenv("ALLOW_RECREATE", "0") == "1"

# ---------------------------
# 유틸
# ---------------------------
def make_point_id(paper_id: str, chunk_idx: int, text: str) -> str:
    base = f"{paper_id}_{chunk_idx}_{text[:50]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def iter_jsonl(path: Path, start_idx: int = 0):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_idx: continue
            s = line.strip()
            if not s: continue
            yield i, orjson.loads(s)

def save_resume(n: int):
    RESUME_FILE.write_text(str(n))

def load_resume() -> int:
    if RESUME_FILE.exists():
        try:
            return int(RESUME_FILE.read_text().strip())
        except:
            return 0
    return 0

# ---------------------------
# Qdrant
# ---------------------------
def ensure_collection(client: QdrantClient):
    exists = False
    try:
        info = client.get_collection(COLLECTION)
        exists = True
        params = info.config.params
        # distance/size 불일치 시 경고
        if params.vectors.size != VECTOR_DIM or str(params.vectors.distance).lower().find("cosine") == -1:
            msg = (f"⚠️ 컬렉션 정의 불일치: size={params.vectors.size}, "
                   f"distance={params.vectors.distance} (필요 size={VECTOR_DIM}, distance=Cosine)")
            print(msg)
            if not ALLOW_RECREATE:
                raise RuntimeError(msg + "  (ALLOW_RECREATE=1 환경변수로 재생성 허용 가능)")
            print("♻️  재생성 진행 (데이터 삭제됨) ...")
            client.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE_ENUM),
            )
            # 재시작이므로 resume 파일도 초기화 권장
            if RESUME_FILE.exists():
                RESUME_FILE.unlink(missing_ok=True)
            print(f"✅ Collection recreated: {COLLECTION} (Cosine/{VECTOR_DIM})")
        else:
            print(f"✅ Found collection: {COLLECTION} (Cosine/{VECTOR_DIM})")
    except Exception as e:
        if not exists:
            print(f"ℹ️ 컬렉션 미존재 → 생성: {COLLECTION}")
            client.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE_ENUM),
            )
            print(f"✅ Created collection: {COLLECTION} (Cosine/{VECTOR_DIM})")
        else:
            raise

# ---------------------------
# Dual GPU Embedding (정규화 포함)
# ---------------------------
class DualGPUEmbedder:
    def __init__(self, model_name: str, batch_size: int = 256):
        # 단일 GPU 환경에서도 동작하도록 가드
        self.models = []
        try:
            self.models.append(HuggingFaceEmbedding(model_name=model_name, device="cuda:0", embed_batch_size=batch_size))
            # 두 번째 GPU가 없으면 except로 넘어감
            self.models.append(HuggingFaceEmbedding(model_name=model_name, device="cuda:1", embed_batch_size=batch_size))
        except Exception:
            # fallback: 단일 GPU 또는 CPU
            if not self.models:
                self.models.append(HuggingFaceEmbedding(model_name=model_name, device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu", embed_batch_size=batch_size))

        self.cache = {} if USE_CACHE else None

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if len(self.models) == 1:
            vecs = self.models[0].get_text_embedding_batch(texts)
        else:
            mid = len(texts) // 2
            parts = [texts[:mid], texts[mid:]]
            res = [None, None]

            def run(idx: int):
                if parts[idx]:
                    res[idx] = self.models[idx].get_text_embedding_batch(parts[idx])

            t1 = threading.Thread(target=run, args=(0,))
            t2 = threading.Thread(target=run, args=(1,))
            t1.start(); t2.start()
            t1.join(); t2.join()

            vecs = []
            for r in res:
                if r: vecs.extend(r)

        # 🔒 Cosine 일관성 확보를 위해 항상 단위벡터화
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr.tolist()

# ---------------------------
# 업서트
# ---------------------------
def upsert_batch(client: QdrantClient, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
    points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
    client.upsert(collection_name=COLLECTION, points=points)

# ---------------------------
# 메인
# ---------------------------
def main():
    # HTTP 모드(안정)로 연결
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    # 재생성했으면 resume 초기화 권장
    start_line = load_resume()
    print(f"↩️ Resume from line {start_line}")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embedder = DualGPUEmbedder(EMBED_MODEL, batch_size=256)

    total_chunks = 0
    ids, payloads, texts = [], [], []

    total_lines = sum(1 for _ in open(JSON_PATH, "r", encoding="utf-8"))
    pbar = tqdm(total=total_lines, desc="Ingesting", ncols=100)

    for i, rec in iter_jsonl(JSON_PATH, start_line):
        # 다양한 스키마 대응
        raw_text = (rec.get("text") or rec.get("_node_text") or "").strip()
        if not raw_text:
            pbar.update(1)
            continue

        paper_id = str(rec.get("paper_id") or rec.get("id") or rec.get("doc_id") or "unknown")
        source   = rec.get("source", "peS2o")

        chunks = splitter.split_text(raw_text)
        for ci, chunk in enumerate(chunks):
            pid = make_point_id(paper_id, ci, chunk)
            ids.append(pid)
            payloads.append({"paper_id": paper_id, "source": source, "_node_text": chunk})
            texts.append(chunk)

        # 배치 임계 시 처리
        if len(texts) >= BATCH_SIZE:
            vecs = embedder.embed_batch(texts)
            upsert_batch(client, ids, vecs, payloads)
            total_chunks += len(texts)
            ids.clear(); payloads.clear(); texts.clear()
            save_resume(i + 1)
        pbar.update(1)

    # 잔여 처리
    if texts:
        vecs = embedder.embed_batch(texts)
        upsert_batch(client, ids, vecs, payloads)
        total_chunks += len(texts)
        save_resume(i + 1)

    print(f"✅ Done. Total {total_chunks} chunks ingested.")

if __name__ == "__main__":
    main()
