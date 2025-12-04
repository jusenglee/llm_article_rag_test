import os, json, math, time, threading, uuid, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# 임베딩/토크나이저
import torch
from sentence_transformers import SentenceTransformer

# 문장 분할기 (간결/빠름)
try:
    from llama_index.core.node_parser import SentenceSplitter
    _HAS_LLAMA_SPLITTER = True
except Exception:
    _HAS_LLAMA_SPLITTER = False

# =========================
# 설정
# =========================
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
USE_GRPC = os.getenv("QDRANT_USE_GRPC", "0") == "1"   # gRPC가 열려있다면 1 권장(네트워크 레이턴시↓)

COLLECTION  = os.getenv("QDRANT_COLLECTION", "mistral_rag_100_000")

EMBED_MODEL = os.getenv("EMBED_MODEL", "../models/e5-mistral")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "512"))

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "120"))
JSON_PATH      = Path(os.getenv("JSON_PATH", "peS2o_sample.jsonl"))
RESUME_FILE    = Path(os.getenv("RESUME_FILE", ".ingest_resume_dualgpu.state"))

# Qdrant
DISTANCE_ENUM  = Distance.COSINE
ALLOW_RECREATE = os.getenv("ALLOW_RECREATE", "0") == "1"

# 업서트 병렬 스레드 수
UPSERT_WORKERS = int(os.getenv("UPSERT_WORKERS", "2"))
# 한 번에 업서트하는 포인트 수
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", str(EMBED_BATCH)))

# tqdm 업데이트 빈도 줄이기
TQDM_MIN_INTERVAL = float(os.getenv("TQDM_MIN_INTERVAL", "0.3"))

# 무작위 샘플링 설정: JSONL 라인 중에서 이 개수만 랜덤 선택
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100000"))  #  10만건
SAMPLE_SEED = int(os.getenv("SAMPLE_SEED", "42"))

# =========================
# 유틸
# =========================
def make_point_id(paper_id: str, chunk_idx: int, text: str) -> str:
    base = f"{paper_id}_{chunk_idx}_{text[:50]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def iter_jsonl_fast(path: Path, start_idx: int = 0):
    """가벼운 JSONL 이터레이터 (표준 json: C 확장 가속 + 오브젝트 풀 최소화)"""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            s = line.strip()
            if not s:
                continue
            yield i, json.loads(s)

def save_resume(n: int):
    RESUME_FILE.write_text(str(n), encoding="utf-8")

def load_resume() -> int:
    if RESUME_FILE.exists():
        try:
            return int(RESUME_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            return 0
    return 0

# =========================
# 임베더 (1~2 GPU 최적화)
# =========================
class DualGPUEmbedder:
    """
    - SentenceTransformer 직접 사용(오버헤드 최소화)
    - normalize_embeddings=True → 코사인 길이 정규화 자동
    - 2개의 GPU가 있으면 텍스트 배치를 반으로 쪼개 두 모델로 동시 encode
    """
    def __init__(self, model_name: str, batch_size: int = 256, torch_dtype: Optional[torch.dtype] = None):
        self.batch_size = batch_size

        n_gpu = torch.cuda.device_count()
        self.devices = []
        if n_gpu >= 1:
            self.devices.append("cuda:0")
        if n_gpu >= 2:
            self.devices.append("cuda:1")

        # dtype 최적화(선택): fp16 권장 (Ampere+)
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if not self.devices:
            # CPU fallback
            self.devices = ["cpu"]

        self.models: List[SentenceTransformer] = []
        for dev in self.devices:
            m = SentenceTransformer(model_name, device=dev)
            self.models.append(m)

        # 임베딩 차원 자동 감지
        self.embedding_dim = self.models[0].get_sentence_embedding_dimension()

    def encode_single(self, model: SentenceTransformer, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            embs = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # 코사인 정규화 일괄 처리
                show_progress_bar=False,
            )
        return embs

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if len(self.models) == 1 or len(texts) == 0:
            embs = self.encode_single(self.models[0], texts)
            return embs.tolist()

        # 2개 이상 모델일 때: 반으로 쪼개 동시 처리 (현재 2GPU까지 사용)
        mid = len(texts) // 2
        parts = [texts[:mid], texts[mid:]]

        res: List[Optional[np.ndarray]] = [None] * len(self.models)

        def run(idx: int, subset: List[str]):
            if subset:
                res[idx] = self.encode_single(self.models[idx], subset)
            else:
                res[idx] = np.empty((0, self.embedding_dim), dtype=np.float32)

        threads = []
        for i in range(len(self.models)):
            t = threading.Thread(target=run, args=(i, parts[i] if i < 2 else []))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        out = np.concatenate([r for r in res if r is not None], axis=0)
        return out.tolist()

# =========================
# Qdrant
# =========================
def ensure_collection(client: QdrantClient, collection: str, vector_dim: int):
    exists = False
    try:
        info = client.get_collection(collection)
        exists = True
        params = info.config.params
        cur_size = params.vectors.size
        cur_dist = str(params.vectors.distance).lower()
        if cur_size != vector_dim or "cosine" not in cur_dist:
            msg = (f"⚠️ 컬렉션 정의 불일치: size={cur_size}, distance={params.vectors.distance} "
                   f"(필요 size={vector_dim}, distance=Cosine)")
            print(msg)
            if not ALLOW_RECREATE:
                raise RuntimeError(msg + "  (ALLOW_RECREATE=1 환경변수로 재생성 허용)")
            print("♻️  재생성 진행 (데이터 삭제됨) ...")
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_dim, distance=DISTANCE_ENUM),
            )
            if RESUME_FILE.exists():
                RESUME_FILE.unlink(missing_ok=True)
            print(f"✅ Collection recreated: {collection} (Cosine/{vector_dim})")
        else:
            print(f"✅ Found collection: {collection} (Cosine/{vector_dim})")
    except Exception:
        if not exists:
            print(f"ℹ️ 컬렉션 미존재 → 생성: {collection}")
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_dim, distance=DISTANCE_ENUM),
            )
            print(f"✅ Created collection: {collection} (Cosine/{vector_dim})")
        else:
            raise

class AsyncUpserter:
    """
    업서트를 비동기 병렬로 처리하여
    - GPU가 다음 배치를 encode 하는 동안
    - 네트워크 업서트는 동시에 진행
    """
    def __init__(self, client: QdrantClient, collection: str, max_workers: int = 4):
        self.client = client
        self.collection = collection
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: List[Future] = []

    def submit(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        fut = self.executor.submit(self.client.upsert, collection_name=self.collection, points=points)
        self.futures.append(fut)

    def drain(self):
        # 모든 업서트 완료 대기 (예외 전파)
        for fut in self.futures:
            fut.result()
        self.futures.clear()

    def shutdown(self):
        self.drain()
        self.executor.shutdown(wait=True)

# =========================
# 메인
# =========================
def main():
    # 1) 모델 로드(차원 자동 감지)
    #    - fp16 사용을 원하면 torch_dtype=torch.float16 전달(환경과 VRAM 여유에 따라)
    dtype = torch.float16 if os.getenv("EMBED_DTYPE_FP16", "1") == "1" and torch.cuda.is_available() else None
    embedder = DualGPUEmbedder(EMBED_MODEL, batch_size=EMBED_BATCH, torch_dtype=dtype)
    VECTOR_DIM = embedder.embedding_dim
    print(f"🧠 Loaded model: {EMBED_MODEL} | dim={VECTOR_DIM} | devices={embedder.devices}")

    # 2) Qdrant 연결
    if USE_GRPC:
        client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
    else:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT)
    ensure_collection(client, COLLECTION, VECTOR_DIM)

    # 3) JSONL 총 라인 수 계산 (샘플링 및 진행률을 위해)
    try:
        total_lines = sum(1 for _ in JSON_PATH.open("r", encoding="utf-8"))
    except Exception:
        total_lines = None

    # 4) 무작위 샘플링 인덱스 결정
    selected_idx_set: Optional[set[int]] = None
    effective_sample_size = None

    if total_lines is not None and SAMPLE_SIZE > 0:
        if SAMPLE_SIZE >= total_lines:
            # 전체 라인 수가 10만보다 적으면 그냥 전부 사용
            print(f"ℹ️ total_lines={total_lines} <= SAMPLE_SIZE={SAMPLE_SIZE} → 전체 라인 사용")
            effective_sample_size = total_lines
        else:
            rng = random.Random(SAMPLE_SEED)
            sampled_indices = rng.sample(range(total_lines), SAMPLE_SIZE)
            sampled_indices.sort()
            selected_idx_set = set(sampled_indices)
            effective_sample_size = SAMPLE_SIZE
            print(f"🎯 무작위 샘플링: total_lines={total_lines}, sample_size={SAMPLE_SIZE}, seed={SAMPLE_SEED}")
    else:
        # total_lines를 가져오지 못했거나 SAMPLE_SIZE=0 이면 전체 사용
        effective_sample_size = total_lines
        if SAMPLE_SIZE > 0 and total_lines is None:
            print("⚠️ total_lines 계산 실패 → 랜덤 샘플링 불가, 전체 라인 sequential ingest")

    # 5) 재개 위치
    #    랜덤 샘플링 모드에서는 resume를 사용하지 않음 (샘플 구성이 매번 달라짐)
    if selected_idx_set is not None:
        start_line = 0
        print("ℹ️ 랜덤 샘플링 모드 → RESUME_FILE 무시 (항상 처음부터 스캔)")
    else:
        start_line = load_resume()
    print(f"↩️ Resume from line {start_line}")

    # 6) 분리기 (청킹 품질 강화)
    if _HAS_LLAMA_SPLITTER:
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        def split_func(txt: str) -> List[str]:
            return splitter.split_text(txt)
    else:
        # 간단한 폴백: 단락 우선 → 길면 고정길이 슬라이딩 윈도우 (overlap 적용)
        def split_func(txt: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
            out: List[str] = []
            # overlap이 너무 크면 살짝 줄인다 (무한루프 방지)
            if overlap >= size:
                overlap = size // 3

            # 두 줄 공백 기준으로 rough paragraph split
            paragraphs = [p.strip() for p in txt.split("\n\n") if p.strip()]
            for para in paragraphs:
                n = len(para)
                if n <= size:
                    out.append(para)
                    continue
                i = 0
                while i < n:
                    j = min(n, i + size)
                    out.append(para[i:j])
                    if j == n:
                        break
                    i = j - overlap
            return out

    # 7) 업서트 비동기 실행기
    upserter = AsyncUpserter(client, COLLECTION, max_workers=UPSERT_WORKERS)

    total_chunks = 0
    docs_ingested = 0  # JSON 레코드 기준
    ids: List[str] = []
    payloads: List[Dict[str, Any]] = []
    texts: List[str] = []

    # 임베딩 시간 측정용
    embed_time_total = 0.0
    ingest_start_ts = time.time()

    # tqdm 진행률 바: 랜덤 샘플링이면 SAMPLE_SIZE 기준, 아니면 total_lines 기준
    pbar_total = effective_sample_size
    pbar = tqdm(total=pbar_total, desc="Ingesting", ncols=100, mininterval=TQDM_MIN_INTERVAL)

    try:
        for i, rec in iter_jsonl_fast(JSON_PATH, start_idx=start_line):
            # 랜덤 샘플링인 경우: 선택된 인덱스만 사용
            if selected_idx_set is not None:
                if i not in selected_idx_set:
                    continue

            raw_text = (rec.get("text") or rec.get("_node_text") or "").strip()
            if not raw_text:
                # 빈 텍스트는 문서 카운트에 포함하지 않고 스킵
                continue

            paper_id = str(rec.get("paper_id") or rec.get("id") or rec.get("doc_id") or "unknown")
            source   = rec.get("source", "peS2o")

            chunks = split_func(raw_text)
            if not chunks:
                continue

            for ci, chunk in enumerate(chunks):
                pid = make_point_id(paper_id, ci, chunk)
                ids.append(pid)
                payloads.append({
                    "paper_id": paper_id,
                    "source": source,
                    "_node_text": chunk,
                    "line_idx": i,
                    "chunk_idx": ci,
                })
                texts.append(chunk)

                # 업서트 단위는 임베딩 배치와 동일하게 맞춤
                if len(texts) >= UPSERT_BATCH_SIZE:
                    t0 = time.time()
                    vecs = embedder.embed_batch(texts)
                    embed_time_total += (time.time() - t0)

                    upserter.submit(ids, vecs, payloads)
                    total_chunks += len(texts)
                    ids.clear(); payloads.clear(); texts.clear()

                    # 랜덤 샘플링 모드가 아닌 경우에만 resume 저장
                    if selected_idx_set is None:
                        save_resume(i + 1)

            docs_ingested += 1
            if pbar_total is not None:
                pbar.update(1)

        # 잔여 처리
        if texts:
            t0 = time.time()
            vecs = embedder.embed_batch(texts)
            embed_time_total += (time.time() - t0)

            upserter.submit(ids, vecs, payloads)
            total_chunks += len(texts)

        # 모든 업서트 완료 대기
        upserter.drain()

        ingest_end_ts = time.time()
        wall_time = ingest_end_ts - ingest_start_ts

        # ===== 임베딩 속도 로깅 =====
        print("\n====== Embedding / Ingest Stats ======")
        print(f"📚 Docs ingested     : {docs_ingested}")
        print(f"🔹 Chunks ingested   : {total_chunks}")
        print(f"⏱ Total wall time   : {wall_time:.2f} s (IO + upsert + embed 포함)")

        if embed_time_total > 0:
            chunks_per_sec = total_chunks / embed_time_total
            docs_per_sec = docs_ingested / embed_time_total if docs_ingested > 0 else 0.0
            print(f"🧪 Embedding time    : {embed_time_total:.2f} s (순수 embed_batch 누적)")
            print(f"⚡ Embedding speed   : {chunks_per_sec:.1f} chunks/s, {docs_per_sec:.1f} docs/s")
        else:
            print("⚠️ Embedding time 측정값이 0입니다. (임베딩 호출이 없었거나 오류)")

        print("======================================\n")
        print(f"✅ Done. Total {docs_ingested} docs, {total_chunks} chunks ingested into '{COLLECTION}'.")

    finally:
        upserter.shutdown()
        pbar.close()

if __name__ == "__main__":
    main()
