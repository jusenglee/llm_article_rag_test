import os
import uuid
from pathlib import Path
import orjson
from typing import Iterator, Dict, Any, List
from tqdm import tqdm
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection, utility
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# ---------------------------
# 설정
# ---------------------------
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "peS2o_rag")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

JSON_PATH = Path("../data/peS2o_sample.jsonl")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120
BATCH_SIZE = 256
RESUME_STATE = Path(".ingest_resume_peS2o.txt")

# ---------------------------
# 유틸
# ---------------------------
def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)

def load_resume_offset() -> int:
    if RESUME_STATE.exists():
        try:
            return int(RESUME_STATE.read_text().strip())
        except Exception:
            return 0
    return 0

def save_resume_offset(n: int) -> None:
    RESUME_STATE.write_text(str(n))

def to_chunks(rec: Dict[str, Any], splitter: SentenceSplitter) -> List[Document]:
    # 1) 최우선: text 필드
    text = (rec.get("text") or "").strip()

    # 2) 대안 경로: title/abstract/sections/body_text에서 합성
    if not text:
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or rec.get("paperAbstract") or "").strip()

        # sections: [{heading, text}] 또는 [{section/section_title, text}] 가정
        sections_txt = []
        secs = rec.get("sections") or rec.get("body_text") or rec.get("pdf_parse", {}).get("body_text") or []
        if isinstance(secs, list):
            for s in secs[:50]:  # 과도한 본문 방지
                if isinstance(s, dict):
                    st = (s.get("text") or "").strip()
                    if st:
                        sections_txt.append(st)

        # body 후보 (일부 데이터셋은 'body'나 'content' 등으로 있을 수 있음)
        body = (rec.get("body") or rec.get("content") or "").strip()

        parts = [title, abstract] + sections_txt + ([body] if body else [])
        text = "\n\n".join([p for p in parts if p])

    if not text:
        return []  # 여전히 비면 스킵

    paper_id = rec.get("id") or rec.get("paper_id") or rec.get("uid") or ""
    source = rec.get("source", "peS2o")

    docs: List[Document] = []
    for chunk in splitter.split_text(text):
        docs.append(Document(text=chunk, metadata={"paper_id": str(paper_id), "source": source}))
    return docs


# ---------------------------
# 메인
# ---------------------------
def main():
    assert JSON_PATH.exists(), f"입력 파일 없음: {JSON_PATH}"

    # Milvus 연결
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # 컬렉션 존재 확인 / 생성
    if not utility.has_collection(COLLECTION):
        print(f"🆕 Creating new collection: {COLLECTION}")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]
        schema = CollectionSchema(fields, description="RAG embeddings for peS2o papers")
        collection = Collection(name=COLLECTION, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        print("✅ Index created.")
    else:
        print(f"✅ Found collection: {COLLECTION}")
        collection = Collection(name=COLLECTION)

    collection.load()

    # 임베딩 모델
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cuda")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 재시작 오프셋
    start_idx = load_resume_offset()
    print(f"↩️  Resume from line index: {start_idx}")

    batch_ids, batch_paper_ids, batch_sources, batch_texts, batch_embeddings = [], [], [], [], []
    total_chunks = 0

    try:
        total_lines = sum(1 for _ in JSON_PATH.open("r", encoding="utf-8"))
    except Exception:
        total_lines = None

    with tqdm(total=total_lines, desc="Ingesting", unit="line", disable=(total_lines is None)) as pbar:
        for i, rec in enumerate(iter_jsonl(JSON_PATH)):
            if i < start_idx:
                if total_lines:
                    pbar.update(1)
                continue

            print(f"[DEBUG] line={i}  keys={list(rec.keys())[:10]}")  # ① JSON 구조
            docs = to_chunks(rec, splitter)
            print(f"[DEBUG] chunks={len(docs)}")                     # ② 청크 수

            for doc in docs:
                emb = embed_model.get_text_embedding(doc.text)
                uid = f"{rec.get('id') or rec.get('paper_id') or ''}_{uuid.uuid4().hex}"
                batch_ids.append(uid)
                batch_paper_ids.append(str(rec.get("id") or rec.get("paper_id") or ""))
                batch_sources.append(rec.get("source", "peS2o"))
                batch_texts.append(doc.text)
                batch_embeddings.append(emb)

            if len(batch_ids) >= BATCH_SIZE:
                print(f"[DEBUG] insert batch: {len(batch_ids)}")
                data = [
                    batch_ids,
                    batch_paper_ids,
                    batch_sources,
                    batch_texts,
                    batch_embeddings,
                ]
                collection.insert(data)
                total_chunks += len(batch_ids)
                batch_ids, batch_paper_ids, batch_sources, batch_texts, batch_embeddings = [], [], [], [], []
                save_resume_offset(i + 1)

            if total_lines:
                pbar.update(1)

        # 잔여 처리
        if batch_ids:
            data = [
                batch_ids,
                batch_paper_ids,
                batch_sources,
                batch_texts,
                batch_embeddings,
            ]
            collection.insert(data)
            total_chunks += len(batch_ids)
            save_resume_offset(i + 1)

    collection.flush()
    print(f"✅ Done. Indexed chunks: {total_chunks}")
    print(f"🔎 Collection: {COLLECTION} @ {MILVUS_HOST}:{MILVUS_PORT}")

if __name__ == "__main__":
    main()
