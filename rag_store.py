# rag_pipeline/rag_store.py
from typing import Any, Tuple
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from settings import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION, EMBED_MODEL,
    QDRANT_HOST_B, QDRANT_PORT_B, COLLECTION_B, EMBED_MODEL_B,
    TOP_K_BASE, logger,
)
from triton_client import get_triton_client

_qdr = _emb = _retriever = None
_qdr2 = _emb2 = _retriever2 = None

def build_rag_objects():
    """
    A 스택만 초기화 (기존 단일 스택용).
    듀얼 스택이 필요하면 아래 build_rag_objects_dual() 사용.
    """
    global _qdr, _emb, _retriever
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    # Qdrant Client (A)
    _qdr = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)

    # Embedding Model (A)
    _emb = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True
    )

    # Retriever 설정 (A)
    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=_emb)
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton Client도 미리 준비
    get_triton_client()

    return _qdr, _emb, _retriever

def build_rag_objects_dual():
    """
    A/B 두 개의 RAG 스택을 모두 초기화해서 돌려줌.
    A: (QDRANT_HOST,  COLLECTION,  EMBED_MODEL)
    B: (QDRANT_HOST_B,COLLECTION_B,EMBED_MODEL_B)
    """
    global _qdr, _emb, _retriever, _qdr2, _emb2, _retriever2

    # A 스택 먼저 초기화 (기존 로직 재사용)
    qdr, emb, retriever = build_rag_objects()

    # B 스택이 이미 준비되어 있으면 그대로 반환
    if _qdr2 and _emb2 and _retriever2:
        return qdr, emb, retriever, _qdr2, _emb2, _retriever2

    # Qdrant Client (B)
    _qdr2 = QdrantClient(host=QDRANT_HOST_B, grpc_port=QDRANT_PORT_B, prefer_grpc=True)

    # Embedding Model (B)
    _emb2 = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_B,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True
    )

    # Retriever 설정 (B)
    vstore2 = QdrantVectorStore(client=_qdr2, collection_name=COLLECTION_B)
    sctx2 = StorageContext.from_defaults(vector_store=vstore2)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vstore2, embed_model=_emb2)
    _retriever2 = index2.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton은 공용
    get_triton_client()

    return qdr, emb, retriever, _qdr2, _emb2, _retriever2
