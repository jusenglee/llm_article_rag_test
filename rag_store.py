# rag_pipeline/rag_store.py
from typing import Any, Tuple
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from settings import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION, EMBED_MODEL,
    QDRANT_HOST_B, QDRANT_PORT_B, COLLECTION_B, TRITON_EMBED_MODEL_B,
    TOP_K_BASE, logger,
    TRITON_URL_EMBED_B, TRITON_EMBED_MODEL_B,   # <= 새로 정의: B용 Triton 설정
)
from triton_client import get_triton_client  # LLM용
from triton_embedding import TritonEmbedding

_qdr = _emb = _retriever = None
_qdr2 = _emb2 = _retriever2 = None


def build_rag_objects():
    """
    A 스택만 초기화 (기존 단일 스택용).
    A는 로컬 HuggingFaceEmbedding 그대로 사용.
    """
    global _qdr, _emb, _retriever
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    # Qdrant Client (A)
    _qdr = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)

    # Embedding Model (A) - 로컬 (예: BAAI/bge-m3, m-e5 등)
    _emb = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda",         # 윈도우 개발환경이면 여기만 "cpu"로 바꿔도 됨
        embed_batch_size=32,
        trust_remote_code=True,
    )

    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    sctx = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(vector_store=vstore, embed_model=_emb)
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)

    # LLM용 Triton 클라이언트 미리 준비
    get_triton_client()

    return _qdr, _emb, _retriever


def build_rag_objects_dual():
    """
    A/B 두 개의 RAG 스택 초기화.
    - A: 로컬 HF 임베딩 (EMBED_MODEL)
    - B: Triton에 올라간 대형 임베딩 (예: e5-mistral-7b-instruct)
    """
    global _qdr, _emb, _retriever, _qdr2, _emb2, _retriever2

    # A 스택 먼저 초기화
    qdr, emb, retriever = build_rag_objects()

    # B 스택이 이미 준비되어 있으면 그대로 반환
    if _qdr2 and _emb2 and _retriever2:
        return qdr, emb, retriever, _qdr2, _emb2, _retriever2

    # Qdrant Client (B)
    _qdr2 = QdrantClient(host=QDRANT_HOST_B, grpc_port=QDRANT_PORT_B, prefer_grpc=True)

    # Embedding Model (B) - Triton 기반 Mistral 7B 임베딩
    _emb2 = TritonEmbedding(
        triton_url=TRITON_URL_EMBED_B,       # 예: "triton:8001"
        model_name=TRITON_EMBED_MODEL_B,     # 예: "e5_mistral_7b_embed"
        embed_batch_size=32,
        use_e5_instruct=True,
    )

    vstore2 = QdrantVectorStore(client=_qdr2, collection_name=COLLECTION_B)
    sctx2 = StorageContext.from_defaults(vector_store=vstore2)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vstore2, embed_model=_emb2)
    _retriever2 = index2.as_retriever(similarity_top_k=TOP_K_BASE)

    # LLM Triton 클라이언트는 공용
    get_triton_client()

    return qdr, emb, retriever, _qdr2, _emb2, _retriever2
