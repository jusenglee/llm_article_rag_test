# rag_pipeline/rag_store.py
"""
RAG 스택(A/B) 별 공용 리소스(Qdrant, Embedding, Retriever)를
초기화하고 캐싱하는 모듈.

- A 스택:
    - QDRANT_HOST / QDRANT_PORT / COLLECTION
    - EMBED_MODEL  = intfloat/multilingual-e5-large-instruct (권장)
    - query_embedding 시 instruct-style query_instruction 적용

- B 스택:
    - QDRANT_HOST_B / QDRANT_PORT_B / COLLECTION_B
    - EMBED_MODEL_B = intfloat/multilingual-e5-large (권장)
    - E5 규칙(query: / passage:) 기반 임베딩
"""

from typing import Tuple

from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION,
    EMBED_MODEL,
    QDRANT_HOST_B,
    QDRANT_PORT_B,
    COLLECTION_B,
    EMBED_MODEL_B,
    TOP_K_BASE,
    logger,
)
from triton_client import get_triton_client

# 전역 캐시(싱글톤) 객체들
_qdr: QdrantClient | None = None
_emb: HuggingFaceEmbedding | None = None
_retriever = None  # llama_index retriever 타입(호환성 상 구체 타입은 생략)

_qdr2: QdrantClient | None = None
_emb2: HuggingFaceEmbedding | None = None
_retriever2 = None


def build_rag_objects() -> Tuple[QdrantClient, HuggingFaceEmbedding, any]:
    """
    A 스택(A-only)을 초기화하고, (QdrantClient, Embedding, Retriever)를 반환.

    - 기존 단일 스택용 엔트리 포인트.
    - 듀얼 스택(A/B)을 모두 쓰고 싶다면 build_rag_objects_dual() 사용 권장.

    전역 변수(_qdr, _emb, _retriever)에 한 번만 생성해두고
    이후 호출부터는 캐시된 객체를 재사용한다.
    """
    global _qdr, _emb, _retriever

    # 이미 초기화되어 있으면 그대로 반환
    if _qdr and _emb and _retriever:
        return _qdr, _emb, _retriever

    # ------------------------
    # 1) Qdrant Client (A)
    # ------------------------
    _qdr = QdrantClient(
        host=QDRANT_HOST,
        grpc_port=QDRANT_PORT,
        prefer_grpc=True,
    )

    # ------------------------
    # 2) Embedding Model (A)
    # ------------------------
    # 예: EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
    _emb = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True,
        # 쿼리 인스트럭션:
        # - 도메인을 "academic or scholarly documents"로 한정
        # - 실제 쿼리는 "Query: " 뒤에 붙음
        query_instruction=(
            "Instruct: Given a user question about academic or scholarly documents, "
            "retrieve relevant passages that answer the question.\n"
            "Query: "
        ),
        # 문서 임베딩은 원문만 사용하는 설정
        # (instruct 모델 카드에서 doc 쪽 인스트럭션 필요 없음으로 명시되어 있음)
        text_instruction=None,
    )

    # ------------------------
    # 3) Retriever 설정 (A)
    # ------------------------
    # - QdrantVectorStore ↔ QdrantClient 연결
    # - VectorStoreIndex.from_vector_store로 인덱스 래핑
    vstore = QdrantVectorStore(client=_qdr, collection_name=COLLECTION)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vstore,
        embed_model=_emb,
    )
    _retriever = index.as_retriever(similarity_top_k=TOP_K_BASE)

    # ------------------------
    # 4) Triton Client warm-up
    # ------------------------
    # LLM 호출을 위한 Triton gRPC 클라이언트를 미리 준비.
    # (실제 반환값은 사용하지 않고, 내부 싱글톤 초기화 효과만 노림)
    get_triton_client()

    return _qdr, _emb, _retriever


def build_rag_objects_dual() -> Tuple[
    QdrantClient, HuggingFaceEmbedding, any,
    QdrantClient, HuggingFaceEmbedding, any,
]:
    """
    A/B 두 개의 RAG 스택을 모두 초기화해서 한 번에 돌려줌.

    반환:
        (
            qdr_a, emb_a, retriever_a,
            qdr_b, emb_b, retriever_b,
        )

    A 스택:
      - (QDRANT_HOST,  QDRANT_PORT,  COLLECTION,  EMBED_MODEL)
    B 스택:
      - (QDRANT_HOST_B, QDRANT_PORT_B, COLLECTION_B, EMBED_MODEL_B)
    """
    global _qdr, _emb, _retriever, _qdr2, _emb2, _retriever2

    # 1) A 스택 먼저 초기화 (기존 로직 재사용)
    qdr, emb, retriever = build_rag_objects()

    # 2) B 스택이 이미 준비되어 있으면 그대로 반환
    if _qdr2 and _emb2 and _retriever2:
        return qdr, emb, retriever, _qdr2, _emb2, _retriever2

    # ------------------------
    # 3) Qdrant Client (B)
    # ------------------------
    _qdr2 = QdrantClient(
        host=QDRANT_HOST_B,
        grpc_port=QDRANT_PORT_B,
        prefer_grpc=True,
    )

    # ------------------------
    # 4) Embedding Model (B) - classic E5
    # ------------------------
    # 예: EMBED_MODEL_B = "intfloat/multilingual-e5-large"
    _emb2 = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_B,
        device="cuda",
        embed_batch_size=32,
        trust_remote_code=True,
        # E5 추천 규칙:
        #   - 쿼리:  "query: {text}"
        #   - 문서:  "passage: {text}"
        query_instruction="query: ",
        text_instruction="passage: ",
    )

    # ------------------------
    # 5) Retriever 설정 (B)
    # ------------------------
    vstore2 = QdrantVectorStore(client=_qdr2, collection_name=COLLECTION_B)
    index2 = VectorStoreIndex.from_vector_store(
        vector_store=vstore2,
        embed_model=_emb2,
    )
    _retriever2 = index2.as_retriever(similarity_top_k=TOP_K_BASE)

    # Triton은 공용 싱글톤을 그대로 사용 (A/B 스택 모두 동일 LLM 사용 가능)
    get_triton_client()

    return qdr, emb, retriever, _qdr2, _emb2, _retriever2
