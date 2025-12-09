from typing import List
import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput


class TritonEmbedding(BaseEmbedding):
    """Triton에 올려둔 임베딩 모델(e5-mistral-7b)을 llama_index용으로 래핑."""

    def __init__(
            self,
            triton_url: str,
            model_name: str,
            embed_batch_size: int = 8,
            timeout_s: float = 60.0,
            use_e5_instruct: bool = True,
            task_description: str | None = None,
    ) -> None:
        super().__init__()
        self._client = InferenceServerClient(triton_url, verbose=False)
        self._model_name = model_name
        self._batch_size = embed_batch_size
        self._timeout_s = timeout_s

        # e5-mistral 전용 옵션
        self._use_e5_instruct = use_e5_instruct
        self._task_description = (
                task_description
                or "Given a web search query, retrieve relevant passages that answer the query"
        )

    # ------------------------
    # e5 쿼리용 프롬프트 빌더
    # ------------------------
    def _build_query_text(self, query: str) -> str:
        if not self._use_e5_instruct:
            return query
        return f"Instruct: {self._task_description}\nQuery: {query}"

    # ------------------------
    # 내부 공용 배치 호출
    # ------------------------
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        result: List[List[float]] = []

        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]

            np_texts = np.array(
                [[t.encode("utf-8")] for t in chunk],  # shape: (B, 1)
                dtype=object,
            )

            infer_input = InferInput("text_input", np_texts.shape, "BYTES")
            infer_input.set_data_from_numpy(np_texts)

            infer_output = InferRequestedOutput("EMBEDDING")

            resp = self._client.infer(
                model_name=self._model_name,
                inputs=[infer_input],
                outputs=[infer_output],
                client_timeout=self._timeout_s,
            )
            emb = resp.as_numpy("EMBEDDING")  # (B, D)
            result.extend(emb.tolist())

        return result

    # ------------------------
    # sync API (BaseEmbedding)
    # ------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        q = self._build_query_text(query)
        return self._embed_batch([q])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        # 문서는 지시어 없이 원문 그대로
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    # ------------------------
    # async API (BaseEmbedding)
    # ------------------------
    async def _aget_query_embedding(self, query: str) -> List[float]:
        q = self._build_query_text(query)
        return self._embed_batch([q])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
