"""Bao AI - In-Memory Vector Store.

Wraps cosine similarity search over whatever an EmbeddingModel produces.
Kept in-memory and process-local (no external vector DB) deliberately —
the knowledge base is small (a curated CSV, plus a handful of uploaded
documents per session), so a dependency like FAISS or Pinecone would be
solving a scale problem this project doesn't have. If the knowledge base
grows past a few thousand rows, this is the file to swap out.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from bao.core.logging import get_logger
from bao.knowledge.embeddings import EmbeddingModel

logger = get_logger(__name__)


@dataclass
class Match:
    index: int
    score: float


class VectorStore:
    """Fits an EmbeddingModel over a corpus once, then answers nearest-match
    queries against it. One VectorStore per corpus (e.g. one for the
    curated facts CSV, a separate one per uploaded-document session).
    """

    def __init__(self, embedding_model: EmbeddingModel):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for VectorStore.")
        self.embedding_model = embedding_model
        self._size = 0

    def build(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Cannot build a VectorStore from an empty corpus.")
        self.embedding_model.fit(corpus)
        self._size = len(corpus)
        logger.info(f"Vector store built with {self._size} entries.")

    def best_match(self, query: str) -> Match | None:
        """Returns the single highest-scoring match, or None if the store
        is empty.
        """
        if self._size == 0:
            return None
        query_vec = self.embedding_model.transform([query])
        similarities = cosine_similarity(query_vec, self.embedding_model.fitted_matrix).flatten()
        best_index = int(np.argmax(similarities))
        return Match(index=best_index, score=float(similarities[best_index]))

    def top_k(self, query: str, k: int = 3) -> list[Match]:
        if self._size == 0:
            return []
        query_vec = self.embedding_model.transform([query])
        similarities = cosine_similarity(query_vec, self.embedding_model.fitted_matrix).flatten()
        ranked_indices = np.argsort(similarities)[::-1][:k]
        return [Match(index=int(i), score=float(similarities[i])) for i in ranked_indices if similarities[i] > 0]

    def __len__(self) -> int:
        return self._size
