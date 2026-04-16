"""Semantic retrieval module for local RAG.

Responsibilities:
- Embed user queries with the same local embedding model.
- Query ChromaDB for nearest chunks.
- Return normalized retrieval results for downstream prompt construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from vector_store import VectorStoreManager


@dataclass(frozen=True)
class RetrievedChunk:
    """Normalized retrieval result."""

    chunk_id: str
    text: str
    distance: float
    score: float
    metadata: Dict[str, Any]


class Retriever:
    """Embeds queries and retrieves top-k chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Return top-k semantically relevant chunks for a query."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        available_chunks = self.vector_store.count()
        if available_chunks <= 0:
            return []
        effective_top_k = min(top_k, available_chunks)

        query_vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        raw = self.vector_store.query(query_embedding=query_vector, top_k=effective_top_k)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # For cosine distance in [0, 2], lower is better. Convert to bounded score.
            score = max(0.0, min(1.0, 1.0 - (float(distance) / 2.0)))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    distance=float(distance),
                    score=score,
                    metadata=metadata or {},
                )
            )

        return results
