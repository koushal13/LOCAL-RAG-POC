"""ChromaDB vector store management for local RAG.

Responsibilities:
- Initialize a local persistent ChromaDB client.
- Build/update a collection from embedding artifacts.
- Expose query primitives used by retrieval modules.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.api.models.Collection import Collection

LOGGER = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStoreManager:
    """Encapsulates local ChromaDB operations."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "rag_chunks",
        distance_metric: str = "cosine",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def get_or_create_collection(self) -> Collection:
        """Return existing collection or create one with configured metric."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def reset_collection(self) -> None:
        """Delete and recreate collection to ensure a clean index build."""
        existing_names = self._list_collection_names()
        if self.collection_name in existing_names:
            self.client.delete_collection(self.collection_name)
        self.get_or_create_collection()

    def _list_collection_names(self) -> set[str]:
        """Return collection names across Chroma versions.

        Chroma < 0.6 returns collection objects from list_collections(),
        while Chroma >= 0.6 returns list[str].
        """
        collections = self.client.list_collections()
        if not collections:
            return set()

        first = collections[0]
        if isinstance(first, str):
            return set(collections)

        return {collection.name for collection in collections}

    def index_embeddings(self, embeddings_file: Path, batch_size: int = 128) -> int:
        """Upsert embedding records from JSONL into ChromaDB.

        Returns the number of indexed chunks.
        """
        if not embeddings_file.exists():
            raise VectorStoreError(f"Embeddings file does not exist: {embeddings_file}")
        if batch_size <= 0:
            raise VectorStoreError("batch_size must be > 0")

        collection = self.get_or_create_collection()
        records = self._load_embeddings(embeddings_file)
        if not records:
            raise VectorStoreError(f"No embeddings found in {embeddings_file}")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        indexed = 0
        for record in records:
            ids.append(record["chunk_id"])
            documents.append(record["text"])
            embeddings.append(record["embedding"])
            metadatas.append(
                {
                    "document_id": record["document_id"],
                    "source_path": record["source_path"],
                    "chunk_index": int(record["chunk_index"]),
                    "char_start": int(record["char_start"]),
                    "char_end": int(record["char_end"]),
                }
            )

            if len(ids) >= batch_size:
                collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
                indexed += len(ids)
                ids, documents, metadatas, embeddings = [], [], [], []

        if ids:
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            indexed += len(ids)

        return indexed

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        include: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Query the collection by vector and return raw Chroma response."""
        if top_k <= 0:
            raise VectorStoreError("top_k must be > 0")

        collection = self.get_or_create_collection()
        query_include = include or ["documents", "metadatas", "distances"]
        return collection.query(query_embeddings=[query_embedding], n_results=top_k, include=query_include)

    def count(self) -> int:
        """Return current collection size."""
        return self.get_or_create_collection().count()

    @staticmethod
    def _load_embeddings(embeddings_file: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with embeddings_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def parse_args() -> argparse.Namespace:
    """CLI parser for standalone vector store indexing."""
    parser = argparse.ArgumentParser(description="Index embeddings into local ChromaDB.")
    parser.add_argument("--embeddings-file", default="./embeddings/embeddings.jsonl", help="Embeddings JSONL path")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Chroma persistent directory")
    parser.add_argument("--collection-name", default="rag_chunks", help="Chroma collection name")
    parser.add_argument("--batch-size", type=int, default=128, help="Upsert batch size")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before indexing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    manager = VectorStoreManager(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection_name,
    )

    if args.reset:
        LOGGER.info("Resetting collection %s", args.collection_name)
        manager.reset_collection()

    indexed_count = manager.index_embeddings(
        embeddings_file=Path(args.embeddings_file),
        batch_size=args.batch_size,
    )
    print(f"Indexed chunks: {indexed_count}")
    print(f"Collection size: {manager.count()}")


if __name__ == "__main__":
    main()
