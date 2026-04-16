"""Embedding generation module for local RAG.

Responsibilities:
- Load chunk artifacts created by ingest.py.
- Generate local embeddings with sentence-transformers/all-MiniLM-L6-v2.
- Persist embedding records as JSONL for indexing in ChromaDB.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    """Input chunk schema loaded from chunks JSONL."""

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class EmbeddedChunk:
    """Chunk with generated embedding vector."""

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    embedding: List[float]


class EmbeddingError(Exception):
    """Raised when embedding prerequisites or inputs are invalid."""


class Embedder:
    """Generates embeddings from chunk artifacts."""

    def __init__(
        self,
        chunks_file: Path,
        embeddings_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be > 0")

        self.chunks_file = chunks_file
        self.embeddings_dir = embeddings_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def run(self) -> Path:
        """Execute embedding generation and return output JSONL path."""
        chunk_records = self.load_chunks(self.chunks_file)
        if not chunk_records:
            raise EmbeddingError(f"No chunks found in {self.chunks_file}")

        model = SentenceTransformer(self.model_name)
        texts = [record.text for record in chunk_records]

        LOGGER.info(
            "Generating embeddings for %s chunks using model %s",
            len(texts),
            self.model_name,
        )
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        embedded_records: List[EmbeddedChunk] = []
        for record, vector in zip(chunk_records, vectors):
            embedded_records.append(
                EmbeddedChunk(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    source_path=record.source_path,
                    chunk_index=record.chunk_index,
                    text=record.text,
                    char_start=record.char_start,
                    char_end=record.char_end,
                    embedding=vector.tolist(),
                )
            )

        output_path = self.persist_embeddings(embedded_records)
        LOGGER.info("Embedding generation completed: %s", output_path)
        return output_path

    @staticmethod
    def load_chunks(chunks_file: Path) -> List[ChunkRecord]:
        """Load chunk records from JSONL produced by ingest.py."""
        if not chunks_file.exists():
            raise EmbeddingError(f"Chunks file does not exist: {chunks_file}")

        records: List[ChunkRecord] = []
        with chunks_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(ChunkRecord(**payload))

        return records

    def persist_embeddings(self, embedded_chunks: Iterable[EmbeddedChunk]) -> Path:
        """Persist embedded chunks as JSONL for indexing stage."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.embeddings_dir / "embeddings.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for record in embedded_chunks:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

        return output_path


def parse_args() -> argparse.Namespace:
    """CLI parser for standalone embedding generation."""
    parser = argparse.ArgumentParser(description="Generate embeddings for chunk artifacts.")
    parser.add_argument("--chunks-file", default="./chunks/chunks.jsonl", help="Path to chunks JSONL")
    parser.add_argument("--embeddings-dir", default="./embeddings", help="Output embeddings directory")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding vector normalization",
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

    embedder = Embedder(
        chunks_file=Path(args.chunks_file),
        embeddings_dir=Path(args.embeddings_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
    )
    output_path = embedder.run()
    print(f"Embeddings written to: {output_path}")


if __name__ == "__main__":
    main()
