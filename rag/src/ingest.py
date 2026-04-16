"""Ingestion module for local RAG.

Responsibilities:
- Discover PDF and TXT files from a data directory.
- Extract text into normalized document objects.
- Split documents into overlapping chunks.
- Persist chunks as JSONL artifacts for downstream embedding.

This module is intentionally self-contained so it can be run independently
as a pipeline stage and reused by other modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class SourceDocument:
    """A normalized source document extracted from disk."""

    document_id: str
    source_path: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A chunk generated from a source document."""

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


class IngestionError(Exception):
    """Raised when ingestion configuration or input is invalid."""


class Ingestor:
    """Handles ingestion and chunking for local document files."""

    def __init__(
        self,
        data_dir: Path,
        chunks_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        recursive: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise IngestionError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise IngestionError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise IngestionError("chunk_overlap must be smaller than chunk_size")

        self.data_dir = data_dir
        self.chunks_dir = chunks_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive = recursive

    def run(self) -> Path:
        """Execute full ingestion flow and return the output JSONL path."""
        source_files = self.discover_files()
        if not source_files:
            raise IngestionError(
                f"No supported files found in {self.data_dir}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        LOGGER.info("Discovered %s source files", len(source_files))

        documents = self.extract_documents(source_files)
        chunks = self.chunk_documents(documents)
        output_path = self.persist_chunks(chunks)

        LOGGER.info("Ingestion completed: %s chunks written to %s", len(chunks), output_path)
        return output_path

    def discover_files(self) -> List[Path]:
        """Discover input files under the configured data directory."""
        if not self.data_dir.exists():
            raise IngestionError(f"Data directory does not exist: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise IngestionError(f"Data path is not a directory: {self.data_dir}")

        pattern = "**/*" if self.recursive else "*"
        files = [
            file_path
            for file_path in self.data_dir.glob(pattern)
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        files.sort()
        return files

    def extract_documents(self, file_paths: Iterable[Path]) -> List[SourceDocument]:
        """Extract plaintext from supported files into source documents."""
        documents: List[SourceDocument] = []

        for file_path in file_paths:
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    text = self._read_pdf(file_path)
                elif suffix == ".txt":
                    text = self._read_txt(file_path)
                else:
                    continue
            except Exception as exc:
                LOGGER.warning("Skipping file due to read error %s: %s", file_path, exc)
                continue

            normalized = self._normalize_text(text)
            if not normalized:
                LOGGER.warning("Skipping empty document: %s", file_path)
                continue

            rel_path = str(file_path.relative_to(self.data_dir))
            document_id = self._stable_id(prefix="doc", text=f"{rel_path}:{len(normalized)}")
            documents.append(
                SourceDocument(
                    document_id=document_id,
                    source_path=rel_path,
                    text=normalized,
                )
            )

        LOGGER.info("Extracted %s documents", len(documents))
        return documents

    def chunk_documents(self, documents: Iterable[SourceDocument]) -> List[TextChunk]:
        """Split each source document into overlapping text chunks."""
        all_chunks: List[TextChunk] = []

        for document in documents:
            doc_chunks = self._chunk_single_document(document)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def persist_chunks(self, chunks: Iterable[TextChunk]) -> Path:
        """Write chunks to JSONL for downstream embedding step."""
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.chunks_dir / "chunks.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")

        return output_path

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        return "\n".join(pages)

    @staticmethod
    def _read_txt(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for legacy text files.
            return file_path.read_text(encoding="latin-1")

    @staticmethod
    def _normalize_text(text: str) -> str:
        # Keep paragraph intent while reducing noisy blank lines/spacing.
        lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        compact = "\n".join(line for line in lines if line)
        return " ".join(compact.split())

    @staticmethod
    def _stable_id(prefix: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def _chunk_single_document(self, document: SourceDocument) -> List[TextChunk]:
        text = document.text
        chunks: List[TextChunk] = []

        start = 0
        index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_seed = f"{document.document_id}:{index}:{start}:{end}:{chunk_text[:80]}"
                chunk_id = self._stable_id(prefix="chunk", text=chunk_seed)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        source_path=document.source_path,
                        chunk_index=index,
                        text=chunk_text,
                        char_start=start,
                        char_end=end,
                    )
                )

            if end >= len(text):
                break

            start += step
            index += 1

        return chunks


def parse_args() -> argparse.Namespace:
    """CLI argument parser for standalone ingestion runs."""
    parser = argparse.ArgumentParser(description="Ingest PDF/TXT files and create chunks JSONL.")
    parser.add_argument("--data-dir", default="./data", help="Path to source documents directory")
    parser.add_argument("--chunks-dir", default="./chunks", help="Path to output chunks directory")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive file discovery in data directory",
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

    ingestor = Ingestor(
        data_dir=Path(args.data_dir),
        chunks_dir=Path(args.chunks_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recursive=not args.no_recursive,
    )
    output_path = ingestor.run()
    print(f"Chunks written to: {output_path}")


if __name__ == "__main__":
    main()
