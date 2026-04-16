"""CLI entrypoint for the complete local RAG workflow.

Commands:
- ingest: Parse PDF/TXT and generate chunks JSONL.
- embed: Generate local embeddings from chunks.
- index: Build/refresh ChromaDB collection from embeddings.
- rebuild: Run ingest + embed + index in sequence.
- ask: Run retrieval-augmented generation via local Ollama.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from embed import Embedder
from ingest import Ingestor
from rag_pipeline import OllamaClient, RAGPipeline
from retrieve import Retriever
from vector_store import VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-ready local RAG CLI")

    parser.add_argument("--data-dir", default="./data", help="Input documents directory")
    parser.add_argument("--chunks-dir", default="./chunks", help="Chunk artifacts directory")
    parser.add_argument("--embeddings-dir", default="./embeddings", help="Embedding artifacts directory")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="Chroma persistence directory")
    parser.add_argument("--collection-name", default="rag_chunks", help="Chroma collection name")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local embedding model name",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:1b",
        help="Local Ollama model name (e.g., llama3.2:1b, llama3:8b, mistral, phi3:mini)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for local Ollama server",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest files and create chunks")
    ingest_parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")

    embed_parser = subparsers.add_parser("embed", help="Create embeddings from chunks")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")

    index_parser = subparsers.add_parser("index", help="Index embeddings into ChromaDB")
    index_parser.add_argument("--batch-size", type=int, default=128, help="Chroma upsert batch size")
    index_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection before indexing",
    )

    rebuild_parser = subparsers.add_parser("rebuild", help="Run ingest + embed + index")
    rebuild_parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    rebuild_parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
    rebuild_parser.add_argument("--embed-batch-size", type=int, default=64, help="Embedding batch size")
    rebuild_parser.add_argument("--index-batch-size", type=int, default=128, help="Indexing batch size")

    ask_parser = subparsers.add_parser("ask", help="Ask a question with local RAG")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks")

    return parser


def run_ingest(args: argparse.Namespace) -> Path:
    ingestor = Ingestor(
        data_dir=Path(args.data_dir),
        chunks_dir=Path(args.chunks_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recursive=True,
    )
    return ingestor.run()


def run_embed(args: argparse.Namespace, batch_size: int | None = None) -> Path:
    chunks_file = Path(args.chunks_dir) / "chunks.jsonl"
    embedder = Embedder(
        chunks_file=chunks_file,
        embeddings_dir=Path(args.embeddings_dir),
        model_name=args.embedding_model,
        batch_size=batch_size if batch_size is not None else args.batch_size,
    )
    return embedder.run()


def run_index(args: argparse.Namespace, batch_size: int | None = None, force_reset: bool | None = None) -> int:
    embeddings_file = Path(args.embeddings_dir) / "embeddings.jsonl"
    manager = VectorStoreManager(
        persist_dir=Path(args.chroma_dir),
        collection_name=args.collection_name,
    )

    should_reset = force_reset if force_reset is not None else args.reset
    if should_reset:
        manager.reset_collection()

    return manager.index_embeddings(
        embeddings_file=embeddings_file,
        batch_size=batch_size if batch_size is not None else args.batch_size,
    )


def run_ask(args: argparse.Namespace) -> str:
    manager = VectorStoreManager(
        persist_dir=Path(args.chroma_dir),
        collection_name=args.collection_name,
    )
    retriever = Retriever(
        vector_store=manager,
        embedding_model_name=args.embedding_model,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        ollama_client=OllamaClient(base_url=args.ollama_url),
        model_name=args.ollama_model,
        top_k=args.top_k,
    )

    response = pipeline.answer_question(args.question)
    lines: List[str] = []
    lines.append("Answer:")
    lines.append(response.answer)
    lines.append("")
    lines.append("Sources:")

    if not response.contexts:
        lines.append("- No chunks retrieved")
    else:
        for context in response.contexts:
            source = context.metadata.get("source_path", "unknown_source")
            chunk_index = context.metadata.get("chunk_index", "?")
            lines.append(
                f"- source={source} chunk_index={chunk_index} score={context.score:.3f}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        output = run_ingest(args)
        print(f"Ingestion complete: {output}")
        return

    if args.command == "embed":
        output = run_embed(args)
        print(f"Embedding complete: {output}")
        return

    if args.command == "index":
        count = run_index(args)
        print(f"Indexed chunks: {count}")
        return

    if args.command == "rebuild":
        chunks_path = run_ingest(args)
        embeddings_path = run_embed(args, batch_size=args.embed_batch_size)
        indexed = run_index(args, batch_size=args.index_batch_size, force_reset=True)
        print("Rebuild complete")
        print(f"- chunks: {chunks_path}")
        print(f"- embeddings: {embeddings_path}")
        print(f"- indexed chunks: {indexed}")
        return

    if args.command == "ask":
        print(run_ask(args))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
