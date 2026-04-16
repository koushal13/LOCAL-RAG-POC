"""Simple local evaluation runner for RAG quality checks.

Purpose:
- Run a fixed list of benchmark questions.
- Print answers and retrieved sources.
- Help quickly assess retrieval grounding after index rebuilds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rag_pipeline import OllamaClient, RAGPipeline
from retrieve import Retriever
from vector_store import VectorStoreManager


DEFAULT_QUESTIONS = [
    "What security control is mandatory for internal tools?",
    "By when must suspected data leaks be reported?",
    "What is the prototype demo deadline?",
    "List two stated risks in the project meeting notes.",
]


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    manager = VectorStoreManager(
        persist_dir=Path(args.chroma_dir),
        collection_name=args.collection_name,
    )
    retriever = Retriever(
        vector_store=manager,
        embedding_model_name=args.embedding_model,
    )
    return RAGPipeline(
        retriever=retriever,
        ollama_client=OllamaClient(base_url=args.ollama_url),
        model_name=args.ollama_model,
        top_k=args.top_k,
    )


def parse_questions(args: argparse.Namespace) -> List[str]:
    if args.question:
        return args.question

    if args.questions_file:
        file_path = Path(args.questions_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Questions file not found: {file_path}")
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]

    return DEFAULT_QUESTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local RAG benchmark questions")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="Chroma persistence directory")
    parser.add_argument("--collection-name", default="rag_chunks", help="Chroma collection name")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for retrieval",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:1b",
        help="Ollama model for generation",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Retrieved context count")
    parser.add_argument(
        "--question",
        action="append",
        help="Single benchmark question (repeatable)",
    )
    parser.add_argument(
        "--questions-file",
        help="Path to newline-delimited benchmark questions",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = build_pipeline(args)
    questions = parse_questions(args)

    print("Running RAG evaluation")
    print(f"- total_questions: {len(questions)}")
    print(f"- model: {args.ollama_model}")
    print(f"- top_k: {args.top_k}")
    print("")

    for index, question in enumerate(questions, start=1):
        print(f"[{index}] Question: {question}")
        response = pipeline.answer_question(question)
        print("Answer:")
        print(response.answer)
        print("Sources:")

        if not response.contexts:
            print("- No sources retrieved")
        else:
            for context in response.contexts:
                source = context.metadata.get("source_path", "unknown_source")
                chunk_index = context.metadata.get("chunk_index", "?")
                print(
                    f"- source={source} chunk_index={chunk_index} score={context.score:.3f}"
                )
        print("")


if __name__ == "__main__":
    main()
