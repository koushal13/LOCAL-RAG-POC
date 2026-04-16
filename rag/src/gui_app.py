"""Browser-based GUI for the local RAG system.

Run:
    PYTHONPATH=src streamlit run src/gui_app.py

This app provides:
- One-click index rebuild (ingest -> embed -> index)
- Interactive question answering
- Source citation display
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

# Ensure local src imports work even if PYTHONPATH is not set.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embed import Embedder
from ingest import Ingestor
from rag_pipeline import OllamaClient, RAGPipeline
from retrieve import Retriever
from vector_store import VectorStoreManager


def resolve_project_path(path_str: str) -> Path:
    """Resolve relative paths from project root so GUI works from any cwd."""
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def rebuild_index(
    data_dir: str,
    chunks_dir: str,
    embeddings_dir: str,
    chroma_dir: str,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    index_batch_size: int,
) -> dict:
    """Run full indexing pipeline and return summary stats."""
    resolved_data_dir = resolve_project_path(data_dir)
    resolved_chunks_dir = resolve_project_path(chunks_dir)
    resolved_embeddings_dir = resolve_project_path(embeddings_dir)
    resolved_chroma_dir = resolve_project_path(chroma_dir)

    ingestor = Ingestor(
        data_dir=resolved_data_dir,
        chunks_dir=resolved_chunks_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        recursive=True,
    )
    chunks_path = ingestor.run()

    embedder = Embedder(
        chunks_file=resolved_chunks_dir / "chunks.jsonl",
        embeddings_dir=resolved_embeddings_dir,
        model_name=embedding_model,
        batch_size=embed_batch_size,
    )
    embeddings_path = embedder.run()

    manager = VectorStoreManager(
        persist_dir=resolved_chroma_dir,
        collection_name=collection_name,
    )
    manager.reset_collection()
    indexed_chunks = manager.index_embeddings(
        embeddings_file=resolved_embeddings_dir / "embeddings.jsonl",
        batch_size=index_batch_size,
    )

    return {
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "indexed_chunks": indexed_chunks,
    }


@st.cache_resource(show_spinner=False)
def get_pipeline(
    chroma_dir: str,
    collection_name: str,
    embedding_model: str,
    ollama_url: str,
    ollama_model: str,
    top_k: int,
) -> RAGPipeline:
    """Create and cache RAG pipeline resources."""
    resolved_chroma_dir = resolve_project_path(chroma_dir)
    manager = VectorStoreManager(
        persist_dir=resolved_chroma_dir,
        collection_name=collection_name,
    )
    retriever = Retriever(
        vector_store=manager,
        embedding_model_name=embedding_model,
    )
    return RAGPipeline(
        retriever=retriever,
        ollama_client=OllamaClient(base_url=ollama_url),
        model_name=ollama_model,
        top_k=top_k,
    )


def render_sources(contexts: List) -> None:
    """Render retrieval source details in a simple table."""
    if not contexts:
        st.info("No sources retrieved.")
        return

    rows = []
    for ctx in contexts:
        rows.append(
            {
                "source": ctx.metadata.get("source_path", "unknown_source"),
                "chunk_index": ctx.metadata.get("chunk_index", "?"),
                "score": round(ctx.score, 3),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Local RAG GUI", page_icon="🧠", layout="wide")
    st.title("Local RAG GUI")
    st.caption("Browser interface for your on-device Retrieval-Augmented Generation system")

    with st.sidebar:
        st.header("Configuration")

        data_dir = st.text_input("Data directory", "./data")
        chunks_dir = st.text_input("Chunks directory", "./chunks")
        embeddings_dir = st.text_input("Embeddings directory", "./embeddings")
        chroma_dir = st.text_input("Chroma directory", "./chroma_db")
        collection_name = st.text_input("Collection name", "rag_chunks")

        st.divider()
        embedding_model = st.text_input(
            "Embedding model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        ollama_model = st.text_input("Ollama model", "llama3.2:1b")
        ollama_url = st.text_input("Ollama URL", "http://localhost:11434")

        st.divider()
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=5000, value=800, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=120, step=20)
        embed_batch_size = st.number_input("Embedding batch size", min_value=1, max_value=512, value=64, step=1)
        index_batch_size = st.number_input("Index batch size", min_value=1, max_value=1024, value=128, step=1)
        top_k = st.number_input("Top-k retrieval", min_value=1, max_value=20, value=4, step=1)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Index Management")
        if st.button("Rebuild Index", type="primary", use_container_width=True):
            try:
                with st.spinner("Running ingest -> embed -> index..."):
                    summary = rebuild_index(
                        data_dir=data_dir,
                        chunks_dir=chunks_dir,
                        embeddings_dir=embeddings_dir,
                        chroma_dir=chroma_dir,
                        collection_name=collection_name,
                        embedding_model=embedding_model,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        embed_batch_size=int(embed_batch_size),
                        index_batch_size=int(index_batch_size),
                    )
                get_pipeline.clear()
                st.success("Rebuild complete")
                st.write(f"Chunks: {summary['chunks_path']}")
                st.write(f"Embeddings: {summary['embeddings_path']}")
                st.write(f"Indexed chunks: {summary['indexed_chunks']}")
            except Exception as exc:
                st.error(f"Rebuild failed: {exc}")

    with col2:
        st.subheader("Ask Questions")
        question = st.text_area(
            "Question",
            value="What are the key requirements in the policy document?",
            height=120,
        )

        if st.button("Ask", use_container_width=True):
            try:
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Retrieving context and generating answer..."):
                        pipeline = get_pipeline(
                            chroma_dir=chroma_dir,
                            collection_name=collection_name,
                            embedding_model=embedding_model,
                            ollama_url=ollama_url,
                            ollama_model=ollama_model,
                            top_k=int(top_k),
                        )
                        response = pipeline.answer_question(question)

                    st.markdown("### Answer")
                    st.write(response.answer)
                    st.markdown("### Sources")
                    render_sources(response.contexts)
            except Exception as exc:
                st.error(f"Question failed: {exc}")


if __name__ == "__main__":
    main()
