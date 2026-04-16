# Local RAG System (Production-Quality, Free, On-Device)

## Overview
This project is a complete local Retrieval-Augmented Generation (RAG) system that runs on your personal machine.

Core stack:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: ChromaDB (local persistent store)
- LLM: Ollama (local models such as `phi3:mini`, `llama3:8b`, or `mistral`)

Why this setup:
- Free to run locally after model download
- Privacy-first (no external LLM API required)
- Production-style modular architecture
- Easy to clone and run for collaborators

## For Non-Technical Readers

### What is this solution?
This is a private, local AI assistant for your own documents.

You place your files (like policies, notes, or reports) into the project, and the system can answer questions based on that content. It does not rely on a cloud AI service to read your files.

### Why did we build this?
Teams and individuals often spend too much time searching through PDFs, meeting notes, and internal documents. We built this to make information retrieval faster, simpler, and more accurate while keeping control of data on your own machine.

### Why is it important?
- Better productivity: people can ask direct questions instead of manually scanning many files.
- Better privacy: sensitive information stays local, which is useful for internal or confidential material.
- Better reliability: answers are grounded in your documents and can include source references.
- Better accessibility: non-technical users can use a browser GUI, while technical users can use CLI automation.

In short, this solution turns scattered documents into a searchable knowledge assistant that is fast, private, and practical.

## Architecture

### High-Level Components
- `src/ingest.py`: PDF/TXT ingestion and chunk generation
- `src/embed.py`: local embedding generation from chunk artifacts
- `src/vector_store.py`: ChromaDB indexing and query primitives
- `src/retrieve.py`: semantic retrieval from ChromaDB
- `src/rag_pipeline.py`: retrieve + prompt + generate via Ollama
- `src/app.py`: CLI orchestration entrypoint
- `src/gui_app.py`: browser-based GUI for rebuild + ask workflows

### ASCII Architecture Diagram
```text
+-------------------+      +--------------------+      +-------------------+
|  Source Documents | ---> |  Ingestion/Chunker | ---> |   Chunk Artifacts |
|   (PDF, TXT)      |      |    (src/ingest.py) |      |   (./chunks/)     |
+-------------------+      +--------------------+      +-------------------+
                                      |
                                      v
                           +------------------------+
                           |  Embeddings Generator  |
                           |     (src/embed.py)     |
                           +------------------------+
                                      |
                                      v
                           +------------------------+
                           |  Chroma Vector Store   |
                           | (src/vector_store.py)  |
                           +------------------------+
                                      |
                     +----------------+----------------+
                     v                                 v
        +------------------------+         +------------------------+
        | Retrieval Engine       |         | Ollama Local LLM       |
        | (src/retrieve.py)      | ----->  | (phi3/llama3/mistral)  |
        +------------------------+         +------------------------+
                     |                                 ^
                     +---------------+-----------------+
                                     |
                                     v
                          +-------------------------+
                          | RAG Pipeline + CLI App  |
                          | (src/rag_pipeline.py,   |
                          |  src/app.py)            |
                          +-------------------------+
```

              ### Process Flow Diagram (Mermaid)
              ```mermaid
              flowchart TD
                A[User Documents\nPDF/TXT files in data folder\nSource knowledge base] --> B[Ingestion + Chunking\nReads files, normalizes text\nSplits into overlapping chunks]
                B --> C[Chunk Artifacts\nStructured chunk records\nChunk text + metadata]
                C --> D[Embedding Generator\nall-MiniLM-L6-v2\nConverts chunk text to vectors]
                D --> E[Embedding Artifacts\nVectorized chunk records\nReady for indexing]
                E --> F[ChromaDB Indexer\nCreates or refreshes collection\nStores vectors + metadata]
                F --> G[Local Vector Store\nPersistent semantic index\nFast nearest-neighbor search]

                H[User Question\nNatural language query] --> I[Query Embedder\nSame embedding model\nVectorizes the question]
                I --> J[Semantic Retriever\nTop-K vector search in ChromaDB\nReturns relevant chunks]
                G --> J

                J --> K[Context Builder\nFormats retrieved chunks\nBuilds grounded prompt]
                K --> L[Answer Engine\nStructured extraction fallback\nThen Ollama generation]
                M[Ollama Local LLM\nllama3.2 / mistral / phi3\nGenerates final answer] --> L
                L --> N[Final Response\nAnswer + source citations\nShown in CLI]

                O[Rebuild Command\nRuns ingest -> embed -> index\nRefreshes full knowledge index] --> B
              ```

## Project Structure
```text
rag/
  data/
  chunks/
  embeddings/
  src/
    ingest.py
    embed.py
    vector_store.py
    retrieve.py
    rag_pipeline.py
    app.py
    evaluate.py
    gui_app.py
  README.md
  LICENSE
  requirements.txt
  .gitignore
```

## Setup Instructions

### Prerequisites
- macOS/Linux/Windows with Python 3.10+
- Ollama installed: https://ollama.com/download
- At least one local LLM pulled in Ollama

Example:
```bash
ollama pull phi3:mini
# or
ollama pull llama3:8b
# or
ollama pull mistral
```

### Installation
From the `rag/` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run Locally

### 1) Add documents
Put `.pdf` and `.txt` files under `data/`.

### 2) Build the full index pipeline
```bash
python src/app.py rebuild
```

This runs:
- Ingestion/chunking
- Embedding generation
- ChromaDB indexing

### 3) Ask questions
```bash
python src/app.py ask "What are the key requirements in the policy document?"
```

Use a different model:
```bash
python src/app.py --ollama-model mistral:latest ask "Summarize the architecture."
```

Use a different top-k retrieval depth:
```bash
python src/app.py ask "List important constraints." --top-k 6
```

### 4) Use the browser GUI
```bash
PYTHONPATH=src streamlit run src/gui_app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## CLI Reference

Ingest only:
```bash
python src/app.py ingest --chunk-size 900 --chunk-overlap 150
```

Embed only:
```bash
python src/app.py embed --batch-size 64
```

Index only:
```bash
python src/app.py index --reset --batch-size 128
```

Rebuild end-to-end:
```bash
python src/app.py rebuild
```

Ask:
```bash
python src/app.py ask "Your question here"
```

Run benchmark evaluation:
```bash
python src/evaluate.py
```

Run browser GUI:
```bash
PYTHONPATH=src streamlit run src/gui_app.py
```

## Example Queries
- "Summarize the main themes discussed across all documents."
- "What steps are required to complete the onboarding process?"
- "Which section mentions security responsibilities and controls?"
- "Extract action items and deadlines from the meeting notes."

## Configuration
You can tune these at runtime via CLI flags:
- `--data-dir`, `--chunks-dir`, `--embeddings-dir`, `--chroma-dir`
- `--embedding-model` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--ollama-model` (default: `llama3.2:1b`)
- `--chunk-size`, `--chunk-overlap`
- `--top-k`

## Testing And Validation
Recommended smoke test flow:
1. Add 2-3 small documents into `data/`.
2. Run `python src/app.py rebuild`.
3. Run 3-5 test questions with known answers.
4. Verify answers cite expected sources.
5. Adjust chunking and `top-k` if answers are weak.

## Future Improvements
- Hybrid retrieval (BM25 + vector) for higher recall
- Cross-encoder reranking for better precision
- Metadata filters (file, date, tags)
- Citation spans and answer confidence scores
- REST API and/or web UI layer
- Evaluation harness for retrieval and answer quality

## License
This repository uses the MIT License.

See `LICENSE` for details.

## Contributing
Suggested collaboration flow:
1. Create a feature branch.
2. Keep modules focused and small.
3. Add docstrings/comments for non-obvious logic.
4. Run local smoke checks before opening PR.
5. Submit PR with clear test notes and example command output.
