"""RAG pipeline orchestration for local retrieval + local generation.

Responsibilities:
- Retrieve relevant chunks from ChromaDB.
- Build a grounded prompt from retrieved context.
- Call local Ollama model for answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

import requests

from retrieve import RetrievedChunk, Retriever


class OllamaError(Exception):
    """Raised when local Ollama calls fail."""


class OllamaClient:
    """Thin client for local Ollama generate API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate text with a locally served Ollama model."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise OllamaError(f"Failed to reach Ollama at {endpoint}: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama request failed ({response.status_code}): {response.text[:500]}"
            )

        body = response.json()
        answer = body.get("response", "").strip()
        if not answer:
            raise OllamaError("Ollama returned an empty response")

        return answer


@dataclass(frozen=True)
class RAGResponse:
    """Structured output for final user response and evidence."""

    answer: str
    contexts: List[RetrievedChunk]
    prompt: str


class RAGPipeline:
    """Coordinates retrieval + generation for question answering."""

    def __init__(
        self,
        retriever: Retriever,
        ollama_client: OllamaClient,
        model_name: str = "phi3:mini",
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.top_k = top_k

    def answer_question(self, question: str) -> RAGResponse:
        """Answer a question using retrieved local context."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        contexts = self.retriever.search(query=question, top_k=self.top_k)
        prompt = self._build_prompt(question=question, contexts=contexts)

        extracted_answer = self._try_extract_structured_answer(question=question, contexts=contexts)
        if extracted_answer is not None:
            return RAGResponse(answer=extracted_answer, contexts=contexts, prompt=prompt)

        answer = self.ollama_client.generate(model=self.model_name, prompt=prompt)

        return RAGResponse(answer=answer, contexts=contexts, prompt=prompt)

    @staticmethod
    def _normalize_label(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()

    @classmethod
    def _extract_target_entity(cls, question: str) -> str | None:
        lowered = question.lower().strip()
        patterns = [
            r"(?:deadline|date)\s+(?:of|for)\s+(.+?)(?:\?|$)",
            r"when\s+is\s+(.+?)(?:\?|$)",
            r"when\s+is\s+the\s+deadline\s+(?:of|for)\s+(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return cls._normalize_label(match.group(1))
        return None

    @classmethod
    def _try_extract_structured_answer(cls, question: str, contexts: List[RetrievedChunk]) -> str | None:
        lowered = question.lower()
        asks_deadline = any(token in lowered for token in ["deadline", "deadlines", "due date", "date"])
        if not asks_deadline:
            return None

        date_pattern = re.compile(
            r"([A-Za-z][A-Za-z0-9 /_-]{2,80})\s*:\s*"
            r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2})",
            re.IGNORECASE,
        )

        extracted: List[Dict[str, str]] = []
        for chunk in contexts:
            source = str(chunk.metadata.get("source_path", "unknown_source"))
            for label, value in date_pattern.findall(chunk.text):
                extracted.append(
                    {
                        "label": label.strip(),
                        "label_norm": cls._normalize_label(label),
                        "value": value.strip(),
                        "source": source,
                    }
                )

        if not extracted:
            return None

        target = cls._extract_target_entity(question)
        if target:
            for item in extracted:
                if target in item["label_norm"] or item["label_norm"] in target:
                    return (
                        f"The deadline for {item['label']} is {item['value']}. "
                        f"(Source: {item['source']})"
                    )

        if "deadlines" in lowered:
            unique_lines: List[str] = []
            seen: set[str] = set()
            for item in extracted:
                line = f"- {item['label']}: {item['value']} (Source: {item['source']})"
                key = f"{item['label_norm']}::{item['value'].lower()}::{item['source'].lower()}"
                if key not in seen:
                    seen.add(key)
                    unique_lines.append(line)
            return "Deadlines found in context:\n" + "\n".join(unique_lines)

        return None

    @staticmethod
    def _build_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
        context_blocks: List[str] = []
        for idx, chunk in enumerate(contexts, start=1):
            source = chunk.metadata.get("source_path", "unknown_source")
            chunk_index = chunk.metadata.get("chunk_index", "?")
            block = (
                f"[Context {idx}]\\n"
                f"source={source} | chunk_index={chunk_index} | score={chunk.score:.3f}\\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        joined_context = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

        return (
            "You are a precise assistant answering ONLY from provided context. "
            "If context is insufficient, clearly say you do not know.\n\n"
            "Context:\n"
            f"{joined_context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Instructions:\n"
            "1) Provide a concise answer grounded in context.\n"
            "2) Cite source file names when possible.\n"
            "3) Do not invent facts not present in context.\n\n"
            "Answer:"
        )
