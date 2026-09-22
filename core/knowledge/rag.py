from __future__ import annotations

from core.knowledge.vector_store import VectorStore
from core.memory.retrieval import retrieve_relevant_chunks


def build_knowledge_context(
    question: str,
    vector_store: VectorStore,
    *,
    result_count: int = 5,
) -> str:
    """
    Retrieve relevant knowledge and format it for Snowball's prompt context.

    This module does not generate answers itself. Snowball's existing
    conversational system remains responsible for reasoning and response
    generation.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    results = retrieve_relevant_chunks(
        question,
        vector_store=vector_store,
        result_count=result_count,
    )

    if not results:
        return ""

    sections = []

    for index, result in enumerate(results, start=1):
        text = result["text"].strip()
        metadata = result.get("metadata") or {}

        source = (
            metadata.get("filename")
            or metadata.get("source_id")
            or metadata.get("document_id")
            or "unknown"
        )

        sections.append(
            f"[KNOWLEDGE {index} | SOURCE: {source}]\n{text}"
        )

    return "\n\n".join(sections)