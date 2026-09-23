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

        authority = metadata.get("authority", "unknown")
        source_type = metadata.get(
            "provenance_source_type",
            metadata.get("source_type", "unknown"),
        )
        source_date = metadata.get("source_date")

        header_parts = [
            f"KNOWLEDGE {index}",
            f"SOURCE: {source}",
            f"TYPE: {source_type}",
            f"AUTHORITY: {authority}",
        ]

        if source_date:
            header_parts.append(f"DATE: {source_date}")

        header = " | ".join(header_parts)

        sections.append(
            f"[{header}]\n{text}"
        )

    return "\n\n".join(sections)