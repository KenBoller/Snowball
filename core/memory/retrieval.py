from __future__ import annotations

from core.knowledge.embeddings import create_embedding
from core.knowledge.vector_store import VectorStore


def retrieve_relevant_chunks(
    question: str,
    vector_store: VectorStore,
    result_count: int = 5,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if result_count <= 0:
        raise ValueError(
            "result_count must be greater than 0."
        )

    if vector_store.count() == 0:
        return []

    query_embedding = create_embedding(question)

    return vector_store.search(
        query_embedding=query_embedding,
        result_count=result_count,
    )