from pathlib import Path

import pytest

from core.knowledge.embeddings import create_embeddings
from core.knowledge.rag import build_knowledge_context
from core.knowledge.vector_store import VectorStore


def test_build_knowledge_context_returns_relevant_content(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    chunks = [
        {
            "chunk_index": 0,
            "text": "Kraken is my Neptune 4 3D printer.",
            "start_char": 0,
            "end_char": 34,
        },
        {
            "chunk_index": 1,
            "text": "Monty is my snake plant.",
            "start_char": 35,
            "end_char": 59,
        },
    ]

    embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="snowball-notes",
        filename="snowball-notes.txt",
    )

    context = build_knowledge_context(
        "Which machine do I use for 3D printing?",
        vector_store=store,
        result_count=1,
    )

    assert "Kraken" in context
    assert "snowball-notes.txt" in context
    assert "[KNOWLEDGE 1" in context


def test_build_knowledge_context_empty_store_returns_empty_string(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    context = build_knowledge_context(
        "What printer do I use?",
        vector_store=store,
    )

    assert context == ""


def test_build_knowledge_context_rejects_empty_question(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        build_knowledge_context(
            "   ",
            vector_store=store,
        )