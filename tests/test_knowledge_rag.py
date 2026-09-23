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


def test_build_knowledge_context_includes_provenance_metadata(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    chunks = [
        {
            "chunk_index": 0,
            "text": (
                "Snowball originally grew from an idea about "
                "one AI learning across multiple games."
            ),
            "start_char": 0,
            "end_char": 86,
        }
    ]

    embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="snowball-history",
        filename="snowball-history.docx",
        metadata={
            "provenance_source_type": "snowball_project_document",
            "authority": "historical_reference",
            "source_date": "2026-09-23",
        },
    )

    context = build_knowledge_context(
        "Where did Snowball come from?",
        vector_store=store,
        result_count=1,
    )

    assert "SOURCE: snowball-history.docx" in context
    assert "TYPE: snowball_project_document" in context
    assert "AUTHORITY: historical_reference" in context
    assert "DATE: 2026-09-23" in context


def test_build_knowledge_context_handles_missing_provenance(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    chunks = [
        {
            "chunk_index": 0,
            "text": "Kraken is a 3D printer.",
            "start_char": 0,
            "end_char": 23,
        }
    ]

    embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="legacy-notes",
        filename="legacy-notes.txt",
    )

    context = build_knowledge_context(
        "What is Kraken?",
        vector_store=store,
        result_count=1,
    )

    assert "SOURCE: legacy-notes.txt" in context
    assert "TYPE: unknown" in context
    assert "AUTHORITY: unknown" in context