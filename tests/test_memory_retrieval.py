from pathlib import Path

import pytest

from core.knowledge.embeddings import create_embeddings
from core.knowledge.vector_store import VectorStore
from core.memory.retrieval import retrieve_relevant_chunks


def test_semantic_retrieval_finds_related_content(
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
            "text": "My favorite fruit is pineapple.",
            "start_char": 35,
            "end_char": 66,
        },
        {
            "chunk_index": 2,
            "text": "The garage door needs to be repaired.",
            "start_char": 67,
            "end_char": 104,
        },
    ]

    embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="semantic-test",
        filename="semantic-test.txt",
    )

    results = retrieve_relevant_chunks(
        "Which machine do I use for 3D printing?",
        vector_store=store,
        result_count=1,
    )

    assert len(results) == 1
    assert "Kraken" in results[0]["text"]


def test_retrieval_empty_store_returns_empty_list(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    results = retrieve_relevant_chunks(
        "What printer do I use?",
        vector_store=store,
    )

    assert results == []


def test_retrieval_rejects_empty_question(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        retrieve_relevant_chunks(
            "   ",
            vector_store=store,
        )


def test_retrieval_rejects_invalid_result_count(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        retrieve_relevant_chunks(
            "Snowball",
            vector_store=store,
            result_count=0,
        )