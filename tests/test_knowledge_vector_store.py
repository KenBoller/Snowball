from pathlib import Path

import pytest

from core.knowledge.vector_store import VectorStore


def test_vector_store_adds_chunks(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")

    chunks = [
        {
            "chunk_index": 0,
            "text": "Kraken is my Neptune 4 printer.",
            "start_char": 0,
            "end_char": 31,
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    ids = store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="printer-notes",
        filename="printers.txt",
    )

    assert ids == ["printer-notes_chunk_0"]
    assert store.count() == 1


def test_vector_store_searches_chunks(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")

    chunks = [
        {
            "chunk_index": 0,
            "text": "Kraken is my Neptune 4 printer.",
            "start_char": 0,
            "end_char": 31,
        },
        {
            "chunk_index": 1,
            "text": "Ferris is my Ender 3 S1 Pro printer.",
            "start_char": 32,
            "end_char": 69,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="printer-notes",
        filename="printers.txt",
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        result_count=1,
    )

    assert len(results) == 1
    assert "Kraken" in results[0]["text"]
    assert results[0]["metadata"]["document_id"] == "printer-notes"


def test_vector_store_rejects_mismatched_chunks_and_embeddings(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        store.add_chunks(
            chunks=[
                {
                    "chunk_index": 0,
                    "text": "Snowball",
                }
            ],
            embeddings=[],
            document_id="test",
            filename="test.txt",
        )


def test_vector_store_empty_add_is_noop(tmp_path: Path):
    store = VectorStore(tmp_path / "vectors")

    ids = store.add_chunks(
        chunks=[],
        embeddings=[],
        document_id="empty",
        filename="empty.txt",
    )

    assert ids == []
    assert store.count() == 0


def test_vector_store_rejects_empty_query_embedding(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        store.search([])


def test_vector_store_rejects_invalid_result_count(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    with pytest.raises(ValueError):
        store.search(
            [1.0, 0.0],
            result_count=0,
        )