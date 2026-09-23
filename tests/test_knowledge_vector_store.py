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


def test_delete_document_removes_only_matching_document(tmp_path):
    store = VectorStore(tmp_path)

    chunks = [
        {
            "chunk_index": 0,
            "text": "Alpha knowledge",
            "start_char": 0,
            "end_char": 15,
        }
    ]

    store.add_chunks(
        chunks=chunks,
        embeddings=[[1.0, 0.0]],
        document_id="alpha",
        filename="alpha.txt",
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=[[0.0, 1.0]],
        document_id="beta",
        filename="beta.txt",
    )

    assert store.count() == 2

    store.delete_document("alpha")

    assert store.count() == 1

    remaining = store.collection.get(
        where={"document_id": "beta"}
    )

    assert remaining["ids"] == ["beta_chunk_0"]


def test_list_documents_groups_chunks_by_document(tmp_path):
    store = VectorStore(tmp_path / "vectors")

    store.add_chunks(
        chunks=[
            {
                "chunk_index": 0,
                "text": "First Snowball chunk.",
                "start_char": 0,
                "end_char": 21,
            },
            {
                "chunk_index": 1,
                "text": "Second Snowball chunk.",
                "start_char": 22,
                "end_char": 44,
            },
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        document_id="snowball-current-state",
        filename="CURRENT_STATE.md",
        metadata={
            "authority": "current_reference",
            "provenance_source_type": "snowball_current_state",
        },
    )

    store.add_chunks(
        chunks=[
            {
                "chunk_index": 0,
                "text": "Historical Snowball chunk.",
                "start_char": 0,
                "end_char": 26,
            },
        ],
        embeddings=[
            [0.0, 1.0, 0.0],
        ],
        document_id="snowball-archive",
        filename="archive.docx",
        metadata={
            "authority": "historical_reference",
            "provenance_source_type": "snowball_project_document",
        },
    )

    documents = store.list_documents()

    assert len(documents) == 2

    by_id = {
        document["document_id"]: document
        for document in documents
    }

    assert by_id["snowball-current-state"]["filename"] == "CURRENT_STATE.md"
    assert by_id["snowball-current-state"]["chunk_count"] == 2
    assert by_id["snowball-current-state"]["authority"] == "current_reference"
    assert (
        by_id["snowball-current-state"]["provenance_source_type"]
        == "snowball_current_state"
    )

    assert by_id["snowball-archive"]["filename"] == "archive.docx"
    assert by_id["snowball-archive"]["chunk_count"] == 1
    assert by_id["snowball-archive"]["authority"] == "historical_reference"


def test_list_documents_handles_missing_optional_metadata(tmp_path):
    store = VectorStore(tmp_path / "vectors")

    store.add_chunks(
        chunks=[
            {
                "chunk_index": 0,
                "text": "Minimal knowledge chunk.",
                "start_char": 0,
                "end_char": 24,
            },
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
        ],
        document_id="minimal-document",
        filename="minimal.txt",
    )

    documents = store.list_documents()

    assert documents == [
        {
            "document_id": "minimal-document",
            "filename": "minimal.txt",
            "chunk_count": 1,
            "authority": None,
            "provenance_source_type": None,
        }
    ]