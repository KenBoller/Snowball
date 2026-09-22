from pathlib import Path

import pytest
from core.knowledge.ingestion import (
    extract_text_from_text_file,
    ingest_document,
    ingest_into_vector_store,
)

from core.knowledge.ingestion import (
    extract_text_from_text_file,
    ingest_document,
)


def test_ingest_text_file(tmp_path: Path):
    document = tmp_path / "notes.txt"
    document.write_text(
        "Kraken is my Neptune 4 printer.",
        encoding="utf-8",
    )

    result = ingest_document(document)

    assert result["source_type"] == "text"
    assert result["filename"] == "notes.txt"
    assert result["full_text"] == "Kraken is my Neptune 4 printer."
    assert result["text_length"] == len(result["full_text"])


def test_ingest_markdown_file(tmp_path: Path):
    document = tmp_path / "snowball.md"
    document.write_text(
        "# Snowball\n\nPersistent personal AI.",
        encoding="utf-8",
    )

    result = ingest_document(document)

    assert result["source_type"] == "text"
    assert result["filename"] == "snowball.md"
    assert "Persistent personal AI." in result["full_text"]


def test_extract_text_file_requires_existing_file(tmp_path: Path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        extract_text_from_text_file(missing)


def test_ingest_rejects_unsupported_document_type(tmp_path: Path):
    document = tmp_path / "binary.xyz"
    document.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest_document(document)


def test_ingest_rejects_file_without_extension(tmp_path: Path):
    document = tmp_path / "README"
    document.write_text("Snowball", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest_document(document)


def test_ingest_into_vector_store(tmp_path: Path):
    from core.knowledge.vector_store import VectorStore

    document = tmp_path / "printers.txt"

    document.write_text(
        (
            "Kraken is my Neptune 4 3D printer. "
            "Ferris is my Ender 3 S1 Pro 3D printer."
        ),
        encoding="utf-8",
    )

    store = VectorStore(tmp_path / "vectors")

    result = ingest_into_vector_store(
        document,
        store,
        document_id="printer-notes",
        chunk_size=50,
        overlap=10,
    )

    assert result["document_id"] == "printer-notes"
    assert result["filename"] == "printers.txt"
    assert result["source_type"] == "text"
    assert result["chunk_count"] > 0
    assert len(result["vector_ids"]) == result["chunk_count"]

    assert store.count() == result["chunk_count"]


def test_ingested_document_can_be_retrieved_semantically(
    tmp_path: Path,
):
    from core.knowledge.vector_store import VectorStore
    from core.memory.retrieval import retrieve_relevant_chunks

    document = tmp_path / "workshop-notes.txt"

    document.write_text(
        (
            "Kraken is my Neptune 4 3D printer. "
            "I use Kraken for making physical printed parts.\n\n"
            "Monty is my snake plant. "
            "Sensors will eventually let Monty react to his environment.\n\n"
            "Brain is PeenQi's clockwork mouse companion."
        ),
        encoding="utf-8",
    )

    store = VectorStore(tmp_path / "vectors")

    ingest_into_vector_store(
        document,
        store,
        document_id="workshop-notes",
        chunk_size=100,
        overlap=20,
    )

    results = retrieve_relevant_chunks(
        "Which machine can make physical objects for me?",
        vector_store=store,
        result_count=1,
    )

    assert len(results) == 1
    assert "Kraken" in results[0]["text"]