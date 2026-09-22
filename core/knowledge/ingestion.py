from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf

from core.knowledge.chunking import chunk_text
from core.knowledge.embeddings import create_embeddings
from core.knowledge.vector_store import VectorStore


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
}


def extract_text_from_pdf(file_path: str | Path) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("File must be a PDF.")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    full_text = "\n\n".join(page["text"] for page in pages)

    return {
        "source_type": "pdf",
        "file_path": str(path),
        "filename": path.name,
        "page_count": len(pages),
        "text_length": len(full_text),
        "pages": pages,
        "full_text": full_text,
    }


def extract_text_from_text_file(file_path: str | Path) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(
            f"Unsupported text file type: {path.suffix}"
        )

    text = path.read_text(encoding="utf-8")

    return {
        "source_type": "text",
        "file_path": str(path),
        "filename": path.name,
        "page_count": None,
        "text_length": len(text),
        "pages": None,
        "full_text": text,
    }


def ingest_document(file_path: str | Path) -> dict:
    path = Path(file_path)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)

    if suffix in TEXT_SUFFIXES:
        return extract_text_from_text_file(path)

    raise ValueError(
        f"Unsupported document type: {suffix or '<no extension>'}"
    )


def ingest_into_vector_store(
    file_path: str | Path,
    vector_store: VectorStore,
    *,
    document_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> dict:
    """
    Ingest a document into Snowball's semantic knowledge store.

    Pipeline:
        document
        -> text extraction
        -> chunking
        -> embeddings
        -> vector storage
    """
    document = ingest_document(file_path)

    chunks = chunk_text(
        document["full_text"],
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if document_id is None:
        document_id = uuid4().hex

    if not chunks:
        return {
            "document_id": document_id,
            "filename": document["filename"],
            "source_type": document["source_type"],
            "chunk_count": 0,
            "vector_ids": [],
        }

    embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    document_metadata = {
        "source_type": document["source_type"],
        "file_path": document["file_path"],
        **(metadata or {}),
    }

    vector_ids = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
        filename=document["filename"],
        metadata=document_metadata,
    )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "source_type": document["source_type"],
        "chunk_count": len(chunks),
        "vector_ids": vector_ids,
    }