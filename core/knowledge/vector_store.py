from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


DEFAULT_COLLECTION_NAME = "snowball_knowledge"


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)

    return sanitized


class VectorStore:
    def __init__(
        self,
        storage_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.storage_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(
        self,
        *,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return []

        ids = []
        documents = []
        metadatas = []

        base_metadata = metadata or {}

        for chunk, embedding in zip(chunks, embeddings):
            chunk_index = chunk["chunk_index"]

            chunk_id = f"{document_id}_chunk_{chunk_index}"

            chunk_metadata = {
                **base_metadata,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk_index,
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            }

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(
                _sanitize_metadata(chunk_metadata)
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return ids

    def search(
        self,
        query_embedding: list[float],
        result_count: int = 5,
    ) -> list[dict]:
        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if result_count <= 0:
            raise ValueError(
                "result_count must be greater than 0."
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        matches = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            matches.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches

    def count(self) -> int:
        return self.collection.count()