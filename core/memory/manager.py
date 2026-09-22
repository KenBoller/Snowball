from __future__ import annotations

from pathlib import Path
from typing import Any

from core.knowledge.ingestion import ingest_into_vector_store
from core.knowledge.rag import build_knowledge_context
from core.knowledge.vector_store import VectorStore
from core.memory.episodic import LegacyMemoryAdapter
from core.memory.semantic import (
    MemoryContextEntry,
    knowledge_source,
    user_statement_source,
)

class MemoryManager:
    """
    Coordinates Snowball's long-term memory and knowledge systems.

    The conversational layer should interact with this manager rather than
    knowing how individual storage or retrieval systems are implemented.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        episodic_memory: LegacyMemoryAdapter | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.episodic_memory = episodic_memory

    def ingest_document(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> dict:
        return ingest_into_vector_store(
            file_path,
            self.vector_store,
            document_id=document_id,
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def get_knowledge_context(
        self,
        question: str,
        *,
        result_count: int = 5,
    ) -> str:
        return build_knowledge_context(
            question,
            vector_store=self.vector_store,
            result_count=result_count,
        )

    def get_episodic_context(
        self,
        question: str,
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.episodic_memory is None:
            return ""

        memory = self.episodic_memory.retrieve(question)

        if not memory:
            return ""

        user_input = memory.get("user_input")
        ai_response = memory.get("ai_response")

        parts = ["[EPISODIC MEMORY]"]

        if user_input:
            parts.append(f"User: {user_input}")

        if ai_response:
            parts.append(f"Snowball: {ai_response}")

        return "\n".join(parts)

    def get_context(
        self,
        question: str,
        *,
        knowledge_result_count: int = 5,
    ) -> dict:
        """
        Build Snowball's unified context package.

        Memory sources remain separate here so authority, provenance,
        contradiction handling, and prompt formatting can evolve without
        changing the conversational interface.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return {
            "episodic": self.get_episodic_context(question),
            "knowledge": self.get_knowledge_context(
                question,
                result_count=knowledge_result_count,
            ),
        }
    
    def get_episodic_entries(
        self,
        question: str,
    ) -> list[MemoryContextEntry]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.episodic_memory is None:
            return []

        memory = self.episodic_memory.retrieve(question)

        if not memory:
            return []

        user_input = memory.get("user_input")

        if not isinstance(user_input, str) or not user_input.strip():
            return []

        source = user_statement_source(
            source_id=memory.get("id"),
            timestamp=memory.get("timestamp"),
        )

        return [
            MemoryContextEntry(
                text=user_input,
                source=source,
            )
        ]

    def get_structured_context(
        self,
        question: str,
        *,
        knowledge_result_count: int = 5,
    ) -> dict[str, list[MemoryContextEntry]]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        episodic = self.get_episodic_entries(question)

        knowledge_results = []

        if self.vector_store.count() > 0:
            from core.memory.retrieval import retrieve_relevant_chunks

            results = retrieve_relevant_chunks(
                question,
                vector_store=self.vector_store,
                result_count=knowledge_result_count,
            )

            for result in results:
                metadata = result.get("metadata") or {}

                source_id = (
                    metadata.get("document_id")
                    or metadata.get("filename")
                )

                knowledge_results.append(
                    MemoryContextEntry(
                        text=result["text"],
                        source=knowledge_source(
                            source_id=source_id,
                            metadata=metadata,
                        ),
                        relevance=result.get("distance"),
                    )
                )

        return {
            "episodic": episodic,
            "knowledge": knowledge_results,
        }