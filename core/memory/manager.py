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
from core.memory.structured import Entity, Fact, Relationship
from core.memory.structured_store import StructuredMemoryStore

import re

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
        structured_store: StructuredMemoryStore | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.episodic_memory = episodic_memory
        self.structured_store = structured_store

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

    def list_knowledge_documents(self) -> list[dict]:
        return self.vector_store.list_documents()

    def get_knowledge_document(
        self,
        document_id: str,
    ) -> dict | None:
        return self.vector_store.get_document(document_id)

    def delete_knowledge_document(
        self,
        document_id: str,
    ) -> bool:
        return self.vector_store.delete_document(document_id)

    def search_knowledge(
        self,
        question: str,
        *,
        result_count: int = 5,
    ) -> list[dict]:
        from core.memory.retrieval import retrieve_relevant_chunks

        return retrieve_relevant_chunks(
            question,
            vector_store=self.vector_store,
            result_count=result_count,
        )

    def get_knowledge_context(
        self,
        question: str,
        *,
        result_count: int = 5,
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.vector_store is None:
            return ""

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
            "structured": self.get_entity_context(question),
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

    def save_entity(self, entity: Entity) -> None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        self.structured_store.save_entity(entity)

    def get_entity(
        self,
        entity_id: str,
    ) -> Entity | None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.get_entity(entity_id)

    def save_fact(self, fact: Fact) -> None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        self.structured_store.save_fact(fact)

    def get_current_facts(
        self,
        subject_id: str,
        *,
        predicate: str | None = None,
    ) -> list[Fact]:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.get_current_facts(
            subject_id,
            predicate=predicate,
        )

    def save_relationship(
        self,
        relationship: Relationship,
    ) -> None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        self.structured_store.save_relationship(
            relationship
        )

    def get_current_relationships(
        self,
        source_entity_id: str,
        *,
        relationship: str | None = None,
    ) -> list[Relationship]:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.get_current_relationships(
            source_entity_id,
            relationship=relationship,
        )

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact: Fact,
    ) -> None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        self.structured_store.supersede_fact(
            old_fact_id,
            new_fact,
        )

    def supersede_relationship(
        self,
        old_relationship_id: str,
        new_relationship: Relationship,
    ) -> None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        self.structured_store.supersede_relationship(
            old_relationship_id,
            new_relationship,
        )

    def get_fact(
        self,
        fact_id: str,
    ) -> Fact | None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.get_fact(fact_id)

    def get_relationship(
        self,
        relationship_id: str,
    ) -> Relationship | None:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.get_relationship(
            relationship_id
        )

    def find_entities_by_name(
        self,
        name: str,
    ) -> list[Entity]:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.find_entities_by_name(name)

    def list_entities(self) -> list[Entity]:
        if self.structured_store is None:
            raise RuntimeError(
                "Structured memory store is not configured"
            )

        return self.structured_store.list_entities()

    def get_entity_context(
        self,
        question: str,
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.structured_store is None:
            return ""

        matched_entities = []

        for entity in self.list_entities():
            pattern = rf"(?<!\w){re.escape(entity.name)}(?!\w)"

            if re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            ):
                matched_entities.append(entity)

        if not matched_entities:
            return ""

        parts = []

        for entity in matched_entities:
            facts = self.get_current_facts(entity.entity_id)

            if not facts:
                continue

            lines = [
                "[STRUCTURED ENTITY]",
                f"ENTITY: {entity.name}",
                f"ENTITY_ID: {entity.entity_id}",
                f"ENTITY_TYPE: {entity.entity_type}",
            ]

            for fact in facts:
                lines.extend(
                    [
                        f"FACT: {fact.predicate} = {fact.value}",
                        f"SOURCE_TYPE: {fact.source.source_type}",
                        f"AUTHORITY: {fact.source.authority}",
                        f"LEARNED_AT: {fact.learned_at}",
                    ]
                )

            parts.append("\n".join(lines))

        return "\n\n".join(parts)