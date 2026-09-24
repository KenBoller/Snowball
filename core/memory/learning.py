from __future__ import annotations

from dataclasses import dataclass

from core.memory.conversational_extractor import extract_memories
from core.memory.extraction import MemoryExtraction
from core.memory.manager import MemoryManager
from core.memory.persistence import (
    PersistenceResult,
    persist_resolved_extraction,
)
from core.memory.resolution import (
    ResolvedExtraction,
    resolve_extraction,
)


@dataclass(frozen=True)
class LearningResult:
    extraction: MemoryExtraction
    resolved: ResolvedExtraction
    persistence: PersistenceResult | None
    persisted: bool


def learn_from_message(
    user_message: str,
    manager: MemoryManager,
    *,
    persist: bool = False,
    source_id: str | None = None,
    timestamp: str | None = None,
    extract_fn=extract_memories,
) -> LearningResult:
    """
    Extract and resolve structured memories from a user message.

    Persistence is disabled by default. The caller must explicitly
    authorize writes with persist=True.
    """

    extraction = extract_fn(user_message)

    existing_entities = tuple(
        manager.list_entities()
    )

    facts_by_entity = {
        entity.entity_id: tuple(
            manager.get_current_facts(
                entity.entity_id
            )
        )
        for entity in existing_entities
    }

    relationships_by_entity = {
        entity.entity_id: tuple(
            manager.get_current_relationships(
                entity.entity_id
            )
        )
        for entity in existing_entities
    }

    resolved = resolve_extraction(
        extraction,
        existing_entities=existing_entities,
        facts_by_entity=facts_by_entity,
        relationships_by_entity=relationships_by_entity,
    )

    persistence = None

    if persist:
        persistence = persist_resolved_extraction(
            resolved,
            manager,
            source_id=source_id,
            timestamp=timestamp,
        )

    return LearningResult(
        extraction=extraction,
        resolved=resolved,
        persistence=persistence,
        persisted=persist,
    )