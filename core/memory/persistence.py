from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from core.memory.manager import MemoryManager
from core.memory.resolution import ResolvedExtraction
from core.memory.semantic import MemorySource
from core.memory.structured import (
    Entity,
    Fact,
    Relationship,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversational_memory_source(
    *,
    source_id: str | None = None,
    timestamp: str | None = None,
) -> MemorySource:
    return MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
        source_id=source_id,
        timestamp=timestamp,
    )


def build_entity(
    *,
    entity_type: str,
    name: str,
    metadata: dict | None = None,
) -> Entity:
    return Entity(
        entity_id=_new_id("entity"),
        entity_type=entity_type,
        name=name,
        metadata=metadata,
    )


def build_fact(
    *,
    subject_id: str,
    predicate: str,
    value,
    source: MemorySource,
    learned_at: str | None = None,
    metadata: dict | None = None,
) -> Fact:
    return Fact(
        fact_id=_new_id("fact"),
        subject_id=subject_id,
        predicate=predicate,
        value=value,
        source=source,
        learned_at=learned_at or _now_iso(),
        metadata=metadata,
    )


def build_relationship(
    *,
    source_entity_id: str,
    relationship: str,
    target_entity_id: str,
    source: MemorySource,
    learned_at: str | None = None,
    metadata: dict | None = None,
) -> Relationship:
    return Relationship(
        relationship_id=_new_id("relationship"),
        source_entity_id=source_entity_id,
        relationship=relationship,
        target_entity_id=target_entity_id,
        source=source,
        learned_at=learned_at or _now_iso(),
        metadata=metadata,
    )


@dataclass(frozen=True)
class PersistenceResult:
    entities_created: tuple[Entity, ...] = ()
    facts_created: tuple[Fact, ...] = ()
    relationships_created: tuple[Relationship, ...] = ()
    facts_superseded: tuple[Fact, ...] = ()
    relationships_superseded: tuple[Relationship, ...] = ()


def persist_resolved_extraction(
    resolved: ResolvedExtraction,
    manager: MemoryManager,
    *,
    source_id: str | None = None,
    timestamp: str | None = None,
) -> PersistenceResult:
    source = conversational_memory_source(
        source_id=source_id,
        timestamp=timestamp,
    )

    learned_at = timestamp or _now_iso()

    reference_map = dict(resolved.references)

    entities_created = []
    facts_created = []
    relationships_created = []
    facts_superseded = []
    relationships_superseded = []

    # Create permanent entities first.
    for decision in resolved.resolution.decisions:
        if (
            decision.memory_type == "entity"
            and decision.action == "create"
        ):
            proposed = decision.proposed

            entity = build_entity(
                entity_type=proposed.entity_type,
                name=proposed.name,
                metadata=proposed.metadata,
            )

            manager.save_entity(entity)
            reference_map[proposed.reference] = entity
            entities_created.append(entity)

    # Facts and relationships can now resolve temporary references
    # to permanent entity IDs.
    for decision in resolved.resolution.decisions:
        if decision.action in {"no_op", "reject"}:
            continue

        if decision.memory_type == "fact":
            proposed = decision.proposed
            subject = reference_map.get(
                proposed.subject_reference
            )

            if subject is None:
                raise ValueError(
                    "Cannot persist fact with unresolved "
                    f"subject reference: "
                    f"{proposed.subject_reference}"
                )

            fact = build_fact(
                subject_id=subject.entity_id,
                predicate=proposed.predicate,
                value=proposed.value,
                source=source,
                learned_at=learned_at,
                metadata=proposed.metadata,
            )

            if decision.action == "create":
                manager.save_fact(fact)
                facts_created.append(fact)

            elif decision.action == "supersede":
                if decision.existing is None:
                    raise ValueError(
                        "Supersede fact decision is missing "
                        "the existing fact."
                    )

                manager.supersede_fact(
                    decision.existing.fact_id,
                    fact,
                )
                facts_superseded.append(fact)

        elif decision.memory_type == "relationship":
            proposed = decision.proposed

            source_entity = reference_map.get(
                proposed.source_reference
            )
            target_entity = reference_map.get(
                proposed.target_reference
            )

            if source_entity is None or target_entity is None:
                raise ValueError(
                    "Cannot persist relationship with "
                    "unresolved entity reference."
                )

            relationship = build_relationship(
                source_entity_id=source_entity.entity_id,
                relationship=proposed.relationship,
                target_entity_id=target_entity.entity_id,
                source=source,
                learned_at=learned_at,
                metadata=proposed.metadata,
            )

            if decision.action == "create":
                manager.save_relationship(relationship)
                relationships_created.append(
                    relationship
                )

            elif decision.action == "supersede":
                if decision.existing is None:
                    raise ValueError(
                        "Supersede relationship decision is "
                        "missing the existing relationship."
                    )

                manager.supersede_relationship(
                    decision.existing.relationship_id,
                    relationship,
                )
                relationships_superseded.append(
                    relationship
                )

    return PersistenceResult(
        entities_created=tuple(entities_created),
        facts_created=tuple(facts_created),
        relationships_created=tuple(
            relationships_created
        ),
        facts_superseded=tuple(facts_superseded),
        relationships_superseded=tuple(
            relationships_superseded
        ),
    )