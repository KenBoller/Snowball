from dataclasses import dataclass
from typing import Any, Literal

from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
    ProposedRelationship,
    validate_extraction_references,
)
from core.memory.structured import (
    Entity,
    Fact,
    Relationship,
)

ResolutionAction = Literal[
    "create",
    "no_op",
    "supersede",
    "reject",
]


@dataclass(frozen=True)
class ResolutionDecision:
    action: ResolutionAction
    memory_type: str
    proposed: Any
    existing: Any | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        valid_actions = {
            "create",
            "no_op",
            "supersede",
            "reject",
        }

        if self.action not in valid_actions:
            raise ValueError(
                f"Invalid resolution action: {self.action}"
            )

        if not self.memory_type.strip():
            raise ValueError(
                "memory_type cannot be blank."
            )


@dataclass(frozen=True)
class MemoryResolution:
    decisions: tuple[ResolutionDecision, ...] = ()

    @property
    def creates(self) -> tuple[ResolutionDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.action == "create"
        )

    @property
    def no_ops(self) -> tuple[ResolutionDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.action == "no_op"
        )

    @property
    def supersedes(self) -> tuple[ResolutionDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.action == "supersede"
        )

    @property
    def rejects(self) -> tuple[ResolutionDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.action == "reject"
        )


@dataclass(frozen=True)
class ResolvedExtraction:
    resolution: MemoryResolution
    references: dict[str, Entity]


def resolve_extraction(
    extraction: MemoryExtraction,
    existing_entities: tuple[Entity, ...],
    facts_by_entity: dict[str, tuple[Fact, ...]],
    relationships_by_entity: dict[
        str,
        tuple[Relationship, ...],
    ],
) -> ResolvedExtraction:
    decisions = []
    references = {}

    validate_extraction_references(extraction)

    for proposed_entity in extraction.entities:
        decision = resolve_entity(
            proposed_entity,
            existing_entities,
        )

        decisions.append(decision)

        if decision.action == "reject":
            continue

        if decision.action == "no_op":
            references[proposed_entity.reference] = (
                decision.existing
            )
            continue

        temporary_entity = Entity(
            entity_id=(
                f"proposal:{proposed_entity.reference}"
            ),
            entity_type=proposed_entity.entity_type,
            name=proposed_entity.name,
            metadata=proposed_entity.metadata,
        )

        references[proposed_entity.reference] = (
            temporary_entity
        )

    for proposed_fact in extraction.facts:
        subject = references.get(
            proposed_fact.subject_reference
        )

        if subject is None:
            decisions.append(
                ResolutionDecision(
                    action="reject",
                    memory_type="fact",
                    proposed=proposed_fact,
                    reason=(
                        "Subject entity could not be resolved."
                    ),
                )
            )
            continue

        existing_facts = facts_by_entity.get(
            subject.entity_id,
            (),
        )

        decisions.append(
            resolve_fact(
                proposed_fact,
                subject,
                existing_facts,
            )
        )

    for proposed_relationship in extraction.relationships:
        source_entity = references.get(
            proposed_relationship.source_reference
        )
        target_entity = references.get(
            proposed_relationship.target_reference
        )

        if source_entity is None or target_entity is None:
            decisions.append(
                ResolutionDecision(
                    action="reject",
                    memory_type="relationship",
                    proposed=proposed_relationship,
                    reason=(
                        "Relationship entities could not "
                        "be resolved."
                    ),
                )
            )
            continue

        existing_relationships = (
            relationships_by_entity.get(
                source_entity.entity_id,
                (),
            )
        )

        decisions.append(
            resolve_relationship(
                proposed_relationship,
                source_entity,
                target_entity,
                existing_relationships,
            )
        )

    return ResolvedExtraction(
        resolution=MemoryResolution(
            decisions=tuple(decisions),
        ),
        references=references,
    )


def resolve_entity(
    proposed: ProposedEntity,
    existing_entities: tuple[Entity, ...],
) -> ResolutionDecision:
    matches = tuple(
        entity
        for entity in existing_entities
        if (
            entity.name.casefold()
            == proposed.name.casefold()
            and entity.entity_type.casefold()
            == proposed.entity_type.casefold()
        )
    )

    if not matches:
        return ResolutionDecision(
            action="create",
            memory_type="entity",
            proposed=proposed,
            reason="No matching entity exists.",
        )

    if len(matches) == 1:
        return ResolutionDecision(
            action="no_op",
            memory_type="entity",
            proposed=proposed,
            existing=matches[0],
            reason="Matching entity already exists.",
        )

    return ResolutionDecision(
        action="reject",
        memory_type="entity",
        proposed=proposed,
        existing=matches,
        reason="Multiple matching entities exist.",
    )


def resolve_fact(
    proposed: ProposedFact,
    subject: Entity,
    existing_facts: tuple[Fact, ...],
) -> ResolutionDecision:
    matches = tuple(
        fact
        for fact in existing_facts
        if (
            fact.status == "current"
            and fact.subject_id == subject.entity_id
            and fact.predicate.casefold()
            == proposed.predicate.casefold()
        )
    )

    if not matches:
        return ResolutionDecision(
            action="create",
            memory_type="fact",
            proposed=proposed,
            reason="No current fact exists for this predicate.",
        )

    if len(matches) > 1:
        return ResolutionDecision(
            action="reject",
            memory_type="fact",
            proposed=proposed,
            existing=matches,
            reason=(
                "Multiple current facts exist for this predicate."
            ),
        )

    existing = matches[0]

    if existing.value == proposed.value:
        return ResolutionDecision(
            action="no_op",
            memory_type="fact",
            proposed=proposed,
            existing=existing,
            reason="Matching fact already exists.",
        )

    return ResolutionDecision(
        action="supersede",
        memory_type="fact",
        proposed=proposed,
        existing=existing,
        reason="Current fact has a different value.",
    )


def resolve_relationship(
    proposed: ProposedRelationship,
    source_entity: Entity,
    target_entity: Entity,
    existing_relationships: tuple[Relationship, ...],
) -> ResolutionDecision:
    matches = tuple(
        relationship
        for relationship in existing_relationships
        if (
            relationship.status == "current"
            and relationship.source_entity_id
            == source_entity.entity_id
            and relationship.relationship.casefold()
            == proposed.relationship.casefold()
        )
    )

    if not matches:
        return ResolutionDecision(
            action="create",
            memory_type="relationship",
            proposed=proposed,
            reason=(
                "No current relationship exists for this slot."
            ),
        )

    if len(matches) > 1:
        return ResolutionDecision(
            action="reject",
            memory_type="relationship",
            proposed=proposed,
            existing=matches,
            reason=(
                "Multiple current relationships exist "
                "for this slot."
            ),
        )

    existing = matches[0]

    if (
        existing.target_entity_id
        == target_entity.entity_id
    ):
        return ResolutionDecision(
            action="no_op",
            memory_type="relationship",
            proposed=proposed,
            existing=existing,
            reason="Matching relationship already exists.",
        )

    return ResolutionDecision(
        action="supersede",
        memory_type="relationship",
        proposed=proposed,
        existing=existing,
        reason=(
            "Current relationship has a different target."
        ),
    )