from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.memory.semantic import MemorySource


VALID_STATUSES = {
    "current",
    "superseded",
}


def _require_non_blank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    name: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.entity_id, "entity_id")
        _require_non_blank(self.entity_type, "entity_type")
        _require_non_blank(self.name, "name")


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject_id: str
    predicate: str
    value: Any
    source: MemorySource
    learned_at: str
    status: str = "current"
    supersedes: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.fact_id, "fact_id")
        _require_non_blank(self.subject_id, "subject_id")
        _require_non_blank(self.predicate, "predicate")
        _require_non_blank(self.learned_at, "learned_at")

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid fact status: {self.status}"
            )

        if self.supersedes is not None:
            _require_non_blank(
                self.supersedes,
                "supersedes",
            )


@dataclass(frozen=True)
class Relationship:
    relationship_id: str
    source_entity_id: str
    relationship: str
    target_entity_id: str
    source: MemorySource
    learned_at: str
    status: str = "current"
    supersedes: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_non_blank(
            self.relationship_id,
            "relationship_id",
        )
        _require_non_blank(
            self.source_entity_id,
            "source_entity_id",
        )
        _require_non_blank(
            self.relationship,
            "relationship",
        )
        _require_non_blank(
            self.target_entity_id,
            "target_entity_id",
        )
        _require_non_blank(
            self.learned_at,
            "learned_at",
        )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid relationship status: {self.status}"
            )

        if self.supersedes is not None:
            _require_non_blank(
                self.supersedes,
                "supersedes",
            )

def supersede_fact(
    old_fact: Fact,
    new_fact: Fact,
) -> tuple[Fact, Fact]:
    """
    Replace a current fact with a corrected fact while preserving history.

    The old fact is returned as superseded.
    The new fact is returned as current and records which fact it replaces.
    """

    if old_fact.status != "current":
        raise ValueError(
            "Can only supersede a current fact"
        )

    if (
        old_fact.subject_id != new_fact.subject_id
        or old_fact.predicate != new_fact.predicate
    ):
        raise ValueError(
            "Facts must have the same subject and predicate"
        )

    historical = replace(
        old_fact,
        status="superseded",
    )

    current = replace(
        new_fact,
        status="current",
        supersedes=old_fact.fact_id,
    )

    return historical, current

def supersede_relationship(
    old_relationship: Relationship,
    new_relationship: Relationship,
) -> tuple[Relationship, Relationship]:
    """
    Replace a current relationship while preserving its history.

    The old relationship is returned as superseded.
    The new relationship is returned as current and records which
    relationship it replaces.
    """

    if old_relationship.status != "current":
        raise ValueError(
            "Can only supersede a current relationship"
        )

    if (
        old_relationship.source_entity_id
        != new_relationship.source_entity_id
        or old_relationship.relationship
        != new_relationship.relationship
    ):
        raise ValueError(
            "Relationships must have the same source entity "
            "and relationship"
        )

    historical = replace(
        old_relationship,
        status="superseded",
    )

    current = replace(
        new_relationship,
        status="current",
        supersedes=old_relationship.relationship_id,
    )

    return historical, current