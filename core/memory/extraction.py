from dataclasses import dataclass
from typing import Any


def _require_nonblank(
    value: str,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} cannot be blank."
        )


@dataclass(frozen=True)
class ProposedEntity:
    reference: str
    entity_type: str
    name: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_nonblank(
            self.reference,
            "reference",
        )
        _require_nonblank(
            self.entity_type,
            "entity_type",
        )
        _require_nonblank(
            self.name,
            "name",
        )


@dataclass(frozen=True)
class ProposedFact:
    subject_reference: str
    predicate: str
    value: Any
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_nonblank(
            self.subject_reference,
            "subject_reference",
        )
        _require_nonblank(
            self.predicate,
            "predicate",
        )


@dataclass(frozen=True)
class ProposedRelationship:
    source_reference: str
    relationship: str
    target_reference: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_nonblank(
            self.source_reference,
            "source_reference",
        )
        _require_nonblank(
            self.relationship,
            "relationship",
        )
        _require_nonblank(
            self.target_reference,
            "target_reference",
        )


@dataclass(frozen=True)
class MemoryExtraction:
    entities: tuple[ProposedEntity, ...] = ()
    facts: tuple[ProposedFact, ...] = ()
    relationships: tuple[
        ProposedRelationship,
        ...
    ] = ()


def validate_extraction_references(
    extraction: MemoryExtraction,
) -> None:
    references = set()

    for entity in extraction.entities:
        if entity.reference in references:
            raise ValueError(
                "Duplicate entity reference: "
                f"{entity.reference}"
            )

        references.add(entity.reference)

    for fact in extraction.facts:
        if fact.subject_reference not in references:
            raise ValueError(
                "Unresolved fact subject reference: "
                f"{fact.subject_reference}"
            )

    for relationship in extraction.relationships:
        if relationship.source_reference not in references:
            raise ValueError(
                "Unresolved relationship source reference: "
                f"{relationship.source_reference}"
            )

        if relationship.target_reference not in references:
            raise ValueError(
                "Unresolved relationship target reference: "
                f"{relationship.target_reference}"
            )