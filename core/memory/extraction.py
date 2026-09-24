from dataclasses import dataclass
from typing import Any

import json


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
        

def parse_memory_extraction(
    data: dict[str, Any],
) -> MemoryExtraction:
    if not isinstance(data, dict):
        raise ValueError(
            "Memory extraction must be an object."
        )

    allowed_keys = {
        "entities",
        "facts",
        "relationships",
    }

    unknown_keys = set(data) - allowed_keys

    if unknown_keys:
        raise ValueError(
            "Unknown memory extraction fields: "
            + ", ".join(sorted(unknown_keys))
        )

    entities_data = data.get("entities", [])
    facts_data = data.get("facts", [])
    relationships_data = data.get(
        "relationships",
        [],
    )

    for field_name, value in (
        ("entities", entities_data),
        ("facts", facts_data),
        ("relationships", relationships_data),
    ):
        if not isinstance(value, list):
            raise ValueError(
                f"{field_name} must be a list."
            )

    try:
        entities = tuple(
            ProposedEntity(**item)
            for item in entities_data
        )

        facts = tuple(
            ProposedFact(**item)
            for item in facts_data
        )

        relationships = tuple(
            ProposedRelationship(**item)
            for item in relationships_data
        )
    except TypeError as exc:
        raise ValueError(
            f"Invalid memory extraction structure: {exc}"
        ) from exc

    extraction = MemoryExtraction(
        entities=entities,
        facts=facts,
        relationships=relationships,
    )

    validate_extraction_references(extraction)

    return extraction


def parse_memory_extraction_json(
    text: str,
) -> MemoryExtraction:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Memory extraction JSON cannot be blank."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Memory extraction is not valid JSON."
        ) from exc

    return parse_memory_extraction(data)