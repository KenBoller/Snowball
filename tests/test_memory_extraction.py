import pytest

from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
    ProposedRelationship,
    validate_extraction_references,
)


def test_memory_extraction_can_describe_structured_proposals():
    printer = ProposedEntity(
        reference="printer",
        entity_type="device",
        name="Kraken",
    )

    owner = ProposedEntity(
        reference="owner",
        entity_type="person",
        name="Flloyd",
    )

    model = ProposedFact(
        subject_reference="printer",
        predicate="model",
        value="Elegoo Neptune 4",
    )

    ownership = ProposedRelationship(
        source_reference="printer",
        relationship="belongs_to",
        target_reference="owner",
    )

    extraction = MemoryExtraction(
        entities=(printer, owner),
        facts=(model,),
        relationships=(ownership,),
    )

    assert extraction.entities == (
        printer,
        owner,
    )
    assert extraction.facts == (model,)
    assert extraction.relationships == (
        ownership,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference", " "),
        ("entity_type", ""),
        ("name", "   "),
    ],
)
def test_proposed_entity_rejects_blank_fields(
    field,
    value,
):
    values = {
        "reference": "printer",
        "entity_type": "device",
        "name": "Kraken",
    }

    values[field] = value

    with pytest.raises(ValueError):
        ProposedEntity(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_reference", ""),
        ("predicate", "   "),
    ],
)
def test_proposed_fact_rejects_blank_fields(
    field,
    value,
):
    values = {
        "subject_reference": "printer",
        "predicate": "model",
        "value": "Elegoo Neptune 4",
    }

    values[field] = value

    with pytest.raises(ValueError):
        ProposedFact(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_reference", ""),
        ("relationship", " "),
        ("target_reference", "   "),
    ],
)
def test_proposed_relationship_rejects_blank_fields(
    field,
    value,
):
    values = {
        "source_reference": "printer",
        "relationship": "belongs_to",
        "target_reference": "owner",
    }

    values[field] = value

    with pytest.raises(ValueError):
        ProposedRelationship(**values)

def test_extraction_reference_validation_accepts_resolved_references():
    extraction = MemoryExtraction(
        entities=(
            ProposedEntity(
                reference="printer",
                entity_type="device",
                name="Kraken",
            ),
            ProposedEntity(
                reference="owner",
                entity_type="person",
                name="Flloyd",
            ),
        ),
        facts=(
            ProposedFact(
                subject_reference="printer",
                predicate="model",
                value="Elegoo Neptune 4",
            ),
        ),
        relationships=(
            ProposedRelationship(
                source_reference="printer",
                relationship="belongs_to",
                target_reference="owner",
            ),
        ),
    )

    validate_extraction_references(extraction)


def test_extraction_reference_validation_rejects_missing_fact_subject():
    extraction = MemoryExtraction(
        facts=(
            ProposedFact(
                subject_reference="ghost",
                predicate="model",
                value="Elegoo Neptune 4",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unresolved fact subject reference: ghost",
    ):
        validate_extraction_references(extraction)


def test_extraction_reference_validation_rejects_missing_relationship_source():
    extraction = MemoryExtraction(
        entities=(
            ProposedEntity(
                reference="owner",
                entity_type="person",
                name="Flloyd",
            ),
        ),
        relationships=(
            ProposedRelationship(
                source_reference="ghost",
                relationship="belongs_to",
                target_reference="owner",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unresolved relationship source reference: ghost"
        ),
    ):
        validate_extraction_references(extraction)


def test_extraction_reference_validation_rejects_missing_relationship_target():
    extraction = MemoryExtraction(
        entities=(
            ProposedEntity(
                reference="printer",
                entity_type="device",
                name="Kraken",
            ),
        ),
        relationships=(
            ProposedRelationship(
                source_reference="printer",
                relationship="belongs_to",
                target_reference="ghost",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unresolved relationship target reference: ghost"
        ),
    ):
        validate_extraction_references(extraction)


def test_extraction_reference_validation_rejects_duplicate_references():
    extraction = MemoryExtraction(
        entities=(
            ProposedEntity(
                reference="printer",
                entity_type="device",
                name="Kraken",
            ),
            ProposedEntity(
                reference="printer",
                entity_type="device",
                name="Ferris",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate entity reference: printer",
    ):
        validate_extraction_references(extraction)