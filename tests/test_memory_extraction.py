import pytest

from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
    ProposedRelationship,
    parse_memory_extraction,
    parse_memory_extraction_json,
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


def test_parse_memory_extraction_from_model_output():
    extraction = parse_memory_extraction(
        {
            "entities": [
                {
                    "reference": "printer",
                    "entity_type": "device",
                    "name": "Kraken",
                }
            ],
            "facts": [
                {
                    "subject_reference": "printer",
                    "predicate": "model",
                    "value": "Elegoo Neptune 4",
                }
            ],
            "relationships": [],
        }
    )

    assert extraction.entities[0].name == "Kraken"
    assert extraction.facts[0].value == (
        "Elegoo Neptune 4"
    )


def test_parse_memory_extraction_allows_empty_output():
    extraction = parse_memory_extraction({})

    assert extraction == MemoryExtraction()


def test_parse_memory_extraction_rejects_unknown_fields():
    with pytest.raises(
        ValueError,
        match="Unknown memory extraction fields",
    ):
        parse_memory_extraction(
            {
                "entities": [],
                "facts": [],
                "relationships": [],
                "confidence": 0.99,
            }
        )


def test_parse_memory_extraction_rejects_non_list_sections():
    with pytest.raises(
        ValueError,
        match="entities must be a list",
    ):
        parse_memory_extraction(
            {
                "entities": {
                    "name": "Kraken",
                }
            }
        )


def test_parse_memory_extraction_rejects_unresolved_model_reference():
    with pytest.raises(
        ValueError,
        match="Unresolved fact subject reference",
    ):
        parse_memory_extraction(
            {
                "facts": [
                    {
                        "subject_reference": "ghost",
                        "predicate": "model",
                        "value": "Neptune 4",
                    }
                ]
            }
        )


def test_parse_memory_extraction_json():
    extraction = parse_memory_extraction_json(
        """
        {
            "entities": [
                {
                    "reference": "printer",
                    "entity_type": "device",
                    "name": "Kraken"
                }
            ],
            "facts": [
                {
                    "subject_reference": "printer",
                    "predicate": "model",
                    "value": "Elegoo Neptune 4"
                }
            ],
            "relationships": []
        }
        """
    )

    assert extraction.entities[0].name == "Kraken"
    assert extraction.facts[0].predicate == "model"


def test_parse_memory_extraction_json_rejects_invalid_json():
    with pytest.raises(
        ValueError,
        match="Memory extraction is not valid JSON",
    ):
        parse_memory_extraction_json(
            "Kraken is my printer."
        )


def test_parse_memory_extraction_json_rejects_blank_text():
    with pytest.raises(
        ValueError,
        match="Memory extraction JSON cannot be blank",
    ):
        parse_memory_extraction_json("   ")


def test_parse_memory_extraction_json_rejects_non_object_json():
    with pytest.raises(
        ValueError,
        match="Memory extraction must be an object",
    ):
        parse_memory_extraction_json(
            '["Kraken"]'
        )