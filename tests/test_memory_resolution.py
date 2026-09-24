import pytest

from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
    ProposedRelationship,
)
from core.memory.resolution import (
    MemoryResolution,
    ResolutionDecision,
    ResolvedExtraction,
    resolve_entity,
    resolve_extraction,
    resolve_fact,
    resolve_relationship,
)
from core.memory.semantic import MemorySource
from core.memory.structured import (
    Entity,
    Fact,
    Relationship,
)


def test_resolution_groups_decisions_by_action():
    create = ResolutionDecision(
        action="create",
        memory_type="entity",
        proposed="Kraken",
    )

    no_op = ResolutionDecision(
        action="no_op",
        memory_type="fact",
        proposed="Elegoo Neptune 4",
        existing="Elegoo Neptune 4",
    )

    supersede = ResolutionDecision(
        action="supersede",
        memory_type="fact",
        proposed="new value",
        existing="old value",
    )

    reject = ResolutionDecision(
        action="reject",
        memory_type="relationship",
        proposed="unknown",
        reason="Target could not be resolved.",
    )

    resolution = MemoryResolution(
        decisions=(
            create,
            no_op,
            supersede,
            reject,
        ),
    )

    assert resolution.creates == (create,)
    assert resolution.no_ops == (no_op,)
    assert resolution.supersedes == (supersede,)
    assert resolution.rejects == (reject,)


def test_resolution_decision_rejects_invalid_action():
    with pytest.raises(
        ValueError,
        match="Invalid resolution action",
    ):
        ResolutionDecision(
            action="explode",
            memory_type="fact",
            proposed="Kraken",
        )


def test_resolution_decision_rejects_blank_memory_type():
    with pytest.raises(
        ValueError,
        match="memory_type cannot be blank",
    ):
        ResolutionDecision(
            action="create",
            memory_type="   ",
            proposed="Kraken",
        )


def test_resolve_entity_creates_when_no_match_exists():
    proposed = ProposedEntity(
        reference="printer",
        entity_type="device",
        name="Kraken",
    )

    decision = resolve_entity(
        proposed,
        (),
    )

    assert decision.action == "create"
    assert decision.memory_type == "entity"
    assert decision.proposed == proposed
    assert decision.existing is None


def test_resolve_entity_no_ops_when_exact_entity_exists():
    proposed = ProposedEntity(
        reference="printer",
        entity_type="device",
        name="Kraken",
    )

    existing = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    decision = resolve_entity(
        proposed,
        (existing,),
    )

    assert decision.action == "no_op"
    assert decision.existing == existing


def test_resolve_entity_matching_is_case_insensitive():
    proposed = ProposedEntity(
        reference="printer",
        entity_type="DEVICE",
        name="KRAKEN",
    )

    existing = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    decision = resolve_entity(
        proposed,
        (existing,),
    )

    assert decision.action == "no_op"
    assert decision.existing == existing


def test_resolve_entity_does_not_match_name_with_different_type():
    proposed = ProposedEntity(
        reference="printer",
        entity_type="device",
        name="Aurora",
    )

    person = Entity(
        entity_id="person:aurora",
        entity_type="person",
        name="Aurora",
    )

    decision = resolve_entity(
        proposed,
        (person,),
    )

    assert decision.action == "create"
    assert decision.existing is None


def test_resolve_entity_rejects_ambiguous_matches():
    proposed = ProposedEntity(
        reference="printer",
        entity_type="device",
        name="Kraken",
    )

    first = Entity(
        entity_id="device:kraken-1",
        entity_type="device",
        name="Kraken",
    )

    second = Entity(
        entity_id="device:kraken-2",
        entity_type="device",
        name="Kraken",
    )

    decision = resolve_entity(
        proposed,
        (
            first,
            second,
        ),
    )

    assert decision.action == "reject"
    assert decision.existing == (
        first,
        second,
    )


def make_test_source() -> MemorySource:
    return MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )


def test_resolve_fact_creates_when_predicate_is_unknown():
    proposed = ProposedFact(
        subject_reference="printer",
        predicate="model",
        value="Elegoo Neptune 4",
    )

    subject = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    decision = resolve_fact(
        proposed,
        subject,
        (),
    )

    assert decision.action == "create"
    assert decision.existing is None


def test_resolve_fact_no_ops_when_value_is_unchanged():
    proposed = ProposedFact(
        subject_reference="printer",
        predicate="model",
        value="Elegoo Neptune 4",
    )

    subject = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    existing = Fact(
        fact_id="fact:kraken-model",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=make_test_source(),
        learned_at="2026-09-24T12:00:00-05:00",
    )

    decision = resolve_fact(
        proposed,
        subject,
        (existing,),
    )

    assert decision.action == "no_op"
    assert decision.existing == existing


def test_resolve_fact_supersedes_when_value_changes():
    proposed = ProposedFact(
        subject_reference="printer",
        predicate="location",
        value="workshop",
    )

    subject = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    existing = Fact(
        fact_id="fact:kraken-location",
        subject_id="device:kraken",
        predicate="location",
        value="office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    decision = resolve_fact(
        proposed,
        subject,
        (existing,),
    )

    assert decision.action == "supersede"
    assert decision.existing == existing


def test_resolve_fact_ignores_superseded_history():
    proposed = ProposedFact(
        subject_reference="printer",
        predicate="location",
        value="workshop",
    )

    subject = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    historical = Fact(
        fact_id="fact:kraken-old-location",
        subject_id="device:kraken",
        predicate="location",
        value="office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
        status="superseded",
    )

    decision = resolve_fact(
        proposed,
        subject,
        (historical,),
    )

    assert decision.action == "create"
    assert decision.existing is None


def test_resolve_fact_rejects_multiple_current_values():
    proposed = ProposedFact(
        subject_reference="printer",
        predicate="location",
        value="workshop",
    )

    subject = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    first = Fact(
        fact_id="fact:location-1",
        subject_id="device:kraken",
        predicate="location",
        value="office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    second = Fact(
        fact_id="fact:location-2",
        subject_id="device:kraken",
        predicate="location",
        value="garage",
        source=make_test_source(),
        learned_at="2026-09-24T12:00:00-05:00",
    )

    decision = resolve_fact(
        proposed,
        subject,
        (
            first,
            second,
        ),
    )

    assert decision.action == "reject"
    assert decision.existing == (
        first,
        second,
    )


def test_resolve_relationship_creates_when_slot_is_unknown():
    proposed = ProposedRelationship(
        source_reference="printer",
        relationship="belongs_to",
        target_reference="owner",
    )

    printer = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    owner = Entity(
        entity_id="person:flloyd",
        entity_type="person",
        name="Flloyd",
    )

    decision = resolve_relationship(
        proposed,
        printer,
        owner,
        (),
    )

    assert decision.action == "create"
    assert decision.existing is None


def test_resolve_relationship_no_ops_when_target_is_unchanged():
    proposed = ProposedRelationship(
        source_reference="printer",
        relationship="belongs_to",
        target_reference="owner",
    )

    printer = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    owner = Entity(
        entity_id="person:flloyd",
        entity_type="person",
        name="Flloyd",
    )

    existing = Relationship(
        relationship_id="relationship:kraken-owner",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=make_test_source(),
        learned_at="2026-09-24T12:00:00-05:00",
    )

    decision = resolve_relationship(
        proposed,
        printer,
        owner,
        (existing,),
    )

    assert decision.action == "no_op"
    assert decision.existing == existing


def test_resolve_relationship_supersedes_when_target_changes():
    proposed = ProposedRelationship(
        source_reference="printer",
        relationship="located_in",
        target_reference="workshop",
    )

    printer = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    workshop = Entity(
        entity_id="place:workshop",
        entity_type="place",
        name="Workshop",
    )

    existing = Relationship(
        relationship_id="relationship:kraken-location",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    decision = resolve_relationship(
        proposed,
        printer,
        workshop,
        (existing,),
    )

    assert decision.action == "supersede"
    assert decision.existing == existing


def test_resolve_relationship_ignores_superseded_history():
    proposed = ProposedRelationship(
        source_reference="printer",
        relationship="located_in",
        target_reference="workshop",
    )

    printer = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    workshop = Entity(
        entity_id="place:workshop",
        entity_type="place",
        name="Workshop",
    )

    historical = Relationship(
        relationship_id="relationship:old-location",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
        status="superseded",
    )

    decision = resolve_relationship(
        proposed,
        printer,
        workshop,
        (historical,),
    )

    assert decision.action == "create"
    assert decision.existing is None


def test_resolve_relationship_rejects_multiple_current_targets():
    proposed = ProposedRelationship(
        source_reference="printer",
        relationship="located_in",
        target_reference="workshop",
    )

    printer = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    workshop = Entity(
        entity_id="place:workshop",
        entity_type="place",
        name="Workshop",
    )

    first = Relationship(
        relationship_id="relationship:location-1",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    second = Relationship(
        relationship_id="relationship:location-2",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:garage",
        source=make_test_source(),
        learned_at="2026-09-24T12:00:00-05:00",
    )

    decision = resolve_relationship(
        proposed,
        printer,
        workshop,
        (
            first,
            second,
        ),
    )

    assert decision.action == "reject"
    assert decision.existing == (
        first,
        second,
    )


def test_resolve_extraction_builds_dry_run_plan():
    kraken = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    existing_model = Fact(
        fact_id="fact:kraken-model",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    existing_location = Fact(
        fact_id="fact:kraken-location",
        subject_id="device:kraken",
        predicate="location",
        value="office",
        source=make_test_source(),
        learned_at="2026-09-23T12:00:00-05:00",
    )

    extraction = MemoryExtraction(
        entities=(
            ProposedEntity(
                reference="printer",
                entity_type="device",
                name="Kraken",
            ),
            ProposedEntity(
                reference="workshop",
                entity_type="place",
                name="Workshop",
            ),
        ),
        facts=(
            ProposedFact(
                subject_reference="printer",
                predicate="model",
                value="Elegoo Neptune 4",
            ),
            ProposedFact(
                subject_reference="printer",
                predicate="location",
                value="workshop",
            ),
        ),
        relationships=(
            ProposedRelationship(
                source_reference="printer",
                relationship="located_in",
                target_reference="workshop",
            ),
        ),
    )

    result = resolve_extraction(
        extraction=extraction,
        existing_entities=(kraken,),
        facts_by_entity={
            "device:kraken": (
                existing_model,
                existing_location,
            ),
        },
        relationships_by_entity={},
    )

    actions = tuple(
        decision.action
        for decision in result.resolution.decisions
    )

    assert actions == (
        "no_op",
        "create",
        "no_op",
        "supersede",
        "create",
    )

    assert (
        result.references["printer"]
        == kraken
    )

    assert (
        result.references["workshop"].entity_id
        == "proposal:workshop"
    )