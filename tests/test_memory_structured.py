from core.memory.semantic import MemorySource
from core.memory.structured import (
    Entity,
    Fact,
    Relationship,
    supersede_fact,
    supersede_relationship,
)
import pytest


def test_entity_represents_named_world_object():
    entity = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    assert entity.entity_id == "device:kraken"
    assert entity.entity_type == "device"
    assert entity.name == "Kraken"


def test_fact_preserves_provenance_and_status():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
        source_id="interaction-123",
        timestamp="2026-09-23T14:00:00-05:00",
    )

    fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    assert fact.subject_id == "device:kraken"
    assert fact.predicate == "model"
    assert fact.value == "Elegoo Neptune 4"
    assert fact.source == source
    assert fact.status == "current"
    assert fact.supersedes is None


def test_fact_can_record_supersession_without_erasing_history():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    corrected_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4 Pro",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
        supersedes="fact:kraken:model:1",
    )

    assert corrected_fact.status == "current"
    assert corrected_fact.supersedes == "fact:kraken:model:1"


def test_relationship_connects_two_entities():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    assert relationship.source_entity_id == "device:kraken"
    assert relationship.relationship == "belongs_to"
    assert relationship.target_entity_id == "person:flloyd"
    assert relationship.status == "current"


def test_entity_rejects_blank_identity_fields():
    with pytest.raises(ValueError, match="entity_id cannot be blank"):
        Entity(
            entity_id="   ",
            entity_type="device",
            name="Kraken",
        )

    with pytest.raises(ValueError, match="entity_type cannot be blank"):
        Entity(
            entity_id="device:kraken",
            entity_type="   ",
            name="Kraken",
        )

    with pytest.raises(ValueError, match="name cannot be blank"):
        Entity(
            entity_id="device:kraken",
            entity_type="device",
            name="   ",
        )


def test_fact_rejects_blank_identity_fields():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    with pytest.raises(ValueError, match="fact_id cannot be blank"):
        Fact(
            fact_id="   ",
            subject_id="device:kraken",
            predicate="model",
            value="Elegoo Neptune 4",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
        )

    with pytest.raises(ValueError, match="subject_id cannot be blank"):
        Fact(
            fact_id="fact:kraken:model:1",
            subject_id="   ",
            predicate="model",
            value="Elegoo Neptune 4",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
        )

    with pytest.raises(ValueError, match="predicate cannot be blank"):
        Fact(
            fact_id="fact:kraken:model:1",
            subject_id="device:kraken",
            predicate="   ",
            value="Elegoo Neptune 4",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
        )


def test_relationship_rejects_blank_identity_fields():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    with pytest.raises(ValueError):
        Relationship(
            relationship_id="relationship:kraken:owner:flloyd",
            source_entity_id="   ",
            relationship="belongs_to",
            target_entity_id="person:flloyd",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
        )


def test_fact_rejects_unknown_status():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    with pytest.raises(ValueError, match="Invalid fact status"):
        Fact(
            fact_id="fact:kraken:model:1",
            subject_id="device:kraken",
            predicate="model",
            value="Elegoo Neptune 4",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
            status="banana",
        )


def test_relationship_rejects_unknown_status():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    with pytest.raises(ValueError, match="Invalid relationship status"):
        Relationship(
            relationship_id="relationship:kraken:owner:flloyd",
            source_entity_id="device:kraken",
            relationship="belongs_to",
            target_entity_id="person:flloyd",
            source=source,
            learned_at="2026-09-23T14:00:00-05:00",
            status="banana",
        )

def test_supersede_fact_preserves_old_fact_as_history():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    new_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4 Pro",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    historical, current = supersede_fact(
        old_fact,
        new_fact,
    )

    assert historical.fact_id == old_fact.fact_id
    assert historical.value == "Elegoo Neptune 4"
    assert historical.status == "superseded"

    assert current.fact_id == new_fact.fact_id
    assert current.value == "Elegoo Neptune 4 Pro"
    assert current.status == "current"
    assert current.supersedes == old_fact.fact_id


def test_supersede_fact_requires_same_subject_and_predicate():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    unrelated_fact = Fact(
        fact_id="fact:ferris:model:1",
        subject_id="device:ferris",
        predicate="model",
        value="Ender-3 S1 Pro",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    with pytest.raises(
        ValueError,
        match="same subject and predicate",
    ):
        supersede_fact(
            old_fact,
            unrelated_fact,
        )


def test_supersede_fact_rejects_already_superseded_fact():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
        status="superseded",
    )

    new_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4 Pro",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    with pytest.raises(
        ValueError,
        match="current fact",
    ):
        supersede_fact(
            old_fact,
            new_fact,
        )

def test_supersede_relationship_preserves_old_relationship_as_history():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_relationship = Relationship(
        relationship_id="relationship:kraken:location:office",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    new_relationship = Relationship(
        relationship_id="relationship:kraken:location:workshop",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:workshop",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    historical, current = supersede_relationship(
        old_relationship,
        new_relationship,
    )

    assert historical.relationship_id == old_relationship.relationship_id
    assert historical.target_entity_id == "place:office"
    assert historical.status == "superseded"

    assert current.relationship_id == new_relationship.relationship_id
    assert current.target_entity_id == "place:workshop"
    assert current.status == "current"
    assert current.supersedes == old_relationship.relationship_id


def test_supersede_relationship_requires_same_source_and_relationship():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_relationship = Relationship(
        relationship_id="relationship:kraken:location:office",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    unrelated_relationship = Relationship(
        relationship_id="relationship:ferris:location:workshop",
        source_entity_id="device:ferris",
        relationship="located_in",
        target_entity_id="place:workshop",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    with pytest.raises(
        ValueError,
        match="same source entity and relationship",
    ):
        supersede_relationship(
            old_relationship,
            unrelated_relationship,
        )


def test_supersede_relationship_rejects_already_superseded_relationship():
    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_relationship = Relationship(
        relationship_id="relationship:kraken:location:office",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:office",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
        status="superseded",
    )

    new_relationship = Relationship(
        relationship_id="relationship:kraken:location:workshop",
        source_entity_id="device:kraken",
        relationship="located_in",
        target_entity_id="place:workshop",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    with pytest.raises(
        ValueError,
        match="current relationship",
    ):
        supersede_relationship(
            old_relationship,
            new_relationship,
        )