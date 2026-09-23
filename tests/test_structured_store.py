from core.memory.structured import Entity
from core.memory.structured_store import StructuredMemoryStore
from core.memory.semantic import MemorySource
from core.memory.structured import Entity, Fact, Relationship

import sqlite3

import pytest


def test_entity_survives_store_restart(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    entity = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
        metadata={
            "manufacturer": "Elegoo",
        },
    )

    store = StructuredMemoryStore(database_path)
    store.save_entity(entity)
    store.close()

    reopened_store = StructuredMemoryStore(database_path)
    loaded = reopened_store.get_entity("device:kraken")
    reopened_store.close()

    assert loaded == entity


def test_get_entity_returns_none_when_missing(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    store = StructuredMemoryStore(database_path)

    loaded = store.get_entity("device:does-not-exist")

    store.close()

    assert loaded is None

def test_fact_survives_store_restart(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
        source_id="interaction-123",
        timestamp="2026-09-23T14:00:00-05:00",
        metadata={
            "channel": "chat",
        },
    )

    fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
        metadata={
            "confirmed": True,
        },
    )

    store = StructuredMemoryStore(database_path)
    store.save_fact(fact)
    store.close()

    reopened_store = StructuredMemoryStore(database_path)
    loaded = reopened_store.get_fact("fact:kraken:model:1")
    reopened_store.close()

    assert loaded == fact


def test_fact_value_can_preserve_structured_json(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    fact = Fact(
        fact_id="fact:flloyd:preferred-wow-races:1",
        subject_id="person:flloyd",
        predicate="prefers_wow_races",
        value=["Gnome", "Undead"],
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    store = StructuredMemoryStore(database_path)
    store.save_fact(fact)
    store.close()

    reopened_store = StructuredMemoryStore(database_path)
    loaded = reopened_store.get_fact(
        "fact:flloyd:preferred-wow-races:1"
    )
    reopened_store.close()

    assert loaded == fact
    assert loaded.value == ["Gnome", "Undead"]


def test_get_fact_returns_none_when_missing(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    store = StructuredMemoryStore(database_path)

    loaded = store.get_fact("fact:does-not-exist")

    store.close()

    assert loaded is None

def test_relationship_survives_store_restart(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
        source_id="interaction-456",
        timestamp="2026-09-23T14:00:00-05:00",
        metadata={
            "channel": "chat",
        },
    )

    relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
        metadata={
            "confirmed": True,
        },
    )

    store = StructuredMemoryStore(database_path)
    store.save_relationship(relationship)
    store.close()

    reopened_store = StructuredMemoryStore(database_path)
    loaded = reopened_store.get_relationship(
        "relationship:kraken:owner:flloyd"
    )
    reopened_store.close()

    assert loaded == relationship


def test_get_relationship_returns_none_when_missing(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    store = StructuredMemoryStore(database_path)

    loaded = store.get_relationship(
        "relationship:does-not-exist"
    )

    store.close()

    assert loaded is None

def test_supersede_fact_preserves_history_and_current_fact(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Neptune 3",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    corrected_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    store = StructuredMemoryStore(database_path)
    store.save_fact(old_fact)

    store.supersede_fact(
        old_fact.fact_id,
        corrected_fact,
    )

    store.close()

    reopened_store = StructuredMemoryStore(database_path)

    historical = reopened_store.get_fact(old_fact.fact_id)
    current = reopened_store.get_fact(corrected_fact.fact_id)

    reopened_store.close()

    assert historical is not None
    assert historical.status == "superseded"
    assert historical.value == "Neptune 3"

    assert current is not None
    assert current.status == "current"
    assert current.value == "Elegoo Neptune 4"
    assert current.supersedes == old_fact.fact_id


def test_supersede_fact_rolls_back_if_new_fact_cannot_be_inserted(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Neptune 3",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    existing_collision = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Collision",
        source=source,
        learned_at="2026-09-23T13:30:00-05:00",
    )

    corrected_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    store = StructuredMemoryStore(database_path)

    store.save_fact(old_fact)
    store.save_fact(existing_collision)

    with pytest.raises(sqlite3.IntegrityError):
        store.supersede_fact(
            old_fact.fact_id,
            corrected_fact,
        )

    unchanged_old = store.get_fact(old_fact.fact_id)
    unchanged_collision = store.get_fact(
        existing_collision.fact_id
    )

    store.close()

    assert unchanged_old is not None
    assert unchanged_old.status == "current"
    assert unchanged_old.supersedes is None

    assert unchanged_collision == existing_collision

def test_supersede_relationship_preserves_history_and_current_relationship(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_relationship = Relationship(
        relationship_id="relationship:kraken:owner:old",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:old-owner",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    corrected_relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    store = StructuredMemoryStore(database_path)
    store.save_relationship(old_relationship)

    store.supersede_relationship(
        old_relationship.relationship_id,
        corrected_relationship,
    )

    store.close()

    reopened_store = StructuredMemoryStore(database_path)

    historical = reopened_store.get_relationship(
        old_relationship.relationship_id
    )
    current = reopened_store.get_relationship(
        corrected_relationship.relationship_id
    )

    reopened_store.close()

    assert historical is not None
    assert historical.status == "superseded"
    assert historical.target_entity_id == "person:old-owner"

    assert current is not None
    assert current.status == "current"
    assert current.target_entity_id == "person:flloyd"
    assert current.supersedes == old_relationship.relationship_id


def test_supersede_relationship_rolls_back_if_new_relationship_cannot_be_inserted(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_relationship = Relationship(
        relationship_id="relationship:kraken:owner:old",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:old-owner",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    existing_collision = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:collision",
        source=source,
        learned_at="2026-09-23T13:30:00-05:00",
    )

    corrected_relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    store = StructuredMemoryStore(database_path)

    store.save_relationship(old_relationship)
    store.save_relationship(existing_collision)

    with pytest.raises(sqlite3.IntegrityError):
        store.supersede_relationship(
            old_relationship.relationship_id,
            corrected_relationship,
        )

    unchanged_old = store.get_relationship(
        old_relationship.relationship_id
    )
    unchanged_collision = store.get_relationship(
        existing_collision.relationship_id
    )

    store.close()

    assert unchanged_old is not None
    assert unchanged_old.status == "current"
    assert unchanged_old.supersedes is None

    assert unchanged_collision == existing_collision

def test_get_current_facts_returns_only_current_facts_for_subject(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_model = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Neptune 3",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    corrected_model = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    nozzle = Fact(
        fact_id="fact:kraken:nozzle:1",
        subject_id="device:kraken",
        predicate="nozzle_size_mm",
        value=0.4,
        source=source,
        learned_at="2026-09-23T14:05:00-05:00",
    )

    ferris_model = Fact(
        fact_id="fact:ferris:model:1",
        subject_id="device:ferris",
        predicate="model",
        value="Ender-3 S1 Pro",
        source=source,
        learned_at="2026-09-23T14:10:00-05:00",
    )

    store = StructuredMemoryStore(database_path)

    store.save_fact(old_model)
    store.supersede_fact(
        old_model.fact_id,
        corrected_model,
    )
    store.save_fact(nozzle)
    store.save_fact(ferris_model)

    current = store.get_current_facts("device:kraken")

    store.close()

    assert current == [
        corrected_model.__class__(
            fact_id=corrected_model.fact_id,
            subject_id=corrected_model.subject_id,
            predicate=corrected_model.predicate,
            value=corrected_model.value,
            source=corrected_model.source,
            learned_at=corrected_model.learned_at,
            status="current",
            supersedes=old_model.fact_id,
            metadata=corrected_model.metadata,
        ),
        nozzle,
    ]


def test_get_current_facts_can_filter_by_predicate(tmp_path):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    model = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    nozzle = Fact(
        fact_id="fact:kraken:nozzle:1",
        subject_id="device:kraken",
        predicate="nozzle_size_mm",
        value=0.4,
        source=source,
        learned_at="2026-09-23T14:05:00-05:00",
    )

    store = StructuredMemoryStore(database_path)
    store.save_fact(model)
    store.save_fact(nozzle)

    current = store.get_current_facts(
        "device:kraken",
        predicate="model",
    )

    store.close()

    assert current == [model]


def test_get_current_facts_returns_empty_list_when_none_exist(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    store = StructuredMemoryStore(database_path)

    current = store.get_current_facts("device:unknown")

    store.close()

    assert current == []

def test_get_current_relationships_returns_only_current_for_source(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    old_owner = Relationship(
        relationship_id="relationship:kraken:owner:old",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:old-owner",
        source=source,
        learned_at="2026-09-23T13:00:00-05:00",
    )

    corrected_owner = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    project_relationship = Relationship(
        relationship_id="relationship:kraken:used-for:snowball",
        source_entity_id="device:kraken",
        relationship="used_for",
        target_entity_id="project:snowball",
        source=source,
        learned_at="2026-09-23T14:05:00-05:00",
    )

    ferris_owner = Relationship(
        relationship_id="relationship:ferris:owner:flloyd",
        source_entity_id="device:ferris",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:10:00-05:00",
    )

    store = StructuredMemoryStore(database_path)

    store.save_relationship(old_owner)
    store.supersede_relationship(
        old_owner.relationship_id,
        corrected_owner,
    )
    store.save_relationship(project_relationship)
    store.save_relationship(ferris_owner)

    current = store.get_current_relationships(
        "device:kraken"
    )

    store.close()

    assert len(current) == 2

    assert current[0].relationship_id == (
        corrected_owner.relationship_id
    )
    assert current[0].status == "current"
    assert current[0].supersedes == old_owner.relationship_id

    assert current[1] == project_relationship


def test_get_current_relationships_can_filter_by_relationship(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    owner = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T14:00:00-05:00",
    )

    project_relationship = Relationship(
        relationship_id="relationship:kraken:used-for:snowball",
        source_entity_id="device:kraken",
        relationship="used_for",
        target_entity_id="project:snowball",
        source=source,
        learned_at="2026-09-23T14:05:00-05:00",
    )

    store = StructuredMemoryStore(database_path)
    store.save_relationship(owner)
    store.save_relationship(project_relationship)

    current = store.get_current_relationships(
        "device:kraken",
        relationship="belongs_to",
    )

    store.close()

    assert current == [owner]


def test_get_current_relationships_returns_empty_list_when_none_exist(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    store = StructuredMemoryStore(database_path)

    current = store.get_current_relationships(
        "device:unknown"
    )

    store.close()

    assert current == []