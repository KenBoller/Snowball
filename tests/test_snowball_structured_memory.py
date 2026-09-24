from core.ai.chat import SnowballAI
from core.memory.semantic import MemorySource
from core.memory.structured import Entity, Fact, Relationship


def test_snowball_structured_memory_survives_restart(tmp_path):
    storage_dir = tmp_path / "storage"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    entity = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    first = SnowballAI(
        storage_dir=str(storage_dir),
    )

    assert first.memory_manager is not None
    assert first.memory_manager.structured_store is not None

    first.memory_manager.save_entity(entity)
    first.memory_manager.save_fact(fact)

    first.memory_manager.structured_store.close()

    second = SnowballAI(
        storage_dir=str(storage_dir),
    )

    assert second.memory_manager is not None
    assert second.memory_manager.get_entity(
        "device:kraken"
    ) == entity

    assert second.memory_manager.get_current_facts(
        "device:kraken",
        predicate="model",
    ) == [fact]

    second.memory_manager.structured_store.close()

def test_structured_memory_reaches_live_prompt(tmp_path):
    storage_dir = tmp_path / "storage"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    entity = Entity(
        entity_id="device:aurora-printer",
        entity_type="device",
        name="Aurora",
    )

    fact = Fact(
        fact_id="fact:aurora:model:1",
        subject_id="device:aurora-printer",
        predicate="model",
        value="Test Model X9",
        source=source,
        learned_at="2026-09-23T17:00:00-05:00",
    )

    snowball = SnowballAI(storage_dir=str(storage_dir))

    snowball.memory_manager.save_entity(entity)
    snowball.memory_manager.save_fact(fact)

    messages = snowball._build_messages(
        "What model is Aurora?"
    )

    system_message = messages[0]["content"]

    assert "Aurora" in system_message
    assert "Test Model X9" in system_message
    assert "user_statement" in system_message
    assert "AUTHORITY: user" in system_message

    snowball.memory_manager.structured_store.close()

def test_structured_relationship_reaches_live_prompt(tmp_path):
    storage_dir = tmp_path / "storage"

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    aurora = Entity(
        entity_id="device:aurora-printer",
        entity_type="device",
        name="Aurora",
    )

    technician = Entity(
        entity_id="person:test-technician",
        entity_type="person",
        name="Test Technician",
    )

    relationship = Relationship(
        relationship_id="relationship:aurora:maintained-by:technician",
        source_entity_id="device:aurora-printer",
        relationship="maintained_by",
        target_entity_id="person:test-technician",
        source=source,
        learned_at="2026-09-24T11:00:00-05:00",
    )

    snowball = SnowballAI(
        storage_dir=str(storage_dir)
    )

    snowball.memory_manager.save_entity(aurora)
    snowball.memory_manager.save_entity(technician)
    snowball.memory_manager.save_relationship(
        relationship
    )

    messages = snowball._build_messages(
        "Who maintains Aurora?"
    )

    system_message = messages[0]["content"]

    assert "ENTITY: Aurora" in system_message
    assert (
        "RELATIONSHIP: maintained_by -> Test Technician"
        in system_message
    )
    assert (
        "TARGET_ENTITY_ID: person:test-technician"
        in system_message
    )
    assert "SOURCE_TYPE: user_statement" in system_message
    assert "AUTHORITY: user" in system_message

    snowball.memory_manager.structured_store.close()