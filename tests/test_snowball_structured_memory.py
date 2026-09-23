from core.ai.chat import SnowballAI
from core.memory.semantic import MemorySource
from core.memory.structured import Entity, Fact


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