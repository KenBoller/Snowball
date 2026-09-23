from core.memory.manager import MemoryManager
from core.memory.structured import Entity
from core.memory.structured_store import StructuredMemoryStore
from core.memory.semantic import MemorySource
from core.memory.structured import Entity, Fact, Relationship

def test_memory_manager_can_save_and_load_structured_entity(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"

    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

    entity = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
        metadata={
            "manufacturer": "Elegoo",
        },
    )

    manager.save_entity(entity)

    loaded = manager.get_entity("device:kraken")

    structured_store.close()

    assert loaded == entity

def test_memory_manager_can_save_and_query_current_fact(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_fact(fact)

    current = manager.get_current_facts(
        "device:kraken",
        predicate="model",
    )

    structured_store.close()

    assert current == [fact]


def test_memory_manager_can_save_and_query_current_relationship(
    tmp_path,
):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

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
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_relationship(relationship)

    current = manager.get_current_relationships(
        "device:kraken",
        relationship="belongs_to",
    )

    structured_store.close()

    assert current == [relationship]

def test_memory_manager_can_supersede_fact(tmp_path):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

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
        learned_at="2026-09-23T14:00:00-05:00",
    )

    corrected_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_fact(old_fact)

    manager.supersede_fact(
        old_fact.fact_id,
        corrected_fact,
    )

    current = manager.get_current_facts(
        "device:kraken",
        predicate="model",
    )

    historical = structured_store.get_fact(
        old_fact.fact_id
    )

    structured_store.close()

    assert historical is not None
    assert historical.status == "superseded"

    assert len(current) == 1
    assert current[0].fact_id == corrected_fact.fact_id
    assert current[0].value == "Elegoo Neptune 4"
    assert current[0].supersedes == old_fact.fact_id


def test_memory_manager_can_supersede_relationship(tmp_path):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

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
        learned_at="2026-09-23T14:00:00-05:00",
    )

    corrected_relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_relationship(old_relationship)

    manager.supersede_relationship(
        old_relationship.relationship_id,
        corrected_relationship,
    )

    current = manager.get_current_relationships(
        "device:kraken",
        relationship="belongs_to",
    )

    historical = structured_store.get_relationship(
        old_relationship.relationship_id
    )

    structured_store.close()

    assert historical is not None
    assert historical.status == "superseded"

    assert len(current) == 1
    assert current[0].relationship_id == (
        corrected_relationship.relationship_id
    )
    assert current[0].target_entity_id == "person:flloyd"
    assert current[0].supersedes == (
        old_relationship.relationship_id
    )

def test_memory_manager_can_get_fact_by_id(tmp_path):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

    source = MemorySource(
        memory_type="structured",
        source_type="user_statement",
        authority="user",
    )

    fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=source,
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_fact(fact)

    loaded = manager.get_fact(fact.fact_id)

    structured_store.close()

    assert loaded == fact


def test_memory_manager_can_get_relationship_by_id(tmp_path):
    database_path = tmp_path / "structured_memory.db"
    structured_store = StructuredMemoryStore(database_path)

    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=structured_store,
    )

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
        learned_at="2026-09-23T15:00:00-05:00",
    )

    manager.save_relationship(relationship)

    loaded = manager.get_relationship(
        relationship.relationship_id
    )

    structured_store.close()

    assert loaded == relationship