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

def test_find_entities_by_name(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )
    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=store,
    )

    kraken = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )

    manager.save_entity(kraken)

    assert manager.find_entities_by_name("Kraken") == [kraken]
    assert manager.find_entities_by_name("kraken") == [kraken]
    assert manager.find_entities_by_name("Unknown") == []

    store.close()

def test_list_entities(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )
    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=store,
    )

    kraken = Entity(
        entity_id="device:kraken",
        entity_type="device",
        name="Kraken",
    )
    ferris = Entity(
        entity_id="device:ferris",
        entity_type="device",
        name="Ferris",
    )

    manager.save_entity(kraken)
    manager.save_entity(ferris)

    assert manager.list_entities() == [
        ferris,
        kraken,
    ]

    store.close()

def test_get_structured_context_for_named_entity(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )
    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=store,
    )

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

    model_fact = Fact(
        fact_id="fact:aurora:model:1",
        subject_id="device:aurora-printer",
        predicate="model",
        value="Test Model X9",
        source=source,
        learned_at="2026-09-23T17:00:00-05:00",
    )

    manager.save_entity(aurora)
    manager.save_fact(model_fact)

    context = manager.get_entity_context(
        "What model is Aurora?"
    )

    assert "Aurora" in context
    assert "model" in context
    assert "Test Model X9" in context
    assert "user_statement" in context
    assert "user" in context

    store.close()

def test_get_entity_context_uses_only_current_fact(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )
    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=store,
    )

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

    old_fact = Fact(
        fact_id="fact:aurora:model:1",
        subject_id="device:aurora-printer",
        predicate="model",
        value="Old Model",
        source=source,
        learned_at="2026-09-23T17:00:00-05:00",
    )

    new_fact = Fact(
        fact_id="fact:aurora:model:2",
        subject_id="device:aurora-printer",
        predicate="model",
        value="Test Model X9",
        source=source,
        learned_at="2026-09-23T17:10:00-05:00",
    )

    manager.save_entity(aurora)
    manager.save_fact(old_fact)
    manager.supersede_fact(
        old_fact.fact_id,
        new_fact,
    )

    context = manager.get_entity_context(
        "What model is Aurora?"
    )

    assert "Test Model X9" in context
    assert "Old Model" not in context

    store.close()

def test_unified_context_includes_structured_memory(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )
    manager = MemoryManager(
        vector_store=None,
        episodic_memory=None,
        structured_store=store,
    )

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

    fact = Fact(
        fact_id="fact:aurora:model:1",
        subject_id="device:aurora-printer",
        predicate="model",
        value="Test Model X9",
        source=source,
        learned_at="2026-09-23T17:00:00-05:00",
    )

    manager.save_entity(aurora)
    manager.save_fact(fact)

    context = manager.get_context(
        "What model is Aurora?"
    )

    assert "structured" in context
    assert "Aurora" in context["structured"]
    assert "Test Model X9" in context["structured"]

    store.close()