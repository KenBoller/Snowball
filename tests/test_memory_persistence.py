from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
    ProposedRelationship,
)
from core.memory.manager import MemoryManager
from core.memory.resolution import resolve_extraction
from core.memory.structured_store import StructuredMemoryStore
from core.memory.persistence import (
    build_entity,
    build_fact,
    build_relationship,
    conversational_memory_source,
    persist_resolved_extraction,
)


def test_conversational_source_has_trusted_provenance():
    source = conversational_memory_source(
        source_id="turn-123",
        timestamp="2026-09-24T18:00:00+00:00",
    )

    assert source.memory_type == "structured"
    assert source.source_type == "user_statement"
    assert source.authority == "user"
    assert source.source_id == "turn-123"
    assert source.timestamp == (
        "2026-09-24T18:00:00+00:00"
    )


def test_build_entity_generates_permanent_id():
    entity = build_entity(
        entity_type="device",
        name="Kraken",
    )

    assert entity.entity_id.startswith("entity_")
    assert not entity.entity_id.startswith("proposal:")


def test_build_fact_generates_trusted_defaults():
    source = conversational_memory_source()

    fact = build_fact(
        subject_id="entity_kraken",
        predicate="nozzle_size",
        value="0.4 mm",
        source=source,
        learned_at="2026-09-24T18:00:00+00:00",
    )

    assert fact.fact_id.startswith("fact_")
    assert fact.subject_id == "entity_kraken"
    assert fact.status == "current"
    assert fact.supersedes is None
    assert fact.source is source


def test_build_relationship_generates_trusted_defaults():
    source = conversational_memory_source()

    relationship = build_relationship(
        source_entity_id="entity_user",
        relationship="girlfriend",
        target_entity_id="entity_jess",
        source=source,
        learned_at="2026-09-24T18:00:00+00:00",
    )

    assert relationship.relationship_id.startswith(
        "relationship_"
    )
    assert relationship.status == "current"
    assert relationship.supersedes is None
    assert relationship.source is source


def test_generated_ids_are_unique():
    first = build_entity(
        entity_type="device",
        name="Kraken",
    )
    second = build_entity(
        entity_type="device",
        name="Ferris",
    )

    assert first.entity_id != second.entity_id


class UnusedVectorStore:
    pass


def build_test_manager(tmp_path):
    structured_store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )

    manager = MemoryManager(
        vector_store=UnusedVectorStore(),
        structured_store=structured_store,
    )

    return manager, structured_store


def test_persist_resolved_extraction_creates_entity_and_fact(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        extraction = MemoryExtraction(
            entities=(
                ProposedEntity(
                    reference="kraken",
                    entity_type="device",
                    name="Kraken",
                ),
            ),
            facts=(
                ProposedFact(
                    subject_reference="kraken",
                    predicate="model",
                    value="Neptune 4",
                ),
            ),
        )

        resolved = resolve_extraction(
            extraction,
            existing_entities=(),
            facts_by_entity={},
            relationships_by_entity={},
        )

        result = persist_resolved_extraction(
            resolved,
            manager,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
        )

        assert len(result.entities_created) == 1
        assert len(result.facts_created) == 1

        entity = result.entities_created[0]
        fact = result.facts_created[0]

        assert entity.name == "Kraken"
        assert entity.entity_type == "device"
        assert entity.entity_id.startswith("entity_")
        assert not entity.entity_id.startswith("proposal:")

        assert fact.subject_id == entity.entity_id
        assert fact.predicate == "model"
        assert fact.value == "Neptune 4"

        stored_entity = manager.get_entity(
            entity.entity_id
        )
        stored_fact = manager.get_fact(
            fact.fact_id
        )

        assert stored_entity == entity
        assert stored_fact == fact

        assert stored_fact.source.memory_type == "structured"
        assert stored_fact.source.source_type == "user_statement"
        assert stored_fact.source.authority == "user"
        assert stored_fact.source.source_id == "test-message-1"

    finally:
        store.close()


def test_persist_resolved_extraction_supersedes_fact(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        # First memory:
        # Kraken has a 0.4 mm nozzle.
        first_extraction = MemoryExtraction(
            entities=(
                ProposedEntity(
                    reference="kraken",
                    entity_type="device",
                    name="Kraken",
                ),
            ),
            facts=(
                ProposedFact(
                    subject_reference="kraken",
                    predicate="nozzle_size",
                    value="0.4 mm",
                ),
            ),
        )

        first_resolved = resolve_extraction(
            first_extraction,
            existing_entities=(),
            facts_by_entity={},
            relationships_by_entity={},
        )

        first_result = persist_resolved_extraction(
            first_resolved,
            manager,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
        )

        kraken = first_result.entities_created[0]
        old_fact = first_result.facts_created[0]

        # Correction:
        # Kraken now has a 0.6 mm nozzle.
        second_extraction = MemoryExtraction(
            entities=(
                ProposedEntity(
                    reference="kraken",
                    entity_type="device",
                    name="Kraken",
                ),
            ),
            facts=(
                ProposedFact(
                    subject_reference="kraken",
                    predicate="nozzle_size",
                    value="0.6 mm",
                ),
            ),
        )

        second_resolved = resolve_extraction(
            second_extraction,
            existing_entities=tuple(
                manager.list_entities()
            ),
            facts_by_entity={
                kraken.entity_id: tuple(
                    manager.get_current_facts(
                        kraken.entity_id
                    )
                )
            },
            relationships_by_entity={
                kraken.entity_id: tuple(
                    manager.get_current_relationships(
                        kraken.entity_id
                    )
                )
            },
        )

        second_result = persist_resolved_extraction(
            second_resolved,
            manager,
            source_id="test-message-2",
            timestamp="2026-09-24T18:05:00+00:00",
        )

        assert second_result.entities_created == ()
        assert second_result.facts_created == ()
        assert len(
            second_result.facts_superseded
        ) == 1

        new_fact = second_result.facts_superseded[0]

        # There should be exactly one current nozzle fact.
        current_facts = manager.get_current_facts(
            kraken.entity_id,
            predicate="nozzle_size",
        )

        assert len(current_facts) == 1
        assert current_facts[0].fact_id == new_fact.fact_id
        assert current_facts[0].value == "0.6 mm"
        assert current_facts[0].status == "current"
        assert current_facts[0].supersedes == old_fact.fact_id

        # The original record must still exist as history.
        historical = manager.get_fact(
            old_fact.fact_id
        )

        assert historical is not None
        assert historical.value == "0.4 mm"
        assert historical.status == "superseded"

        # The correction has its own provenance.
        assert new_fact.source.source_id == "test-message-2"
        assert new_fact.source.memory_type == "structured"
        assert new_fact.source.source_type == "user_statement"
        assert new_fact.source.authority == "user"

    finally:
        store.close()


def test_persist_resolved_extraction_supersedes_relationship(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        first_extraction = MemoryExtraction(
            entities=(
                ProposedEntity(
                    reference="user",
                    entity_type="person",
                    name="User",
                ),
                ProposedEntity(
                    reference="jess",
                    entity_type="person",
                    name="Jess",
                ),
            ),
            relationships=(
                ProposedRelationship(
                    source_reference="user",
                    relationship="partner",
                    target_reference="jess",
                ),
            ),
        )

        first_resolved = resolve_extraction(
            first_extraction,
            existing_entities=(),
            facts_by_entity={},
            relationships_by_entity={},
        )

        first_result = persist_resolved_extraction(
            first_resolved,
            manager,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
        )

        assert len(first_result.entities_created) == 2
        assert len(
            first_result.relationships_created
        ) == 1

        user = next(
            entity
            for entity in first_result.entities_created
            if entity.name == "User"
        )
        jess = next(
            entity
            for entity in first_result.entities_created
            if entity.name == "Jess"
        )

        old_relationship = (
            first_result.relationships_created[0]
        )

        # Same relationship slot, different target.
        second_extraction = MemoryExtraction(
            entities=(
                ProposedEntity(
                    reference="user",
                    entity_type="person",
                    name="User",
                ),
                ProposedEntity(
                    reference="alex",
                    entity_type="person",
                    name="Alex",
                ),
            ),
            relationships=(
                ProposedRelationship(
                    source_reference="user",
                    relationship="partner",
                    target_reference="alex",
                ),
            ),
        )

        second_resolved = resolve_extraction(
            second_extraction,
            existing_entities=tuple(
                manager.list_entities()
            ),
            facts_by_entity={
                user.entity_id: tuple(
                    manager.get_current_facts(
                        user.entity_id
                    )
                )
            },
            relationships_by_entity={
                user.entity_id: tuple(
                    manager.get_current_relationships(
                        user.entity_id
                    )
                )
            },
        )

        second_result = persist_resolved_extraction(
            second_resolved,
            manager,
            source_id="test-message-2",
            timestamp="2026-09-24T18:05:00+00:00",
        )

        assert len(second_result.entities_created) == 1
        assert (
            second_result.entities_created[0].name
            == "Alex"
        )
        assert len(
            second_result.relationships_superseded
        ) == 1

        new_relationship = (
            second_result.relationships_superseded[0]
        )
        alex = second_result.entities_created[0]

        current = manager.get_current_relationships(
            user.entity_id,
            relationship="partner",
        )

        assert len(current) == 1
        assert (
            current[0].relationship_id
            == new_relationship.relationship_id
        )
        assert current[0].target_entity_id == alex.entity_id
        assert (
            current[0].supersedes
            == old_relationship.relationship_id
        )

        historical = manager.get_relationship(
            old_relationship.relationship_id
        )

        assert historical is not None
        assert historical.target_entity_id == jess.entity_id
        assert historical.status == "superseded"

        assert (
            new_relationship.source.source_id
            == "test-message-2"
        )
        assert (
            new_relationship.source.memory_type
            == "structured"
        )
        assert (
            new_relationship.source.source_type
            == "user_statement"
        )
        assert new_relationship.source.authority == "user"

    finally:
        store.close()