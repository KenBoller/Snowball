from core.memory.extraction import (
    MemoryExtraction,
    ProposedEntity,
    ProposedFact,
)
from core.memory.learning import learn_from_message
from core.memory.manager import MemoryManager
from core.memory.structured_store import StructuredMemoryStore


class UnusedVectorStore:
    pass


def build_test_manager(tmp_path):
    store = StructuredMemoryStore(
        tmp_path / "structured_memory.db"
    )

    manager = MemoryManager(
        vector_store=UnusedVectorStore(),
        structured_store=store,
    )

    return manager, store


def kraken_extraction(_message):
    return MemoryExtraction(
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


def test_learning_is_dry_run_by_default(tmp_path):
    manager, store = build_test_manager(tmp_path)

    try:
        result = learn_from_message(
            "My Neptune 4 printer is named Kraken.",
            manager,
            extract_fn=kraken_extraction,
        )

        assert result.persisted is False
        assert result.persistence is None

        assert len(
            result.resolved.resolution.creates
        ) == 2

        # Most important invariant:
        # dry-run learning must not mutate storage.
        assert manager.list_entities() == []

    finally:
        store.close()


def test_learning_persists_only_when_authorized(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        result = learn_from_message(
            "My Neptune 4 printer is named Kraken.",
            manager,
            persist=True,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
            extract_fn=kraken_extraction,
        )

        assert result.persisted is True
        assert result.persistence is not None

        entities = manager.list_entities()

        assert len(entities) == 1
        assert entities[0].name == "Kraken"

        facts = manager.get_current_facts(
            entities[0].entity_id
        )

        assert len(facts) == 1
        assert facts[0].predicate == "model"
        assert facts[0].value == "Neptune 4"

        assert (
            facts[0].source.memory_type
            == "structured"
        )
        assert (
            facts[0].source.source_type
            == "user_statement"
        )
        assert facts[0].source.authority == "user"

    finally:
        store.close()


def test_learning_no_op_does_not_duplicate_memory(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        first = learn_from_message(
            "My Neptune 4 printer is named Kraken.",
            manager,
            persist=True,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
            extract_fn=kraken_extraction,
        )

        assert first.persistence is not None
        assert len(first.persistence.entities_created) == 1
        assert len(first.persistence.facts_created) == 1

        second = learn_from_message(
            "My Neptune 4 printer is named Kraken.",
            manager,
            persist=True,
            source_id="test-message-2",
            timestamp="2026-09-24T18:05:00+00:00",
            extract_fn=kraken_extraction,
        )

        assert second.persistence is not None

        assert second.persistence.entities_created == ()
        assert second.persistence.facts_created == ()
        assert second.persistence.facts_superseded == ()

        entities = manager.list_entities()
        assert len(entities) == 1

        facts = manager.get_current_facts(
            entities[0].entity_id
        )
        assert len(facts) == 1

    finally:
        store.close()


def test_learning_persists_fact_correction(
    tmp_path,
):
    manager, store = build_test_manager(tmp_path)

    try:
        learn_from_message(
            "Kraken has a 0.4 mm nozzle.",
            manager,
            persist=True,
            source_id="test-message-1",
            timestamp="2026-09-24T18:00:00+00:00",
            extract_fn=lambda _message: MemoryExtraction(
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
            ),
        )

        result = learn_from_message(
            "Kraken has a 0.6 mm nozzle.",
            manager,
            persist=True,
            source_id="test-message-2",
            timestamp="2026-09-24T18:05:00+00:00",
            extract_fn=lambda _message: MemoryExtraction(
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
            ),
        )

        assert result.persistence is not None
        assert len(result.persistence.facts_superseded) == 1

        kraken = manager.list_entities()[0]

        current = manager.get_current_facts(
            kraken.entity_id,
            predicate="nozzle_size",
        )

        assert len(current) == 1
        assert current[0].value == "0.6 mm"
        assert current[0].status == "current"

        old_fact_id = current[0].supersedes
        assert old_fact_id is not None

        historical = manager.get_fact(old_fact_id)

        assert historical is not None
        assert historical.value == "0.4 mm"
        assert historical.status == "superseded"

    finally:
        store.close()