from core.memory.semantic import (
    MemorySource,
    knowledge_source,
    user_statement_source,
)


def test_user_statement_source():
    source = user_statement_source(
        source_id="interaction-123",
        timestamp="2026-09-22T12:00:00",
    )

    assert isinstance(source, MemorySource)
    assert source.memory_type == "episodic"
    assert source.source_type == "user_statement"
    assert source.authority == "user"
    assert source.source_id == "interaction-123"
    assert source.timestamp == "2026-09-22T12:00:00"


def test_knowledge_source():
    source = knowledge_source(
        source_id="snowball-roadmap",
        metadata={"filename": "roadmap.md"},
    )

    assert source.memory_type == "semantic"
    assert source.source_type == "knowledge_document"
    assert source.authority == "reference"
    assert source.source_id == "snowball-roadmap"
    assert source.metadata == {
        "filename": "roadmap.md",
    }


def test_memory_source_is_immutable():
    source = user_statement_source()

    try:
        source.authority = "something_else"
        changed = True
    except Exception:
        changed = False

    assert changed is False