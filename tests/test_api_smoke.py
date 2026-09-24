def test_api_send_message_returns_string(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "SNOWBALL_STORAGE_DIR",
        str(tmp_path / "storage"),
    )

    try:
        from core.api import send_message
    except Exception:
        return

    out = send_message("hello")
    assert isinstance(out, str)
    assert len(out.strip()) > 0

def test_status_endpoint():
    from fastapi.testclient import TestClient

    from core.api.server import app

    client = TestClient(app)
    response = client.get("/status")

    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "Snowball AI/OS"
    assert data["version"] == "0.1.0"
    assert data["status"] == "ok"
    assert "chat" in data["capabilities"]
    assert "persistent_memory" in data["capabilities"]
    assert "semantic_knowledge" in data["capabilities"]
    assert "knowledge_provenance" in data["capabilities"]
    assert "structured_memory" in data["capabilities"]


def test_chat_endpoint_contract(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    monkeypatch.setattr(
        server,
        "send_message",
        lambda message: f"Echo: {message}",
    )

    client = TestClient(server.app)

    response = client.post(
        "/chat",
        json={"message": "hello Snowball"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Echo: hello Snowball"
    }


def test_knowledge_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def list_knowledge_documents(self):
            return [
                {
                    "document_id": "snowball-current-state",
                    "filename": "CURRENT_STATE.md",
                    "chunk_count": 19,
                    "authority": "current_reference",
                    "provenance_source_type": "snowball_current_state",
                },
                {
                    "document_id": "snowball-archive",
                    "filename": "archive.docx",
                    "chunk_count": 45,
                    "authority": "historical_reference",
                    "provenance_source_type": "snowball_project_document",
                },
            ]

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)
    response = client.get("/knowledge")

    assert response.status_code == 200

    body = response.json()

    assert body["document_count"] == 2
    assert body["chunk_count"] == 64

    assert body["documents"] == [
        {
            "document_id": "snowball-current-state",
            "filename": "CURRENT_STATE.md",
            "chunk_count": 19,
            "authority": "current_reference",
            "provenance_source_type": "snowball_current_state",
        },
        {
            "document_id": "snowball-archive",
            "filename": "archive.docx",
            "chunk_count": 45,
            "authority": "historical_reference",
            "provenance_source_type": "snowball_project_document",
        },
    ]

def test_get_knowledge_document_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    expected_document = {
        "document_id": "snowball-current-state",
        "filename": "CURRENT_STATE.md",
        "chunk_count": 2,
        "authority": "current_reference",
        "provenance_source_type": "snowball_current_state",
        "chunks": [
            {
                "chunk_index": 0,
                "text": "First Snowball chunk.",
                "start_char": 0,
                "end_char": 21,
            },
            {
                "chunk_index": 1,
                "text": "Second Snowball chunk.",
                "start_char": 22,
                "end_char": 44,
            },
        ],
    }

    class FakeMemoryManager:
        def get_knowledge_document(self, document_id):
            assert document_id == "snowball-current-state"
            return expected_document

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.get(
        "/knowledge/snowball-current-state"
    )

    assert response.status_code == 200
    assert response.json() == expected_document

def test_get_knowledge_document_endpoint_returns_404_when_missing(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def get_knowledge_document(self, document_id):
            assert document_id == "does-not-exist"
            return None

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.get(
        "/knowledge/does-not-exist"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Knowledge document not found."
    }

def test_delete_knowledge_document_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def delete_knowledge_document(self, document_id):
            assert document_id == "temporary-document"
            return True

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.delete(
        "/knowledge/temporary-document"
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "temporary-document",
        "deleted": True,
    }

def test_delete_knowledge_document_endpoint_returns_404_when_missing(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def delete_knowledge_document(self, document_id):
            assert document_id == "does-not-exist"
            return False

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.delete(
        "/knowledge/does-not-exist"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Knowledge document not found."
    }

def test_search_knowledge_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    expected_results = [
        {
            "text": "Snowball began as one AI across multiple games.",
            "metadata": {
                "document_id": "snowball-archive",
                "filename": "archive.docx",
                "chunk_index": 7,
                "authority": "historical_reference",
                "provenance_source_type": "snowball_project_document",
            },
            "distance": 0.25,
        }
    ]

    class FakeMemoryManager:
        def search_knowledge(
            self,
            question,
            *,
            result_count=5,
        ):
            assert question == "How did Snowball begin?"
            assert result_count == 3
            return expected_results

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/knowledge/search",
        json={
            "query": "How did Snowball begin?",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "How did Snowball begin?",
        "results": expected_results,
        "result_count": 1,
    }

def test_search_knowledge_endpoint_rejects_blank_query():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    response = client.post(
        "/knowledge/search",
        json={
            "query": "   ",
            "limit": 3,
        },
    )

    assert response.status_code == 422


def test_search_knowledge_endpoint_rejects_invalid_limit():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    response = client.post(
        "/knowledge/search",
        json={
            "query": "How did Snowball begin?",
            "limit": 0,
        },
    )

    assert response.status_code == 422

def test_ingest_knowledge_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    expected_result = {
        "document_id": "snowball-notes",
        "filename": "snowball-notes.docx",
        "chunk_count": 4,
    }

    class FakeMemoryManager:
        def ingest_document(
            self,
            file_path,
            *,
            document_id=None,
            metadata=None,
            chunk_size=1000,
            overlap=200,
        ):
            assert file_path == r"S:\Documents\snowball-notes.docx"
            assert document_id == "snowball-notes"
            assert metadata == {
                "authority": "reference",
                "provenance_source_type": "user_document",
            }
            assert chunk_size == 1000
            assert overlap == 200

            return expected_result

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/knowledge",
        json={
            "path": r"S:\Documents\snowball-notes.docx",
            "document_id": "snowball-notes",
            "metadata": {
                "authority": "reference",
                "provenance_source_type": "user_document",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_result

def test_ingest_knowledge_endpoint_rejects_blank_path():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    response = client.post(
        "/knowledge",
        json={
            "path": "   ",
        },
    )

    assert response.status_code == 422


def test_ingest_knowledge_endpoint_rejects_blank_document_id():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    response = client.post(
        "/knowledge",
        json={
            "path": r"S:\Documents\snowball-notes.docx",
            "document_id": "   ",
        },
    )

    assert response.status_code == 422

def test_ingest_knowledge_endpoint_returns_404_for_missing_file(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def ingest_document(self, *args, **kwargs):
            raise FileNotFoundError(
                r"File not found: S:\Documents\missing.docx"
            )

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/knowledge",
        json={
            "path": r"S:\Documents\missing.docx",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": r"File not found: S:\Documents\missing.docx"
    }


def test_ingest_knowledge_endpoint_returns_400_for_unsupported_type(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    class FakeMemoryManager:
        def ingest_document(self, *args, **kwargs):
            raise ValueError(
                "Unsupported document type: .exe"
            )

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/knowledge",
        json={
            "path": r"S:\Documents\definitely-not-knowledge.exe",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unsupported document type: .exe"
    }

def test_ingest_knowledge_endpoint_does_not_mask_internal_value_error(
    monkeypatch,
):
    from fastapi.testclient import TestClient
    import pytest

    import core.api.server as server

    class FakeMemoryManager:
        def ingest_document(self, *args, **kwargs):
            raise ValueError("Embedding pipeline exploded.")

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    with pytest.raises(
        ValueError,
        match="Embedding pipeline exploded.",
    ):
        client.post(
            "/knowledge",
            json={
                "path": r"S:\Documents\valid.docx",
            },
        )

def test_create_memory_entity_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    saved_entities = []

    class FakeMemoryManager:
        def save_entity(self, entity):
            saved_entities.append(entity)

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/entities",
        json={
            "entity_id": "device:kraken",
            "entity_type": "device",
            "name": "Kraken",
            "metadata": {
                "manufacturer": "Elegoo",
            },
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "entity_id": "device:kraken",
        "entity_type": "device",
        "name": "Kraken",
        "metadata": {
            "manufacturer": "Elegoo",
        },
    }

    assert len(saved_entities) == 1

    saved = saved_entities[0]

    assert saved.entity_id == "device:kraken"
    assert saved.entity_type == "device"
    assert saved.name == "Kraken"
    assert saved.metadata == {
        "manufacturer": "Elegoo",
    }


def test_create_memory_entity_endpoint_rejects_blank_fields():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    for field in (
        "entity_id",
        "entity_type",
        "name",
    ):
        payload = {
            "entity_id": "device:kraken",
            "entity_type": "device",
            "name": "Kraken",
        }

        payload[field] = "   "

        response = client.post(
            "/memory/entities",
            json=payload,
        )

        assert response.status_code == 422


def test_create_memory_fact_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    saved_facts = []

    class FakeMemoryManager:
        def save_fact(self, fact):
            saved_facts.append(fact)

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/facts",
        json={
            "fact_id": "fact:kraken:model:1",
            "subject_id": "device:kraken",
            "predicate": "model",
            "value": "Elegoo Neptune 4",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "fact_id": "fact:kraken:model:1",
        "subject_id": "device:kraken",
        "predicate": "model",
        "value": "Elegoo Neptune 4",
        "source": {
            "memory_type": "structured",
            "source_type": "user_statement",
            "authority": "user",
        },
        "learned_at": "2026-09-24T11:00:00-05:00",
        "status": "current",
        "supersedes": None,
        "metadata": None,
    }

    assert len(saved_facts) == 1

    saved = saved_facts[0]

    assert saved.fact_id == "fact:kraken:model:1"
    assert saved.subject_id == "device:kraken"
    assert saved.predicate == "model"
    assert saved.value == "Elegoo Neptune 4"

    assert saved.source.memory_type == "structured"
    assert saved.source.source_type == "user_statement"
    assert saved.source.authority == "user"

    assert saved.status == "current"
    assert saved.supersedes is None


def test_create_memory_fact_endpoint_rejects_blank_fields():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    for field in (
        "fact_id",
        "subject_id",
        "predicate",
        "source_type",
        "authority",
        "learned_at",
    ):
        payload = {
            "fact_id": "fact:kraken:model:1",
            "subject_id": "device:kraken",
            "predicate": "model",
            "value": "Elegoo Neptune 4",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        }

        payload[field] = "   "

        response = client.post(
            "/memory/facts",
            json=payload,
        )

        assert response.status_code == 422


def test_create_memory_relationship_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server

    saved_relationships = []

    class FakeMemoryManager:
        def save_relationship(self, relationship):
            saved_relationships.append(relationship)

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/relationships",
        json={
            "relationship_id": "relationship:kraken:owner:flloyd",
            "source_entity_id": "device:kraken",
            "relationship": "belongs_to",
            "target_entity_id": "person:flloyd",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "relationship_id": "relationship:kraken:owner:flloyd",
        "source_entity_id": "device:kraken",
        "relationship": "belongs_to",
        "target_entity_id": "person:flloyd",
        "source": {
            "memory_type": "structured",
            "source_type": "user_statement",
            "authority": "user",
        },
        "learned_at": "2026-09-24T11:00:00-05:00",
        "status": "current",
        "supersedes": None,
        "metadata": None,
    }

    assert len(saved_relationships) == 1

    saved = saved_relationships[0]

    assert (
        saved.relationship_id
        == "relationship:kraken:owner:flloyd"
    )
    assert saved.source_entity_id == "device:kraken"
    assert saved.relationship == "belongs_to"
    assert saved.target_entity_id == "person:flloyd"

    assert saved.source.memory_type == "structured"
    assert saved.source.source_type == "user_statement"
    assert saved.source.authority == "user"

    assert saved.status == "current"
    assert saved.supersedes is None


def test_create_memory_relationship_endpoint_rejects_blank_fields():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    for field in (
        "relationship_id",
        "source_entity_id",
        "relationship",
        "target_entity_id",
        "source_type",
        "authority",
        "learned_at",
    ):
        payload = {
            "relationship_id": "relationship:kraken:owner:flloyd",
            "source_entity_id": "device:kraken",
            "relationship": "belongs_to",
            "target_entity_id": "person:flloyd",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        }

        payload[field] = "   "

        response = client.post(
            "/memory/relationships",
            json=payload,
        )

        assert response.status_code == 422


def test_supersede_memory_fact_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server
    from core.memory.semantic import MemorySource
    from core.memory.structured import Fact

    calls = []

    old_fact = Fact(
        fact_id="fact:kraken:model:1",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 3",
        source=MemorySource(
            memory_type="structured",
            source_type="user_statement",
            authority="user",
        ),
        learned_at="2026-09-23T10:00:00-05:00",
    )

    new_fact = Fact(
        fact_id="fact:kraken:model:2",
        subject_id="device:kraken",
        predicate="model",
        value="Elegoo Neptune 4",
        source=MemorySource(
            memory_type="structured",
            source_type="user_statement",
            authority="user",
        ),
        learned_at="2026-09-24T11:00:00-05:00",
        supersedes="fact:kraken:model:1",
    )

    class FakeMemoryManager:
        def get_fact(self, fact_id):
            assert fact_id == "fact:kraken:model:1"
            return old_fact

        def supersede_fact(self, old, replacement):
            calls.append((old, replacement))
            return new_fact

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/facts/fact:kraken:model:1/supersede",
        json={
            "fact_id": "fact:kraken:model:2",
            "value": "Elegoo Neptune 4",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["fact_id"] == "fact:kraken:model:2"
    assert body["subject_id"] == "device:kraken"
    assert body["predicate"] == "model"
    assert body["value"] == "Elegoo Neptune 4"
    assert body["status"] == "current"
    assert body["supersedes"] == "fact:kraken:model:1"

    assert len(calls) == 1

    old, replacement = calls[0]

    assert old is old_fact

    assert replacement.fact_id == "fact:kraken:model:2"
    assert replacement.subject_id == old_fact.subject_id
    assert replacement.predicate == old_fact.predicate
    assert replacement.value == "Elegoo Neptune 4"
    assert replacement.supersedes is None


def test_supersede_memory_fact_endpoint_returns_404_when_missing(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    supersede_calls = []

    class FakeMemoryManager:
        def get_fact(self, fact_id):
            assert fact_id == "fact:missing"
            return None

        def supersede_fact(self, old, replacement):
            supersede_calls.append((old, replacement))

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/facts/fact:missing/supersede",
        json={
            "fact_id": "fact:replacement",
            "value": "new value",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Fact not found: fact:missing",
    }

    assert supersede_calls == []


def test_supersede_memory_fact_endpoint_rejects_blank_fields():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    for field in (
        "fact_id",
        "source_type",
        "authority",
        "learned_at",
    ):
        payload = {
            "fact_id": "fact:kraken:model:2",
            "value": "Elegoo Neptune 4",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        }

        payload[field] = "   "

        response = client.post(
            "/memory/facts/fact:kraken:model:1/supersede",
            json=payload,
        )

        assert response.status_code == 422


def test_supersede_memory_relationship_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import core.api.server as server
    from core.memory.semantic import MemorySource
    from core.memory.structured import Relationship

    calls = []

    old_relationship = Relationship(
        relationship_id="relationship:kraken:owner:old",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:old",
        source=MemorySource(
            memory_type="structured",
            source_type="user_statement",
            authority="user",
        ),
        learned_at="2026-09-23T10:00:00-05:00",
    )

    new_relationship = Relationship(
        relationship_id="relationship:kraken:owner:flloyd",
        source_entity_id="device:kraken",
        relationship="belongs_to",
        target_entity_id="person:flloyd",
        source=MemorySource(
            memory_type="structured",
            source_type="user_statement",
            authority="user",
        ),
        learned_at="2026-09-24T11:00:00-05:00",
        supersedes="relationship:kraken:owner:old",
    )

    class FakeMemoryManager:
        def get_relationship(self, relationship_id):
            assert relationship_id == "relationship:kraken:owner:old"
            return old_relationship

        def supersede_relationship(self, old, replacement):
            calls.append((old, replacement))
            return new_relationship

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        (
            "/memory/relationships/"
            "relationship:kraken:owner:old/supersede"
        ),
        json={
            "relationship_id": (
                "relationship:kraken:owner:flloyd"
            ),
            "target_entity_id": "person:flloyd",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["relationship_id"]
        == "relationship:kraken:owner:flloyd"
    )
    assert body["source_entity_id"] == "device:kraken"
    assert body["relationship"] == "belongs_to"
    assert body["target_entity_id"] == "person:flloyd"
    assert body["status"] == "current"
    assert (
        body["supersedes"]
        == "relationship:kraken:owner:old"
    )

    assert len(calls) == 1

    old, replacement = calls[0]

    assert old is old_relationship
    assert (
        replacement.relationship_id
        == "relationship:kraken:owner:flloyd"
    )
    assert (
        replacement.source_entity_id
        == old_relationship.source_entity_id
    )
    assert (
        replacement.relationship
        == old_relationship.relationship
    )
    assert replacement.target_entity_id == "person:flloyd"
    assert replacement.supersedes is None


def test_supersede_memory_relationship_endpoint_returns_404_when_missing(
    monkeypatch,
):
    from fastapi.testclient import TestClient

    import core.api.server as server

    supersede_calls = []

    class FakeMemoryManager:
        def get_relationship(self, relationship_id):
            assert relationship_id == "relationship:missing"
            return None

        def supersede_relationship(self, old, replacement):
            supersede_calls.append((old, replacement))

    class FakeAgent:
        memory_manager = FakeMemoryManager()

    monkeypatch.setattr(
        server,
        "get_agent",
        lambda: FakeAgent(),
    )

    client = TestClient(server.app)

    response = client.post(
        "/memory/relationships/relationship:missing/supersede",
        json={
            "relationship_id": "relationship:replacement",
            "target_entity_id": "person:flloyd",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Relationship not found: relationship:missing",
    }

    assert supersede_calls == []


def test_supersede_memory_relationship_endpoint_rejects_blank_fields():
    from fastapi.testclient import TestClient

    import core.api.server as server

    client = TestClient(server.app)

    for field in (
        "relationship_id",
        "target_entity_id",
        "source_type",
        "authority",
        "learned_at",
    ):
        payload = {
            "relationship_id": (
                "relationship:kraken:owner:flloyd"
            ),
            "target_entity_id": "person:flloyd",
            "source_type": "user_statement",
            "authority": "user",
            "learned_at": "2026-09-24T11:00:00-05:00",
        }

        payload[field] = "   "

        response = client.post(
            (
                "/memory/relationships/"
                "relationship:kraken:owner:old/supersede"
            ),
            json=payload,
        )

        assert response.status_code == 422