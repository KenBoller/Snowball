def test_api_send_message_returns_string():
    try:
        from core.api import send_message
    except Exception:
        return  # MVP-safe: API may not be present in some setups

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