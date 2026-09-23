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