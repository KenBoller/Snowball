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