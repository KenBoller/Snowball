import types

class DummyResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

def test_chat_calls_ollama_and_returns_content(
    monkeypatch,
    tmp_path,
):
    # ensure we don't use memory/DM for this test
    monkeypatch.setenv("SNOWBALL_USE_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_USE_DM", "0")
    monkeypatch.setenv("SNOWBALL_FANOUT", "0")

    # mock requests.post
    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/api/chat")
        assert "model" in (json or {})
        return DummyResp({"message": {"content": "hello from fake ollama"}})

    import core.ai.chat as chatmod
    monkeypatch.setattr(chatmod.requests, "post", fake_post)

    ai = chatmod.SnowballAI(
        logger=None,
        storage_dir=str(tmp_path / "storage"),
    )
    out = ai.chat("hello")
    assert out.strip() == "hello from fake ollama"

def test_chat_fallback_when_provider_errors(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("SNOWBALL_USE_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_USE_DM", "0")
    monkeypatch.setenv("SNOWBALL_FANOUT", "0")

    def fake_post(url, json=None, timeout=None):
        raise RuntimeError("provider down")

    import core.ai.chat as chatmod
    monkeypatch.setattr(chatmod.requests, "post", fake_post)

    ai = chatmod.SnowballAI(
        logger=None,
        storage_dir=str(tmp_path / "storage"),
    )
    out = ai.chat("help me")
    assert isinstance(out, str)
    assert len(out) > 0
