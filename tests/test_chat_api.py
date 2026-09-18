def test_chat_api_singleton_same_instance(monkeypatch):
    # disable optional modules
    monkeypatch.setenv("SNOWBALL_USE_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_USE_DM", "0")
    monkeypatch.setenv("SNOWBALL_USE_SENTIMENT", "0")
    monkeypatch.setenv("SNOWBALL_FANOUT", "0")

    from core.api.chat_api import get_chat_agent, reset_chat_agent

    reset_chat_agent()
    a = get_chat_agent()
    b = get_chat_agent()

    assert a is b
