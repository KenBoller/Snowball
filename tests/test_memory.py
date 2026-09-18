def test_memory_backward_compat_role_style(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOWBALL_DISABLE_LOCAL_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_LOCAL_MEMORY_DIR", str(tmp_path))

    from core.ai.memory import Memory

    m = Memory(logger=None)
    m.store_interaction("user", "hello")
    m.store_interaction("assistant", "hi there")

    items = m.get_all_interactions(limit=10)
    assert len(items) >= 1
    assert any(d.get("type") == "interaction" for d in items)


def test_memory_new_style_store_and_last(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOWBALL_DISABLE_LOCAL_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_LOCAL_MEMORY_DIR", str(tmp_path))

    from core.ai.memory import Memory

    m = Memory(logger=None)
    m.store_interaction(user_input="What time is it?", ai_response="It is noon.", query_type="General")

    last = m.get_last_interaction()
    assert last is not None
    assert last.get("user_input") == "What time is it?"


def test_memory_retrieval_prefers_user_fact_over_bad_old_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("SNOWBALL_DISABLE_LOCAL_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_LOCAL_MEMORY_DIR", str(tmp_path))

    from core.ai.memory import Memory

    m = Memory(logger=None)

    # The authoritative user-provided fact.
    m.store_interaction(
        user_input="My Neptune 4 printer is named Kraken.",
        ai_response="Got it. Kraken is your Neptune 4 printer.",
        query_type="General",
    )

    # Simulate Snowball previously giving bad answers to the same question.
    m.store_interaction(
        user_input="What is Kraken?",
        ai_response="Kraken is a giant sea monster from mythology.",
        query_type="General",
    )

    m.store_interaction(
        user_input="What is Kraken?",
        ai_response="Kraken might be a software project or service.",
        query_type="General",
    )

    # Ask again.
    result = m.get_memory("What is Kraken?")

    assert result is not None
    assert result.get("user_input") == "My Neptune 4 printer is named Kraken."
    assert "Neptune 4" in result.get("user_input", "")
