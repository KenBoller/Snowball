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


def test_memory_persists_fact_across_reinitialization(tmp_path, monkeypatch):
    """
    Regression test for Snowball's core persistence promise:

    Learn a fact -> write it to disk -> create a fresh Memory instance ->
    retrieve the fact without manually reseeding it.
    """
    monkeypatch.setenv("SNOWBALL_DISABLE_LOCAL_MEMORY", "0")
    monkeypatch.setenv("SNOWBALL_LOCAL_MEMORY_DIR", str(tmp_path))

    from core.ai.memory import Memory

    # First Snowball lifetime: learn something new.
    first_memory = Memory(logger=None)
    first_memory.store_interaction(
        user_input="My test rover is named Cobalt.",
        ai_response="Got it. Cobalt is your test rover.",
        query_type="General",
    )

    # Confirm persistence actually reached disk.
    memory_file = tmp_path / "local_memory.jsonl"
    assert memory_file.exists()
    assert "Cobalt" in memory_file.read_text(encoding="utf-8")

    # Simulate a fresh Snowball lifetime.
    del first_memory
    second_memory = Memory(logger=None)

    # The new instance must recover the fact from persisted memory.
    result = second_memory.get_memory("What is Cobalt?")

    assert result is not None
    assert result.get("user_input") == "My test rover is named Cobalt."
    assert "Cobalt" in result.get("user_input", "")
