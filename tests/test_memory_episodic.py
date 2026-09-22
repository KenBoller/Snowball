from core.memory.episodic import LegacyMemoryAdapter


class FakeLegacyMemory:
    def __init__(self):
        self.interactions = []
        self.closed = False

    def store_interaction(
        self,
        *,
        user_input,
        ai_response,
        query_type="General",
    ):
        self.interactions.append(
            {
                "type": "interaction",
                "user_input": user_input,
                "ai_response": ai_response,
                "query_type": query_type,
            }
        )

    def get_memory(self, query):
        for interaction in reversed(self.interactions):
            if query.lower() in interaction["user_input"].lower():
                return interaction

        return self.interactions[-1] if self.interactions else None

    def get_all_interactions(self, limit=100):
        return list(reversed(self.interactions[-limit:]))

    def get_last_interaction(self):
        if not self.interactions:
            return None
        return self.interactions[-1]

    def close(self):
        self.closed = True


def test_adapter_stores_interaction():
    legacy = FakeLegacyMemory()
    adapter = LegacyMemoryAdapter(memory=legacy)

    adapter.remember_interaction(
        "My Neptune 4 printer is named Kraken.",
        "Got it. Kraken is your Neptune 4 printer.",
    )

    assert len(legacy.interactions) == 1
    assert (
        legacy.interactions[0]["user_input"]
        == "My Neptune 4 printer is named Kraken."
    )


def test_adapter_retrieves_memory():
    legacy = FakeLegacyMemory()
    adapter = LegacyMemoryAdapter(memory=legacy)

    adapter.remember_interaction(
        "Kraken",
        "Kraken is your Neptune 4 printer.",
    )

    result = adapter.retrieve("Kraken")

    assert result is not None
    assert result["ai_response"] == "Kraken is your Neptune 4 printer."


def test_adapter_returns_recent_interactions():
    legacy = FakeLegacyMemory()
    adapter = LegacyMemoryAdapter(memory=legacy)

    adapter.remember_interaction("one", "first")
    adapter.remember_interaction("two", "second")

    results = adapter.recent(limit=2)

    assert len(results) == 2
    assert results[0]["user_input"] == "two"
    assert results[1]["user_input"] == "one"


def test_adapter_returns_last_interaction():
    legacy = FakeLegacyMemory()
    adapter = LegacyMemoryAdapter(memory=legacy)

    adapter.remember_interaction("one", "first")
    adapter.remember_interaction("two", "second")

    result = adapter.last()

    assert result is not None
    assert result["user_input"] == "two"


def test_adapter_closes_legacy_memory():
    legacy = FakeLegacyMemory()
    adapter = LegacyMemoryAdapter(memory=legacy)

    adapter.close()

    assert legacy.closed is True