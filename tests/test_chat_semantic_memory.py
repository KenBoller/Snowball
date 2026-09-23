from core.ai import chat as chat_module
from core.ai.chat import SnowballAI


class FakeMemory:
    """Minimal legacy memory stub so the test never touches real user memory."""

    def __init__(self, *args, **kwargs):
        pass

    def store_interaction(self, *args, **kwargs):
        pass

    def get_memory(self, *args, **kwargs):
        return None

    def get_all_interactions(self, limit=10):
        return []

    def get_last_interaction(self):
        return None

    def close(self):
        pass


class FakeMemoryManager:
    """Provides semantic knowledge exactly as the real manager would."""

    def get_context(self, question, knowledge_result_count=5):
        return {
            "episodic": "",
            "knowledge": (
                "[KNOWLEDGE 1 | SOURCE: semantic-regression-test.txt | "
                "TYPE: snowball_project_document | "
                "AUTHORITY: historical_reference | "
                "DATE: 2025-09-30]\n"
                "Project Firefly's emergency access code is violet-orbit-7319."
            ),
        }


def test_chat_uses_semantic_knowledge_context(monkeypatch, tmp_path):
    # Never touch Snowball's real local memory during this test.
    monkeypatch.setattr(chat_module, "Memory", FakeMemory)

    agent = SnowballAI(storage_dir=str(tmp_path / "storage"))

    # Replace the manager with deterministic semantic knowledge.
    agent.memory_manager = FakeMemoryManager()

    def fake_query_with_fallback(primary_model, messages):
        system_prompt = messages[0]["content"]

        # This is the critical regression assertion:
        # semantic knowledge must reach the live model prompt.
        assert "Project Firefly" in system_prompt
        assert "violet-orbit-7319" in system_prompt

        return (
            "The emergency access code for Project Firefly "
            "is violet-orbit-7319."
        )

    monkeypatch.setattr(
        agent,
        "_query_with_fallback",
        fake_query_with_fallback,
    )

    response = agent.chat(
        "What is the emergency access code for Project Firefly?"
    )

    assert "violet-orbit-7319" in response