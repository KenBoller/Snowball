from __future__ import annotations

from typing import Any

from core.ai.memory import Memory


class LegacyMemoryAdapter:
    """
    Adapter around Snowball's existing persistent interaction memory.

    This keeps the proven JSONL-backed memory system intact while allowing
    the newer MemoryManager to consume it through a smaller, stable
    interface.
    """

    def __init__(self, memory: Memory | None = None):
        self.memory = memory or Memory()

    def remember_interaction(
        self,
        user_input: str,
        ai_response: str,
        query_type: str = "General",
    ) -> None:
        self.memory.store_interaction(
            user_input=user_input,
            ai_response=ai_response,
            query_type=query_type,
        )

    def retrieve(self, query: str) -> dict[str, Any] | None:
        return self.memory.get_memory(query)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.memory.get_all_interactions(limit=limit)

    def last(self) -> dict[str, Any] | None:
        return self.memory.get_last_interaction()

    def close(self) -> None:
        self.memory.close()