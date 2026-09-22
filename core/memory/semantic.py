from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemorySource:
    """
    Describes where retrieved context came from.

    Provenance is kept separate from retrieval relevance so Snowball can
    reason about the trust and meaning of information independently from
    how closely it matches a query.
    """

    memory_type: str
    source_type: str
    authority: str
    source_id: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None


def user_statement_source(
    *,
    source_id: str | None = None,
    timestamp: str | None = None,
) -> MemorySource:
    return MemorySource(
        memory_type="episodic",
        source_type="user_statement",
        authority="user",
        source_id=source_id,
        timestamp=timestamp,
    )


def knowledge_source(
    *,
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemorySource:
    return MemorySource(
        memory_type="semantic",
        source_type="knowledge_document",
        authority="reference",
        source_id=source_id,
        metadata=metadata,
    )


@dataclass(frozen=True)
class MemoryContextEntry:
    """
    A piece of retrieved context together with its provenance.

    Retrieval determines whether an entry is relevant.
    MemorySource describes what the entry is and where it came from.
    """

    text: str
    source: MemorySource
    relevance: float | None = None