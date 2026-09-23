from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.memory.semantic import MemorySource
from core.memory.structured import (
    Entity,
    Fact,
    Relationship,
    supersede_fact as build_fact_supersession,
    supersede_relationship as build_relationship_supersession,
)


class StructuredMemoryStore:
    """
    Persistent storage for Snowball's structured world model.

    Domain objects remain independent of SQLite. This store is responsible
    for translating between structured-memory objects and durable records.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(self.database_path)

        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value_json TEXT NOT NULL,
                source_json TEXT NOT NULL,
                learned_at TEXT NOT NULL,
                status TEXT NOT NULL,
                supersedes TEXT,
                metadata TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                source_json TEXT NOT NULL,
                learned_at TEXT NOT NULL,
                status TEXT NOT NULL,
                supersedes TEXT,
                metadata TEXT
            )
            """
        )

        self.connection.commit()

    def save_entity(self, entity: Entity) -> None:
        metadata = (
            json.dumps(entity.metadata)
            if entity.metadata is not None
            else None
        )

        self.connection.execute(
            """
            INSERT INTO entities (
                entity_id,
                entity_type,
                name,
                metadata
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                entity_type = excluded.entity_type,
                name = excluded.name,
                metadata = excluded.metadata
            """,
            (
                entity.entity_id,
                entity.entity_type,
                entity.name,
                metadata,
            ),
        )

        self.connection.commit()

    def get_entity(
        self,
        entity_id: str,
    ) -> Entity | None:
        row = self.connection.execute(
            """
            SELECT
                entity_id,
                entity_type,
                name,
                metadata
            FROM entities
            WHERE entity_id = ?
            """,
            (entity_id,),
        ).fetchone()

        if row is None:
            return None
            return self._entity_from_row(row)

        metadata = (
            json.loads(row[3])
            if row[3] is not None
            else None
        )

        return Entity(
            entity_id=row[0],
            entity_type=row[1],
            name=row[2],
            metadata=metadata,
        )

    def save_fact(self, fact: Fact) -> None:
        source_json = json.dumps(
            {
                "memory_type": fact.source.memory_type,
                "source_type": fact.source.source_type,
                "authority": fact.source.authority,
                "source_id": fact.source.source_id,
                "timestamp": fact.source.timestamp,
                "metadata": fact.source.metadata,
            }
        )

        metadata = (
            json.dumps(fact.metadata)
            if fact.metadata is not None
            else None
        )

        self.connection.execute(
            """
            INSERT INTO facts (
                fact_id,
                subject_id,
                predicate,
                value_json,
                source_json,
                learned_at,
                status,
                supersedes,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                predicate = excluded.predicate,
                value_json = excluded.value_json,
                source_json = excluded.source_json,
                learned_at = excluded.learned_at,
                status = excluded.status,
                supersedes = excluded.supersedes,
                metadata = excluded.metadata
            """,
            (
                fact.fact_id,
                fact.subject_id,
                fact.predicate,
                json.dumps(fact.value),
                source_json,
                fact.learned_at,
                fact.status,
                fact.supersedes,
                metadata,
            ),
        )

        self.connection.commit()

    def get_fact(
        self,
        fact_id: str,
    ) -> Fact | None:
        row = self.connection.execute(
            """
            SELECT
                fact_id,
                subject_id,
                predicate,
                value_json,
                source_json,
                learned_at,
                status,
                supersedes,
                metadata
            FROM facts
            WHERE fact_id = ?
            """,
            (fact_id,),
        ).fetchone()

        if row is None:
            return None

        return self._fact_from_row(row)

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact: Fact,
    ) -> None:
        old_fact = self.get_fact(old_fact_id)

        if old_fact is None:
            raise ValueError(
                f"Cannot supersede missing fact: {old_fact_id}"
            )

        historical, current = build_fact_supersession(
            old_fact,
            new_fact,
        )

        source_json = json.dumps(
            {
                "memory_type": current.source.memory_type,
                "source_type": current.source.source_type,
                "authority": current.source.authority,
                "source_id": current.source.source_id,
                "timestamp": current.source.timestamp,
                "metadata": current.source.metadata,
            }
        )

        metadata = (
            json.dumps(current.metadata)
            if current.metadata is not None
            else None
        )

        try:
            self.connection.execute("BEGIN")

            self.connection.execute(
                """
                UPDATE facts
                SET status = ?
                WHERE fact_id = ?
                """,
                (
                    historical.status,
                    historical.fact_id,
                ),
            )

            self.connection.execute(
                """
                INSERT INTO facts (
                    fact_id,
                    subject_id,
                    predicate,
                    value_json,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current.fact_id,
                    current.subject_id,
                    current.predicate,
                    json.dumps(current.value),
                    source_json,
                    current.learned_at,
                    current.status,
                    current.supersedes,
                    metadata,
                ),
            )

            self.connection.commit()

        except Exception:
            self.connection.rollback()
            raise

    def supersede_relationship(
        self,
        old_relationship_id: str,
        new_relationship: Relationship,
    ) -> None:
        old_relationship = self.get_relationship(
            old_relationship_id
        )

        if old_relationship is None:
            raise ValueError(
                "Cannot supersede missing relationship: "
                f"{old_relationship_id}"
            )

        historical, current = build_relationship_supersession(
            old_relationship,
            new_relationship,
        )

        source_json = json.dumps(
            {
                "memory_type": current.source.memory_type,
                "source_type": current.source.source_type,
                "authority": current.source.authority,
                "source_id": current.source.source_id,
                "timestamp": current.source.timestamp,
                "metadata": current.source.metadata,
            }
        )

        metadata = (
            json.dumps(current.metadata)
            if current.metadata is not None
            else None
        )

        try:
            self.connection.execute("BEGIN")

            self.connection.execute(
                """
                UPDATE relationships
                SET status = ?
                WHERE relationship_id = ?
                """,
                (
                    historical.status,
                    historical.relationship_id,
                ),
            )

            self.connection.execute(
                """
                INSERT INTO relationships (
                    relationship_id,
                    source_entity_id,
                    relationship,
                    target_entity_id,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current.relationship_id,
                    current.source_entity_id,
                    current.relationship,
                    current.target_entity_id,
                    source_json,
                    current.learned_at,
                    current.status,
                    current.supersedes,
                    metadata,
                ),
            )

            self.connection.commit()

        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()

    def save_relationship(
        self,
        relationship: Relationship,
    ) -> None:
        source_json = json.dumps(
            {
                "memory_type": relationship.source.memory_type,
                "source_type": relationship.source.source_type,
                "authority": relationship.source.authority,
                "source_id": relationship.source.source_id,
                "timestamp": relationship.source.timestamp,
                "metadata": relationship.source.metadata,
            }
        )

        metadata = (
            json.dumps(relationship.metadata)
            if relationship.metadata is not None
            else None
        )

        self.connection.execute(
            """
            INSERT INTO relationships (
                relationship_id,
                source_entity_id,
                relationship,
                target_entity_id,
                source_json,
                learned_at,
                status,
                supersedes,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relationship_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                relationship = excluded.relationship,
                target_entity_id = excluded.target_entity_id,
                source_json = excluded.source_json,
                learned_at = excluded.learned_at,
                status = excluded.status,
                supersedes = excluded.supersedes,
                metadata = excluded.metadata
            """,
            (
                relationship.relationship_id,
                relationship.source_entity_id,
                relationship.relationship,
                relationship.target_entity_id,
                source_json,
                relationship.learned_at,
                relationship.status,
                relationship.supersedes,
                metadata,
            ),
        )

        self.connection.commit()

    def get_relationship(
        self,
        relationship_id: str,
    ) -> Relationship | None:
        row = self.connection.execute(
            """
            SELECT
                relationship_id,
                source_entity_id,
                relationship,
                target_entity_id,
                source_json,
                learned_at,
                status,
                supersedes,
                metadata
            FROM relationships
            WHERE relationship_id = ?
            """,
            (relationship_id,),
        ).fetchone()

        if row is None:
            return None

        return self._relationship_from_row(row)

    def get_current_relationships(
        self,
        source_entity_id: str,
        *,
        relationship: str | None = None,
    ) -> list[Relationship]:
        if relationship is None:
            rows = self.connection.execute(
                """
                SELECT
                    relationship_id,
                    source_entity_id,
                    relationship,
                    target_entity_id,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                FROM relationships
                WHERE source_entity_id = ?
                  AND status = 'current'
                ORDER BY learned_at, relationship_id
                """,
                (source_entity_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT
                    relationship_id,
                    source_entity_id,
                    relationship,
                    target_entity_id,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                FROM relationships
                WHERE source_entity_id = ?
                  AND relationship = ?
                  AND status = 'current'
                ORDER BY learned_at, relationship_id
                """,
                (
                    source_entity_id,
                    relationship,
                ),
            ).fetchall()

        return [
            self._relationship_from_row(row)
            for row in rows
        ]

    def _fact_from_row(self, row: tuple) -> Fact:
        source_data = json.loads(row[4])

        source = MemorySource(
            memory_type=source_data["memory_type"],
            source_type=source_data["source_type"],
            authority=source_data["authority"],
            source_id=source_data.get("source_id"),
            timestamp=source_data.get("timestamp"),
            metadata=source_data.get("metadata"),
        )

        metadata = (
            json.loads(row[8])
            if row[8] is not None
            else None
        )

        return Fact(
            fact_id=row[0],
            subject_id=row[1],
            predicate=row[2],
            value=json.loads(row[3]),
            source=source,
            learned_at=row[5],
            status=row[6],
            supersedes=row[7],
            metadata=metadata,
        )

    def get_current_facts(
        self,
        subject_id: str,
        *,
        predicate: str | None = None,
    ) -> list[Fact]:
        if predicate is None:
            rows = self.connection.execute(
                """
                SELECT
                    fact_id,
                    subject_id,
                    predicate,
                    value_json,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                FROM facts
                WHERE subject_id = ?
                  AND status = 'current'
                ORDER BY learned_at, fact_id
                """,
                (subject_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT
                    fact_id,
                    subject_id,
                    predicate,
                    value_json,
                    source_json,
                    learned_at,
                    status,
                    supersedes,
                    metadata
                FROM facts
                WHERE subject_id = ?
                  AND predicate = ?
                  AND status = 'current'
                ORDER BY learned_at, fact_id
                """,
                (
                    subject_id,
                    predicate,
                ),
            ).fetchall()

        return [
            self._fact_from_row(row)
            for row in rows
        ]

    def _relationship_from_row(
        self,
        row: tuple,
    ) -> Relationship:
        source_data = json.loads(row[4])

        source = MemorySource(
            memory_type=source_data["memory_type"],
            source_type=source_data["source_type"],
            authority=source_data["authority"],
            source_id=source_data.get("source_id"),
            timestamp=source_data.get("timestamp"),
            metadata=source_data.get("metadata"),
        )

        metadata = (
            json.loads(row[8])
            if row[8] is not None
            else None
        )

        return Relationship(
            relationship_id=row[0],
            source_entity_id=row[1],
            relationship=row[2],
            target_entity_id=row[3],
            source=source,
            learned_at=row[5],
            status=row[6],
            supersedes=row[7],
            metadata=metadata,
        )

    def find_entities_by_name(
        self,
        name: str,
    ) -> list[Entity]:
        rows = self.connection.execute(
            """
            SELECT
                entity_id,
                entity_type,
                name,
                metadata
            FROM entities
            WHERE LOWER(name) = LOWER(?)
            ORDER BY entity_id
            """,
            (name,),
        ).fetchall()

        return [
            Entity(
                entity_id=row[0],
                entity_type=row[1],
                name=row[2],
                metadata=(
                    json.loads(row[3])
                    if row[3] is not None
                    else None
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _entity_from_row(row) -> Entity:
        return Entity(
            entity_id=row[0],
            entity_type=row[1],
            name=row[2],
            metadata=(
                json.loads(row[3])
                if row[3] is not None
                else None
            ),
        )

    def list_entities(self) -> list[Entity]:
        rows = self.connection.execute(
            """
            SELECT
                entity_id,
                entity_type,
                name,
                metadata
            FROM entities
            ORDER BY entity_id
            """
        ).fetchall()

        return [
            self._entity_from_row(row)
            for row in rows
        ]