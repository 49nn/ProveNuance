"""
PostgresKnowledgeStore — implementacja portu KnowledgeStore na PostgreSQL.

Schemat:
  - facts           — trójki (h, r, t) z statusem i provenance
  - rules           — reguły Horn z ciałem jako ARRAY
  - snapshots       — punkty w czasie (labelled)
  - snapshot_facts  — które fakty były asserted w danym snapshot
  - snapshot_rules  — które reguły były asserted w danym snapshot
  - fact_history    — pełny audit log każdej zmiany statusu faktu
  - rule_history    — pełny audit log każdej zmiany statusu reguły

Unikalność:
  - facts: UNIQUE(h, r, t)   → upsert idempotentny
  - rules: UNIQUE(head, body) → upsert idempotentny (body jako TEXT[])
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import asyncpg

from contracts import (
    Fact,
    FactDiff,
    FactRef,
    FactStatus,
    Rule,
    RuleDiff,
    RuleRef,
    SnapshotDiff,
    SnapshotRef,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    h          TEXT NOT NULL,
    r          TEXT NOT NULL,
    t          TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'hypothesis',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    provenance TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_fact_hrt UNIQUE (h, r, t)
);

CREATE TABLE IF NOT EXISTS rules (
    rule_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    head       TEXT NOT NULL,
    body       TEXT[] NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'hypothesis',
    priority   INTEGER NOT NULL DEFAULT 0,
    provenance TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_rule_headbody UNIQUE (head, body)
);

CREATE INDEX IF NOT EXISTS idx_facts_status  ON facts(status);
CREATE INDEX IF NOT EXISTS idx_rules_status  ON rules(status);
CREATE INDEX IF NOT EXISTS idx_rules_priority ON rules(priority DESC);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS snapshot_facts (
    snapshot_id UUID NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    fact_id     UUID NOT NULL REFERENCES facts(fact_id)         ON DELETE CASCADE,
    PRIMARY KEY (snapshot_id, fact_id)
);

CREATE TABLE IF NOT EXISTS snapshot_rules (
    snapshot_id UUID NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    rule_id     UUID NOT NULL REFERENCES rules(rule_id)         ON DELETE CASCADE,
    PRIMARY KEY (snapshot_id, rule_id)
);

CREATE TABLE IF NOT EXISTS fact_history (
    id          BIGSERIAL PRIMARY KEY,
    fact_id     UUID NOT NULL,
    old_status  TEXT,
    new_status  TEXT NOT NULL,
    reason      TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rule_history (
    id          BIGSERIAL PRIMARY KEY,
    rule_id     UUID NOT NULL,
    old_status  TEXT,
    new_status  TEXT NOT NULL,
    reason      TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fact_history_fact ON fact_history(fact_id);
CREATE INDEX IF NOT EXISTS idx_rule_history_rule ON rule_history(rule_id);
"""


class PostgresKnowledgeStore:
    """Implementacja KnowledgeStore na PostgreSQL + asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ──────────────────────── Lifecycle ──────────────────────────────────

    @classmethod
    async def create(cls, dsn: str) -> "PostgresKnowledgeStore":
        """Factory: tworzy pool połączeń i aplikuje schemat."""
        pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        store = cls(pool)
        await store._apply_schema()
        return store

    async def _apply_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)

    async def close(self) -> None:
        await self._pool.close()

    async def reset_all(self) -> dict[str, int]:
        """
        Deletes all KnowledgeStore rows (facts, rules, snapshots, history).
        Returns row counts captured before truncation.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                counts = {
                    "facts": int(await conn.fetchval("SELECT COUNT(*) FROM facts")),
                    "rules": int(await conn.fetchval("SELECT COUNT(*) FROM rules")),
                    "snapshots": int(await conn.fetchval("SELECT COUNT(*) FROM snapshots")),
                    "fact_history": int(await conn.fetchval("SELECT COUNT(*) FROM fact_history")),
                    "rule_history": int(await conn.fetchval("SELECT COUNT(*) FROM rule_history")),
                }
                await conn.execute(
                    """
                    TRUNCATE TABLE
                        snapshot_facts,
                        snapshot_rules,
                        fact_history,
                        rule_history,
                        snapshots,
                        facts,
                        rules
                    RESTART IDENTITY
                    """
                )
        return counts

    # ──────────────────────── Facts ──────────────────────────────────────

    async def upsert_fact(self, f: Fact) -> FactRef:
        """
        INSERT … ON CONFLICT (h,r,t) DO UPDATE.
        Jeśli fakt już istnieje: aktualizuje confidence, provenance, updated_at.
        Zwraca FactRef z created=True jeśli nowy.
        """
        now = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO facts (fact_id, h, r, t, status, confidence, provenance, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                ON CONFLICT (h, r, t) DO UPDATE
                    SET confidence  = EXCLUDED.confidence,
                        provenance  = EXCLUDED.provenance,
                        updated_at  = EXCLUDED.updated_at
                RETURNING fact_id,
                          (xmax = 0) AS inserted
                """,
                f.fact_id,
                f.h,
                f.r,
                f.t,
                f.status.value,
                f.confidence,
                f.provenance,
                now,
            )
        return FactRef(fact_id=str(row["fact_id"]), created=bool(row["inserted"]))

    async def get_fact(self, fact_id: str) -> Fact:
        """Rzuca KeyError jeśli nie znaleziono."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM facts WHERE fact_id = $1", fact_id
            )
        if row is None:
            raise KeyError(f"Fact not found: {fact_id}")
        return _row_to_fact(row)

    async def get_asserted_facts(
        self, snapshot: Optional[SnapshotRef] = None
    ) -> AsyncIterator[Fact]:
        """
        Iteruje przez asserted fakty.
        Jeśli snapshot podany: zwraca fakty z tamtego punktu w czasie.
        """
        if snapshot:
            sql = """
                SELECT f.* FROM facts f
                JOIN snapshot_facts sf ON sf.fact_id = f.fact_id
                WHERE sf.snapshot_id = $1
                ORDER BY f.created_at
            """
            params = [snapshot.snapshot_id]
        else:
            sql = "SELECT * FROM facts WHERE status = 'asserted' ORDER BY created_at"
            params = []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        for row in rows:
            yield _row_to_fact(row)

    async def promote_fact(self, fact_id: str, reason: str) -> None:
        """Zmienia status hypothesis → asserted + audit log."""
        now = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT status FROM facts WHERE fact_id = $1", fact_id
                )
                if row is None:
                    raise KeyError(f"Fact not found: {fact_id}")
                old_status = row["status"]
                await conn.execute(
                    "UPDATE facts SET status = 'asserted', updated_at = $1 WHERE fact_id = $2",
                    now,
                    fact_id,
                )
                await conn.execute(
                    """
                    INSERT INTO fact_history (fact_id, old_status, new_status, reason, changed_at)
                    VALUES ($1, $2, 'asserted', $3, $4)
                    """,
                    fact_id,
                    old_status,
                    reason,
                    now,
                )

    async def retract_fact(self, fact_id: str, reason: str) -> None:
        """Zmienia status → retracted + audit log."""
        now = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT status FROM facts WHERE fact_id = $1", fact_id
                )
                if row is None:
                    raise KeyError(f"Fact not found: {fact_id}")
                old_status = row["status"]
                await conn.execute(
                    "UPDATE facts SET status = 'retracted', updated_at = $1 WHERE fact_id = $2",
                    now,
                    fact_id,
                )
                await conn.execute(
                    """
                    INSERT INTO fact_history (fact_id, old_status, new_status, reason, changed_at)
                    VALUES ($1, $2, 'retracted', $3, $4)
                    """,
                    fact_id,
                    old_status,
                    reason,
                    now,
                )

    # ──────────────────────── Rules ──────────────────────────────────────

    async def upsert_rule(self, r: Rule) -> RuleRef:
        """
        INSERT … ON CONFLICT (head, body) DO UPDATE.
        Zwraca RuleRef z created=True jeśli nowa.
        """
        now = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO rules (rule_id, head, body, status, priority, provenance, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (head, body) DO UPDATE
                    SET provenance = EXCLUDED.provenance,
                        priority   = EXCLUDED.priority
                RETURNING rule_id,
                          (xmax = 0) AS inserted
                """,
                r.rule_id,
                r.head,
                r.body,          # asyncpg obsługuje list[str] → TEXT[]
                r.status.value,
                r.priority,
                r.provenance,
                now,
            )
        return RuleRef(rule_id=str(row["rule_id"]), created=bool(row["inserted"]))

    async def get_rule(self, rule_id: str) -> Rule:
        """Rzuca KeyError jeśli nie znaleziono."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM rules WHERE rule_id = $1", rule_id
            )
        if row is None:
            raise KeyError(f"Rule not found: {rule_id}")
        return _row_to_rule(row)

    async def get_asserted_rules(
        self, snapshot: Optional[SnapshotRef] = None
    ) -> AsyncIterator[Rule]:
        """Iteruje przez asserted reguły (lub reguły z danego snapshot)."""
        if snapshot:
            sql = """
                SELECT r.* FROM rules r
                JOIN snapshot_rules sr ON sr.rule_id = r.rule_id
                WHERE sr.snapshot_id = $1
                ORDER BY r.priority DESC, r.created_at
            """
            params = [snapshot.snapshot_id]
        else:
            sql = """
                SELECT * FROM rules
                WHERE status = 'asserted'
                ORDER BY priority DESC, created_at
            """
            params = []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        for row in rows:
            yield _row_to_rule(row)

    async def promote_rule(self, rule_id: str, reason: str) -> None:
        """Zmienia status hypothesis → asserted + audit log."""
        now = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT status FROM rules WHERE rule_id = $1", rule_id
                )
                if row is None:
                    raise KeyError(f"Rule not found: {rule_id}")
                old_status = row["status"]
                await conn.execute(
                    "UPDATE rules SET status = 'asserted' WHERE rule_id = $1",
                    rule_id,
                )
                await conn.execute(
                    """
                    INSERT INTO rule_history (rule_id, old_status, new_status, reason, changed_at)
                    VALUES ($1, $2, 'asserted', $3, $4)
                    """,
                    rule_id,
                    old_status,
                    reason,
                    now,
                )

    # ──────────────────────── List (poza portem) ─────────────────────────

    async def list_facts(
        self, status: str = "asserted", limit: int = 200
    ) -> list[Fact]:
        """Zwraca fakty wg statusu ('hypothesis'|'asserted'|'retracted'|'all')."""
        async with self._pool.acquire() as conn:
            if status == "all":
                rows = await conn.fetch(
                    "SELECT * FROM facts ORDER BY created_at LIMIT $1", limit
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM facts WHERE status = $1 ORDER BY created_at LIMIT $2",
                    status, limit,
                )
        return [_row_to_fact(r) for r in rows]

    async def list_rules(
        self, status: str = "asserted", limit: int = 200
    ) -> list[Rule]:
        """Zwraca reguły wg statusu ('hypothesis'|'asserted'|'all')."""
        async with self._pool.acquire() as conn:
            if status == "all":
                rows = await conn.fetch(
                    "SELECT * FROM rules ORDER BY priority DESC, created_at LIMIT $1",
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM rules WHERE status = $1
                       ORDER BY priority DESC, created_at LIMIT $2""",
                    status, limit,
                )
        return [_row_to_rule(r) for r in rows]

    # ──────────────────────── Snapshots ──────────────────────────────────

    async def create_snapshot(self, label: Optional[str] = None) -> SnapshotRef:
        """
        Tworzy snapshot: kopiuje wszystkie aktualnie asserted fakty i reguły
        do tabel snapshot_facts / snapshot_rules.
        """
        snap = SnapshotRef(label=label)
        now = datetime.now(tz=timezone.utc)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO snapshots (snapshot_id, label, created_at) VALUES ($1, $2, $3)",
                    snap.snapshot_id,
                    snap.label,
                    now,
                )
                await conn.execute(
                    """
                    INSERT INTO snapshot_facts (snapshot_id, fact_id)
                    SELECT $1, fact_id FROM facts WHERE status = 'asserted'
                    """,
                    snap.snapshot_id,
                )
                await conn.execute(
                    """
                    INSERT INTO snapshot_rules (snapshot_id, rule_id)
                    SELECT $1, rule_id FROM rules WHERE status = 'asserted'
                    """,
                    snap.snapshot_id,
                )

        return snap

    async def diff_snapshots(self, a_id: str, b_id: str) -> SnapshotDiff:
        """
        Oblicza różnicę między dwoma snapshotami.
        - added facts/rules: w B ale nie w A
        - removed facts/rules: w A ale nie w B
        - changed: fakty o tym samym fact_id ale różnych wartościach
          (nie możliwe dla immutable trójek — pozostawione puste)
        """
        async with self._pool.acquire() as conn:
            # Sprawdź istnienie
            for snap_id in (a_id, b_id):
                if not await conn.fetchval(
                    "SELECT 1 FROM snapshots WHERE snapshot_id = $1", snap_id
                ):
                    raise KeyError(f"Snapshot not found: {snap_id}")

            # Fakty dodane (w B, nie w A)
            added_fact_rows = await conn.fetch(
                """
                SELECT f.* FROM facts f
                JOIN snapshot_facts sf ON sf.fact_id = f.fact_id AND sf.snapshot_id = $2
                WHERE f.fact_id NOT IN (
                    SELECT fact_id FROM snapshot_facts WHERE snapshot_id = $1
                )
                """,
                a_id, b_id,
            )
            # Fakty usunięte (w A, nie w B)
            removed_fact_rows = await conn.fetch(
                """
                SELECT f.* FROM facts f
                JOIN snapshot_facts sf ON sf.fact_id = f.fact_id AND sf.snapshot_id = $1
                WHERE f.fact_id NOT IN (
                    SELECT fact_id FROM snapshot_facts WHERE snapshot_id = $2
                )
                """,
                a_id, b_id,
            )
            # Reguły dodane
            added_rule_rows = await conn.fetch(
                """
                SELECT r.* FROM rules r
                JOIN snapshot_rules sr ON sr.rule_id = r.rule_id AND sr.snapshot_id = $2
                WHERE r.rule_id NOT IN (
                    SELECT rule_id FROM snapshot_rules WHERE snapshot_id = $1
                )
                """,
                a_id, b_id,
            )
            # Reguły usunięte
            removed_rule_rows = await conn.fetch(
                """
                SELECT r.* FROM rules r
                JOIN snapshot_rules sr ON sr.rule_id = r.rule_id AND sr.snapshot_id = $1
                WHERE r.rule_id NOT IN (
                    SELECT rule_id FROM snapshot_rules WHERE snapshot_id = $2
                )
                """,
                a_id, b_id,
            )

        return SnapshotDiff(
            snapshot_a=a_id,
            snapshot_b=b_id,
            fact_diff=FactDiff(
                added=[_row_to_fact(r) for r in added_fact_rows],
                removed=[_row_to_fact(r) for r in removed_fact_rows],
                changed=[],  # immutable trójki — nigdy się nie zmieniają
            ),
            rule_diff=RuleDiff(
                added=[_row_to_rule(r) for r in added_rule_rows],
                removed=[_row_to_rule(r) for r in removed_rule_rows],
            ),
        )


# ──────────────────────── Helpers ────────────────────────────────────────

def _row_to_fact(row: asyncpg.Record) -> Fact:
    return Fact(
        fact_id=str(row["fact_id"]),
        h=row["h"],
        r=row["r"],
        t=row["t"],
        status=FactStatus(row["status"]),
        confidence=float(row["confidence"]),
        provenance=list(row["provenance"] or []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_rule(row: asyncpg.Record) -> Rule:
    return Rule(
        rule_id=str(row["rule_id"]),
        head=row["head"],
        body=list(row["body"] or []),
        status=FactStatus(row["status"]),
        priority=int(row["priority"]),
        provenance=list(row["provenance"] or []),
        created_at=row["created_at"],
    )
