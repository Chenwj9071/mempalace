#!/usr/bin/env python3
"""
event_index.py - Sidecar SQLite index for event-time filtering.

Stores only the metadata needed to shortlist candidate drawer IDs for
search_events. Canonical documents and full metadata remain in Chroma.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing

INDEX_FILENAME = "event_index.sqlite3"


def get_event_index_path(palace_path: str) -> str:
    return os.path.join(palace_path, INDEX_FILENAME)


def get_source_db_mtime(palace_path: str) -> str:
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return ""
    return f"{os.path.getmtime(db_path):.6f}"


def _connect(index_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_entries (
            drawer_id TEXT PRIMARY KEY,
            wing TEXT,
            room TEXT,
            record_kind TEXT,
            added_by TEXT,
            source_file TEXT,
            source_session_id TEXT,
            task_hint TEXT,
            timestamp_source TEXT,
            confidence TEXT NOT NULL,
            event_start TEXT,
            event_end TEXT,
            event_at TEXT
        );

        CREATE TABLE IF NOT EXISTS event_index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_event_entries_time
            ON event_entries(event_start, event_end);
        CREATE INDEX IF NOT EXISTS idx_event_entries_wing
            ON event_entries(wing);
        CREATE INDEX IF NOT EXISTS idx_event_entries_room
            ON event_entries(room);
        CREATE INDEX IF NOT EXISTS idx_event_entries_record_kind
            ON event_entries(record_kind);
        CREATE INDEX IF NOT EXISTS idx_event_entries_added_by
            ON event_entries(added_by);
        CREATE INDEX IF NOT EXISTS idx_event_entries_source_session_id
            ON event_entries(source_session_id);
        CREATE INDEX IF NOT EXISTS idx_event_entries_task_hint
            ON event_entries(task_hint);
        """
    )
    conn.commit()


def read_index_meta(palace_path: str) -> dict[str, str]:
    index_path = get_event_index_path(palace_path)
    if not os.path.isfile(index_path):
        return {}
    with closing(_connect(index_path)) as conn:
        _ensure_schema(conn)
        rows = conn.execute("SELECT key, value FROM event_index_meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def is_event_index_stale(palace_path: str, source_count: int) -> bool:
    meta = read_index_meta(palace_path)
    if not meta:
        return True
    return meta.get("source_db_mtime") != get_source_db_mtime(palace_path) or meta.get("source_count") != str(
        source_count
    )


def replace_event_index(
    palace_path: str,
    rows: list[dict],
    *,
    source_count: int,
    source_db_mtime: str,
) -> str:
    os.makedirs(palace_path, exist_ok=True)
    index_path = get_event_index_path(palace_path)
    with closing(_connect(index_path)) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM event_entries")
        conn.executemany(
            """
            INSERT INTO event_entries (
                drawer_id,
                wing,
                room,
                record_kind,
                added_by,
                source_file,
                source_session_id,
                task_hint,
                timestamp_source,
                confidence,
                event_start,
                event_end,
                event_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["drawer_id"],
                    row.get("wing"),
                    row.get("room"),
                    row.get("record_kind"),
                    row.get("added_by"),
                    row.get("source_file"),
                    row.get("source_session_id"),
                    row.get("task_hint"),
                    row.get("timestamp_source"),
                    row.get("confidence", "low"),
                    row.get("event_start"),
                    row.get("event_end"),
                    row.get("event_at"),
                )
                for row in rows
            ],
        )
        conn.execute("DELETE FROM event_index_meta")
        conn.executemany(
            "INSERT INTO event_index_meta(key, value) VALUES(?, ?)",
            [
                ("source_count", str(source_count)),
                ("source_db_mtime", source_db_mtime),
            ],
        )
        conn.commit()
    return index_path


def query_event_index(
    palace_path: str,
    *,
    time_from: str | None,
    time_to: str | None,
    wing: str | None,
    rooms: list[str],
    record_kinds: list[str],
    agents: list[str],
    session_ids: list[str],
    include_low_confidence: bool,
) -> list[str]:
    index_path = get_event_index_path(palace_path)
    if not os.path.isfile(index_path):
        return []

    conditions: list[str] = []
    params: list[str] = []

    if wing:
        conditions.append("wing = ?")
        params.append(wing)
    if rooms:
        conditions.append(f"room IN ({', '.join('?' for _ in rooms)})")
        params.extend(rooms)
    if record_kinds:
        conditions.append(f"record_kind IN ({', '.join('?' for _ in record_kinds)})")
        params.extend(record_kinds)
    if agents:
        conditions.append(f"added_by IN ({', '.join('?' for _ in agents)})")
        params.extend(agents)
    if session_ids:
        conditions.append(f"source_session_id IN ({', '.join('?' for _ in session_ids)})")
        params.extend(session_ids)
    if not include_low_confidence:
        conditions.append("confidence != 'low'")

    if time_from:
        if include_low_confidence:
            conditions.append("((event_end IS NOT NULL AND event_end >= ?) OR event_end IS NULL OR event_start IS NULL)")
        else:
            conditions.append("event_end IS NOT NULL AND event_end >= ?")
        params.append(time_from)
    if time_to:
        if include_low_confidence:
            conditions.append(
                "((event_start IS NOT NULL AND event_start < ?) OR event_start IS NULL OR event_end IS NULL)"
            )
        else:
            conditions.append("event_start IS NOT NULL AND event_start < ?")
        params.append(time_to)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT drawer_id
        FROM event_entries
        {where_clause}
        ORDER BY COALESCE(event_end, event_start, event_at, '') DESC, drawer_id ASC
    """

    with closing(_connect(index_path)) as conn:
        _ensure_schema(conn)
        rows = conn.execute(sql, params).fetchall()
    return [row["drawer_id"] for row in rows]
