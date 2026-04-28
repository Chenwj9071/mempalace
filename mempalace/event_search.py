#!/usr/bin/env python3
"""
event_search.py - Search memories by real event time instead of filed_at.

This module powers the first-stage "search_events" MVP:
- filter records by event_at / event_time_start / event_time_end
- keep transcript + memory as the default corpus
- return grouped summaries first, then optional evidence
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .event_index import (
    get_source_db_mtime,
    is_event_index_stale,
    query_event_index,
    replace_event_index,
)
from .palace import get_collection
from .query_sanitizer import sanitize_query
from .searcher import SearchError, _bm25_scores

_TASK_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,31}")
_SHORT_TASK_TOKENS = {"ai", "bi", "ci", "db", "id", "kg", "mcp", "ui", "ux"}
_TASK_STOPWORDS = {
    "about",
    "added",
    "after",
    "agent",
    "agents",
    "already",
    "assistant",
    "before",
    "being",
    "build",
    "built",
    "change",
    "changes",
    "check",
    "checked",
    "clean",
    "confirmed",
    "continue",
    "created",
    "current",
    "debug",
    "default",
    "deliverable",
    "deliverables",
    "documented",
    "done",
    "during",
    "error",
    "evidence",
    "expanded",
    "expansion",
    "feature",
    "file",
    "files",
    "filter",
    "filtered",
    "fix",
    "fixed",
    "follow",
    "from",
    "general",
    "group",
    "grouped",
    "grouping",
    "groups",
    "implemented",
    "implementation",
    "including",
    "into",
    "item",
    "items",
    "later",
    "limit",
    "matched",
    "memory",
    "message",
    "messages",
    "metadata",
    "mode",
    "need",
    "needs",
    "next",
    "output",
    "passed",
    "plan",
    "planning",
    "progress",
    "query",
    "record",
    "records",
    "result",
    "results",
    "room",
    "score",
    "search",
    "session",
    "sessions",
    "source",
    "step",
    "task",
    "tests",
    "time",
    "today",
    "tool",
    "tools",
    "update",
    "updated",
    "user",
    "using",
    "verified",
    "wire",
    "wired",
}

_BATCH_SIZE = 1000
_DEFAULT_RECORD_KINDS = ("transcript", "memory")
_VALID_GROUP_BY = {"task", "session", "source_file", "day"}
_VALID_EXPAND_LEVELS = {"overview", "grouped", "evidence"}
_VALID_CONFIDENCE = {"high", "medium", "low"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

_DECISION_KEYWORDS = (
    "decide",
    "decided",
    "decision",
    "agreed",
    "choose",
    "chose",
    "selected",
    "plan is",
    "决定",
    "采用",
    "方案",
)
_BLOCKER_KEYWORDS = (
    "blocker",
    "blocked",
    "blocking",
    "issue",
    "problem",
    "risk",
    "error",
    "fail",
    "阻塞",
    "问题",
    "失败",
    "风险",
)
_DELIVERABLE_KEYWORDS = (
    "implemented",
    "implementation",
    "added",
    "created",
    "migrated",
    "pushed",
    "merged",
    "documented",
    "测试通过",
    "已完成",
    "实现",
    "新增",
    "迁移",
    "文档",
)
_NEXT_STEP_KEYWORDS = (
    "next",
    "todo",
    "follow up",
    "follow-up",
    "next step",
    "待办",
    "下一步",
    "后续",
)


_SESSION_ID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_FIXED_TIMEZONE_FALLBACKS = {
    "Asia/Shanghai": timezone(timedelta(hours=8), "Asia/Shanghai"),
}


def _parse_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    parsed: list[str] = []
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            parsed.extend(part.strip() for part in raw.split(",") if part.strip())
    return parsed


def _parse_iso_timestamp(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_query_timezone(timezone_name: str | None) -> tuple[timezone | ZoneInfo, str]:
    if not timezone_name:
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc
        label = getattr(local_tz, "key", None) or local_tz.tzname(None) or "local"
        return local_tz, label
    if timezone_name.upper() == "UTC":
        return timezone.utc, "UTC"
    if re.fullmatch(r"[+-]\d{2}:\d{2}", timezone_name):
        sign = 1 if timezone_name[0] == "+" else -1
        hours = int(timezone_name[1:3])
        minutes = int(timezone_name[4:6])
        offset = timedelta(hours=hours, minutes=minutes) * sign
        return timezone(offset, timezone_name), timezone_name
    try:
        return ZoneInfo(timezone_name), timezone_name
    except ZoneInfoNotFoundError:
        fallback = _FIXED_TIMEZONE_FALLBACKS.get(timezone_name)
        if fallback is not None:
            return fallback, timezone_name
        raise ValueError(f"Unknown timezone: {timezone_name}") from None


def _parse_user_timestamp(value: str, query_timezone: timezone | ZoneInfo) -> datetime | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=query_timezone)
    return parsed.astimezone(timezone.utc)


def _parse_time_boundary(
    value: str | None, is_end: bool, query_timezone: timezone | ZoneInfo
) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = date.fromisoformat(value)
        if is_end:
            day = day + timedelta(days=1)
        return datetime.combine(day, time.min, tzinfo=query_timezone).astimezone(timezone.utc)
    return _parse_user_timestamp(value, query_timezone)


def _normalize_time_range(
    time_from: str | None,
    time_to: str | None,
    query_timezone: timezone | ZoneInfo,
) -> tuple[datetime | None, datetime | None]:
    start = _parse_time_boundary(time_from, is_end=False, query_timezone=query_timezone)
    end = _parse_time_boundary(time_to, is_end=True, query_timezone=query_timezone)
    if start and end and start >= end:
        raise ValueError("time_from must be earlier than time_to")
    return start, end


def _extract_session_ids(value: str | None) -> list[str]:
    if not value:
        return []
    session_ids: list[str] = []
    for match in _SESSION_ID_RE.finditer(value):
        session_id = match.group(0).lower()
        if session_id not in session_ids:
            session_ids.append(session_id)
    return session_ids


def _coerce_confidence(meta: dict, has_event_fields: bool) -> tuple[str, str | None]:
    timestamp_source = meta.get("timestamp_source")
    if timestamp_source == "message_timestamp":
        return "high", timestamp_source
    if timestamp_source in {"file_mtime", "source_modified_at", "source_created_at"}:
        return "medium", timestamp_source
    if timestamp_source == "ingest_time":
        return "low", timestamp_source
    if has_event_fields:
        return "high", timestamp_source or "event_metadata"
    if meta.get("source_modified_at"):
        return "medium", "source_modified_at"
    if meta.get("source_created_at"):
        return "medium", "source_created_at"
    return "low", timestamp_source


def _resolve_event_window(meta: dict) -> dict:
    event_start = _parse_iso_timestamp(meta.get("event_time_start"))
    event_end = _parse_iso_timestamp(meta.get("event_time_end"))
    event_at = _parse_iso_timestamp(meta.get("event_at"))
    source_modified_at = _parse_iso_timestamp(meta.get("source_modified_at"))
    source_created_at = _parse_iso_timestamp(meta.get("source_created_at"))
    filed_at = _parse_iso_timestamp(meta.get("filed_at"))

    has_event_fields = any(meta.get(key) for key in ("event_time_start", "event_time_end", "event_at"))
    start = event_start or event_at or event_end
    end = event_end or event_at or event_start

    if start is None and end is None:
        fallback = source_modified_at or source_created_at
        if fallback is not None:
            start = fallback
            end = fallback
    if start is None and end is None and filed_at is not None:
        start = filed_at
        end = filed_at
    if start is None or end is None:
        return {
            "start": None,
            "end": None,
            "event_at": event_at,
            "confidence": "low",
            "timestamp_source": meta.get("timestamp_source"),
        }

    confidence, timestamp_source = _coerce_confidence(meta, has_event_fields)
    return {
        "start": min(start, end),
        "end": max(start, end),
        "event_at": event_at or end,
        "confidence": confidence,
        "timestamp_source": timestamp_source or ("filed_at" if filed_at is not None else None),
    }


def _matches_time_window(
    record_start: datetime,
    record_end: datetime,
    time_from: datetime | None,
    time_to: datetime | None,
) -> bool:
    if time_from and record_end < time_from:
        return False
    if time_to and record_start >= time_to:
        return False
    return True


def _basename(path: str | None) -> str:
    return Path(path).name if path else "unknown"


def _extract_sentences(text: str, limit: int = 4) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split((text or "").strip()):
        cleaned = " ".join(part.strip().split())
        if cleaned:
            sentences.append(cleaned)
        if len(sentences) >= limit:
            break
    return sentences


def _collect_keyword_sentences(records: list[dict], keywords: tuple[str, ...], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    for record in records:
        for sentence in _extract_sentences(record["text"], limit=6):
            lowered = sentence.lower()
            if not any(keyword in lowered for keyword in keywords):
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            collected.append(sentence)
            if len(collected) >= limit:
                return collected
    return collected


def _build_summary(records: list[dict], limit: int = 2) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for record in records:
        for sentence in _extract_sentences(record["text"], limit=3):
            if sentence in seen:
                continue
            seen.add(sentence)
            parts.append(sentence)
            if len(parts) >= limit:
                return " | ".join(parts)
    return ""


def _tokenize_task_text(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in _TASK_TOKEN_RE.finditer(text)]


def _meaningful_task_tokens(text: str | None) -> list[str]:
    tokens: list[str] = []
    for token in _tokenize_task_text(text):
        if len(token) < 3 and token not in _SHORT_TASK_TOKENS:
            continue
        if token in _TASK_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _normalize_task_hint(value: str | None) -> str | None:
    tokens = _meaningful_task_tokens(value)
    if not tokens:
        return None
    normalized: list[str] = []
    for token in tokens:
        if token not in normalized:
            normalized.append(token)
        if len(normalized) >= 4:
            break
    return "-".join(normalized) if normalized else None


def derive_task_hint(text: str | None, preferred_terms: set[str] | None = None) -> str | None:
    tokens = _meaningful_task_tokens(text)
    if not tokens:
        return None

    preferred = preferred_terms or set()
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, index)

    ranked = sorted(
        counts,
        key=lambda token: (
            token in preferred,
            "_" in token or "-" in token,
            counts[token],
            len(token),
            -first_seen[token],
        ),
        reverse=True,
    )
    if not ranked:
        return None

    primary = ranked[0]
    if primary in preferred or "_" in primary or "-" in primary:
        return primary

    selected = [primary]
    for token in ranked[1:]:
        if token in selected:
            continue
        selected.append(token)
        if len(selected) >= 2:
            break
    return "-".join(selected)


def _task_group_key(record: dict, preferred_terms: set[str] | None) -> tuple[str, str] | None:
    meta = record["metadata"]
    explicit_hint = _normalize_task_hint(meta.get("task_hint") or meta.get("task"))
    if explicit_hint:
        return f"task:hint:{explicit_hint}", f"Task {explicit_hint}"

    if not preferred_terms:
        return None

    derived_hint = derive_task_hint(record["text"], preferred_terms=preferred_terms)
    if derived_hint:
        room = meta.get("room") or "general"
        return f"task:derived:{room}:{derived_hint}", f"Task {derived_hint}"

    return None


def _group_key(record: dict, group_by: str, preferred_terms: set[str] | None = None) -> tuple[str, str]:
    meta = record["metadata"]
    source_session_id = meta.get("source_session_id")
    source_file = meta.get("source_file") or ""
    day_key = record["time_start"].date().isoformat()

    if group_by == "session":
        if source_session_id:
            return f"session:{source_session_id}", f"Session {source_session_id}"
        return f"source:{source_file}", _basename(source_file)
    if group_by == "source_file":
        return f"source:{source_file}", _basename(source_file)
    if group_by == "day":
        return f"day:{day_key}", day_key

    task_group = _task_group_key(record, preferred_terms)
    if task_group:
        return task_group
    if source_session_id:
        return f"task:session:{source_session_id}", f"Session {source_session_id}"
    if source_file:
        return f"task:source:{source_file}", _basename(source_file)
    return f"task:day:{day_key}", day_key


def _build_group_payload(
    group_id: str,
    title: str,
    records: list[dict],
    expand_level: str,
    limit_evidence_per_group: int,
) -> dict:
    sorted_records = sorted(records, key=lambda item: (item["query_score"], item["time_start"]), reverse=True)
    time_start = min(record["time_start"] for record in records)
    time_end = max(record["time_end"] for record in records)
    confidence_levels = sorted(
        {record["confidence"] for record in records if record["confidence"] in _VALID_CONFIDENCE}
    )
    session_ids = {
        record["metadata"].get("source_session_id")
        for record in records
        if record["metadata"].get("source_session_id")
    }
    sources = {
        record["metadata"].get("source_file")
        for record in records
        if record["metadata"].get("source_file")
    }

    payload = {
        "group_id": group_id,
        "title": title,
        "time_range": {
            "start": time_start.isoformat(),
            "end": time_end.isoformat(),
        },
        "summary": _build_summary(sorted_records),
        "record_count": len(records),
        "session_count": len(session_ids),
        "source_count": len(sources),
        "evidence_count": len(records),
        "confidence_levels": confidence_levels,
    }

    if expand_level in {"grouped", "evidence"}:
        next_steps = _collect_keyword_sentences(sorted_records, _NEXT_STEP_KEYWORDS, limit=1)
        payload.update(
            {
                "progress": _collect_keyword_sentences(sorted_records, _DELIVERABLE_KEYWORDS, limit=3)
                or _extract_sentences(sorted_records[0]["text"], limit=3),
                "decisions": _collect_keyword_sentences(sorted_records, _DECISION_KEYWORDS, limit=3),
                "blockers": _collect_keyword_sentences(sorted_records, _BLOCKER_KEYWORDS, limit=3),
                "deliverables": _collect_keyword_sentences(sorted_records, _DELIVERABLE_KEYWORDS, limit=3),
                "next_step": next_steps[0] if next_steps else "",
            }
        )

    if expand_level == "evidence":
        evidence = []
        for record in sorted_records[:limit_evidence_per_group]:
            evidence.append(
                {
                    "drawer_id": record["drawer_id"],
                    "record_kind": record["record_kind"],
                    "room": record["metadata"].get("room", "unknown"),
                    "source_file": record["metadata"].get("source_file", ""),
                    "source_session_id": record["metadata"].get("source_session_id"),
                    "timestamp_source": record["timestamp_source"],
                    "confidence": record["confidence"],
                    "time_range": {
                        "start": record["time_start"].isoformat(),
                        "end": record["time_end"].isoformat(),
                    },
                    "query_score": round(record["query_score"], 3),
                    "text": record["text"],
                }
            )
        payload["evidence"] = evidence

    payload["_sort_key"] = (
        max(record["query_score"] for record in records),
        time_start,
    )
    return payload


def _build_record(
    drawer_id: str,
    document: str,
    metadata: dict,
    time_from: datetime | None,
    time_to: datetime | None,
    wing: str | None,
    rooms: list[str],
    record_kinds: list[str],
    agents: list[str],
    session_ids: list[str],
    include_low_confidence: bool,
) -> dict | None:
    metadata = metadata or {}
    if wing and metadata.get("wing") != wing:
        return None

    room = metadata.get("room")
    if rooms and room not in rooms:
        return None

    record_kind = metadata.get("record_kind", "drawer")
    if record_kinds and record_kind not in record_kinds:
        return None

    added_by = metadata.get("added_by") or metadata.get("agent")
    if agents and added_by not in agents:
        return None

    source_session_id = metadata.get("source_session_id")
    if session_ids and (not source_session_id or source_session_id.lower() not in session_ids):
        return None

    resolved = _resolve_event_window(metadata)
    record_start = resolved["start"]
    record_end = resolved["end"]
    if record_start is None or record_end is None:
        if not include_low_confidence:
            return None
    elif not _matches_time_window(record_start, record_end, time_from, time_to):
        return None

    if resolved["confidence"] == "low" and not include_low_confidence:
        return None

    return {
        "drawer_id": drawer_id,
        "text": document,
        "metadata": metadata,
        "record_kind": record_kind,
        "confidence": resolved["confidence"],
        "timestamp_source": resolved["timestamp_source"],
        "time_start": record_start,
        "time_end": record_end,
        "query_score": 0.0,
    }


def _rebuild_event_index(palace_path: str, collection) -> str:
    rows: list[dict] = []
    offset = 0
    source_count = collection.count()
    while True:
        batch = collection.get(include=["metadatas"], limit=_BATCH_SIZE, offset=offset)
        ids = batch.get("ids") or []
        if not ids:
            break
        metadatas = batch.get("metadatas") or []
        for drawer_id, metadata in zip(ids, metadatas):
            metadata = metadata or {}
            resolved = _resolve_event_window(metadata)
            rows.append(
                {
                    "drawer_id": drawer_id,
                    "wing": metadata.get("wing"),
                    "room": metadata.get("room"),
                    "record_kind": metadata.get("record_kind", "drawer"),
                    "added_by": metadata.get("added_by") or metadata.get("agent"),
                    "source_file": metadata.get("source_file"),
                    "source_session_id": metadata.get("source_session_id"),
                    "task_hint": metadata.get("task_hint"),
                    "timestamp_source": resolved["timestamp_source"],
                    "confidence": resolved["confidence"],
                    "event_start": resolved["start"].isoformat() if resolved["start"] else None,
                    "event_end": resolved["end"].isoformat() if resolved["end"] else None,
                    "event_at": resolved["event_at"].isoformat() if resolved["event_at"] else None,
                }
            )
        offset += len(ids)
    return replace_event_index(
        palace_path,
        rows,
        source_count=source_count,
        source_db_mtime=get_source_db_mtime(palace_path),
    )


def _iter_records_by_ids(
    collection,
    drawer_ids: list[str],
    time_from: datetime | None,
    time_to: datetime | None,
    wing: str | None,
    rooms: list[str],
    record_kinds: list[str],
    agents: list[str],
    session_ids: list[str],
    include_low_confidence: bool,
):
    for start in range(0, len(drawer_ids), _BATCH_SIZE):
        batch_ids = drawer_ids[start : start + _BATCH_SIZE]
        batch = collection.get(ids=batch_ids, include=["documents", "metadatas"])
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        for drawer_id, document, metadata in zip(ids, documents, metadatas):
            record = _build_record(
                drawer_id=drawer_id,
                document=document,
                metadata=metadata,
                time_from=time_from,
                time_to=time_to,
                wing=wing,
                rooms=rooms,
                record_kinds=record_kinds,
                agents=agents,
                session_ids=session_ids,
                include_low_confidence=include_low_confidence,
            )
            if record is not None:
                yield record


def _iter_filtered_records(
    palace_path: str,
    time_from: datetime | None,
    time_to: datetime | None,
    wing: str | None,
    rooms: list[str],
    record_kinds: list[str],
    agents: list[str],
    session_ids: list[str],
    include_low_confidence: bool,
    collection=None,
):
    if collection is None:
        try:
            collection = get_collection(palace_path, create=False)
        except Exception as exc:
            raise SearchError(f"No palace found at {palace_path}") from exc

    where = {"wing": wing} if wing else None
    offset = 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=_BATCH_SIZE, offset=offset, where=where)
        ids = batch.get("ids") or []
        if not ids:
            break
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        for drawer_id, document, metadata in zip(ids, documents, metadatas):
            record = _build_record(
                drawer_id=drawer_id,
                document=document,
                metadata=metadata,
                time_from=time_from,
                time_to=time_to,
                wing=wing,
                rooms=rooms,
                record_kinds=record_kinds,
                agents=agents,
                session_ids=session_ids,
                include_low_confidence=include_low_confidence,
            )
            if record is not None:
                yield record
        offset += len(ids)


def search_events(
    palace_path: str,
    time_from: str | None,
    time_to: str | None,
    query: str | None = None,
    wing: str | None = None,
    rooms=None,
    record_kinds=None,
    agents=None,
    session_ids=None,
    timezone_name: str | None = None,
    group_by: str = "task",
    expand_level: str = "overview",
    limit_groups: int = 10,
    limit_evidence_per_group: int = 3,
    include_low_confidence: bool = False,
) -> dict:
    if group_by not in _VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of: {', '.join(sorted(_VALID_GROUP_BY))}")
    if expand_level not in _VALID_EXPAND_LEVELS:
        raise ValueError(f"expand_level must be one of: {', '.join(sorted(_VALID_EXPAND_LEVELS))}")

    parsed_rooms = _parse_list(rooms)
    parsed_record_kinds = _parse_list(record_kinds) or list(_DEFAULT_RECORD_KINDS)
    parsed_agents = _parse_list(agents)
    parsed_session_ids = [session_id.lower() for session_id in _parse_list(session_ids)]
    query_timezone, resolved_timezone_name = _resolve_query_timezone(timezone_name)
    parsed_time_from, parsed_time_to = _normalize_time_range(time_from, time_to, query_timezone)

    raw_query = (query or "").strip()
    query_info = sanitize_query(raw_query) if raw_query else None
    clean_query = query_info["clean_query"] if query_info else ""
    inferred_session_ids = _extract_session_ids(clean_query)
    for session_id in inferred_session_ids:
        if session_id not in parsed_session_ids:
            parsed_session_ids.append(session_id)
    preferred_terms = set(_meaningful_task_tokens(clean_query))
    try:
        collection = get_collection(palace_path, create=False)
    except Exception as exc:
        raise SearchError(f"No palace found at {palace_path}") from exc

    index_backend = "collection_scan"
    candidate_records = 0
    records: list[dict]
    try:
        source_count = collection.count()
        if is_event_index_stale(palace_path, source_count):
            _rebuild_event_index(palace_path, collection)

        candidate_ids = query_event_index(
            palace_path,
            time_from=parsed_time_from.isoformat() if parsed_time_from else None,
            time_to=parsed_time_to.isoformat() if parsed_time_to else None,
            wing=wing,
            rooms=parsed_rooms,
            record_kinds=parsed_record_kinds,
            agents=parsed_agents,
            session_ids=parsed_session_ids,
            include_low_confidence=include_low_confidence,
        )
        candidate_records = len(candidate_ids)
        records = list(
            _iter_records_by_ids(
                collection=collection,
                drawer_ids=candidate_ids,
                time_from=parsed_time_from,
                time_to=parsed_time_to,
                wing=wing,
                rooms=parsed_rooms,
                record_kinds=parsed_record_kinds,
                agents=parsed_agents,
                session_ids=parsed_session_ids,
                include_low_confidence=include_low_confidence,
            )
        )
        index_backend = "sidecar"
    except Exception:
        records = list(
            _iter_filtered_records(
                palace_path=palace_path,
                time_from=parsed_time_from,
                time_to=parsed_time_to,
                wing=wing,
                rooms=parsed_rooms,
                record_kinds=parsed_record_kinds,
                agents=parsed_agents,
                session_ids=parsed_session_ids,
                include_low_confidence=include_low_confidence,
                collection=collection,
            )
        )
        candidate_records = len(records)

    if clean_query and records:
        bm25_scores = _bm25_scores(clean_query, [record["text"] for record in records])
        max_score = max(bm25_scores) if bm25_scores else 0.0
        for record, score in zip(records, bm25_scores):
            record["query_score"] = (score / max_score) if max_score > 0 else 0.0

    groups: dict[str, list[dict]] = defaultdict(list)
    titles: dict[str, str] = {}
    for record in records:
        group_id, title = _group_key(record, group_by, preferred_terms=preferred_terms)
        groups[group_id].append(record)
        titles[group_id] = title

    payload_groups = [
        _build_group_payload(group_id, titles[group_id], grouped_records, expand_level, limit_evidence_per_group)
        for group_id, grouped_records in groups.items()
    ]
    payload_groups.sort(key=lambda group: group["_sort_key"], reverse=True)
    for group in payload_groups:
        group.pop("_sort_key", None)
    payload_groups = payload_groups[:limit_groups]

    result = {
        "query": clean_query or None,
        "time_range": {
            "from": parsed_time_from.isoformat() if parsed_time_from else None,
            "to": parsed_time_to.isoformat() if parsed_time_to else None,
        },
        "filters": {
            "wing": wing,
            "rooms": parsed_rooms,
            "record_kinds": parsed_record_kinds,
            "agents": parsed_agents,
            "session_ids": parsed_session_ids,
            "timezone": resolved_timezone_name,
            "group_by": group_by,
            "expand_level": expand_level,
            "include_low_confidence": include_low_confidence,
        },
        "stats": {
            "matched_records": len(records),
            "candidate_records": candidate_records,
            "returned_groups": len(payload_groups),
            "index_backend": index_backend,
        },
        "groups": payload_groups,
    }
    if query_info and query_info["was_sanitized"]:
        result["query_sanitized"] = True
        result["sanitizer"] = {
            "method": query_info["method"],
            "original_length": query_info["original_length"],
            "clean_length": query_info["clean_length"],
            "clean_query": query_info["clean_query"],
        }
    return result


def print_event_search(result: dict) -> None:
    query = result.get("query")
    time_range = result.get("time_range", {})
    print(f"\n{'=' * 60}")
    print("  Event Search")
    print(f"{'=' * 60}")
    if query:
        print(f'  Query: "{query}"')
    if time_range.get("from") or time_range.get("to"):
        print(f"  Range: {time_range.get('from') or '*'} -> {time_range.get('to') or '*'}")
    print()

    groups = result.get("groups") or []
    if not groups:
        print("  No event groups matched.")
        print()
        return

    for index, group in enumerate(groups, start=1):
        print(f"  [{index}] {group['title']}")
        print(f"      Time:   {group['time_range']['start']} -> {group['time_range']['end']}")
        print(
            "      Counts: "
            f"{group['record_count']} records, {group['session_count']} sessions, {group['source_count']} sources"
        )
        if group.get("summary"):
            print(f"      Summary: {group['summary']}")

        if "progress" in group and group["progress"]:
            print("      Progress:")
            for item in group["progress"]:
                print(f"        - {item}")
        if "decisions" in group and group["decisions"]:
            print("      Decisions:")
            for item in group["decisions"]:
                print(f"        - {item}")
        if "blockers" in group and group["blockers"]:
            print("      Blockers:")
            for item in group["blockers"]:
                print(f"        - {item}")
        if group.get("next_step"):
            print(f"      Next: {group['next_step']}")

        for evidence in group.get("evidence", []):
            print(
                "      Evidence: "
                f"{evidence['record_kind']} {evidence['time_range']['start']} {evidence['source_file']}"
            )
            for line in evidence["text"].strip().splitlines()[:6]:
                print(f"        {line}")
        print()
