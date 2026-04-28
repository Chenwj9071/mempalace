"""Tests for event_search.py."""

from pathlib import Path

from mempalace.event_index import get_event_index_path
from mempalace.event_search import print_event_search, search_events


def test_search_events_filters_by_event_time_not_filed_at(collection, palace_path):
    collection.add(
        ids=[
            "drawer_transcript_old",
            "drawer_transcript_today",
            "drawer_memory_today",
        ],
        documents=[
            "Historical session rebuilt today, but the work actually happened last week.",
            "Implemented search_events MVP and verified grouped output.",
            "Decision: default to overview before expanding evidence.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "planning",
                "source_file": "old-session.jsonl",
                "source_session_id": "sess-old",
                "record_kind": "transcript",
                "added_by": "codex",
                "filed_at": "2026-04-20T10:00:00",
                "event_time_start": "2026-04-12T09:00:00",
                "event_time_end": "2026-04-12T09:30:00",
                "event_at": "2026-04-12T09:30:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "planning",
                "source_file": "today-session.jsonl",
                "source_session_id": "sess-today",
                "record_kind": "transcript",
                "added_by": "codex",
                "filed_at": "2026-04-20T10:30:00",
                "event_time_start": "2026-04-20T09:00:00",
                "event_time_end": "2026-04-20T09:30:00",
                "event_at": "2026-04-20T09:30:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "planning",
                "source_file": "today-session.jsonl",
                "source_session_id": "sess-today",
                "record_kind": "memory",
                "added_by": "codex",
                "filed_at": "2026-04-20T10:35:00",
                "event_time_start": "2026-04-20T09:00:00",
                "event_time_end": "2026-04-20T09:30:00",
                "event_at": "2026-04-20T09:30:00",
                "timestamp_source": "message_timestamp",
            },
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-20",
        time_to="2026-04-20",
    )

    assert result["stats"]["matched_records"] == 2
    assert result["stats"]["returned_groups"] == 1
    assert result["groups"][0]["title"] == "Session sess-today"


def test_search_events_overview_grouped_and_evidence(collection, palace_path):
    collection.add(
        ids=["drawer_1", "drawer_2"],
        documents=[
            "Implemented grouped event search and documented the time window rules.",
            "Next step: wire MCP search_events and expose evidence expansion.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "session-a.jsonl",
                "source_session_id": "sess-a",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-18T11:00:00",
                "event_time_end": "2026-04-18T11:10:00",
                "event_at": "2026-04-18T11:10:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "session-a.jsonl",
                "source_session_id": "sess-a",
                "record_kind": "memory",
                "added_by": "codex",
                "event_time_start": "2026-04-18T11:00:00",
                "event_time_end": "2026-04-18T11:10:00",
                "event_at": "2026-04-18T11:10:00",
                "timestamp_source": "message_timestamp",
            },
        ],
    )

    overview = search_events(
        palace_path=palace_path,
        time_from="2026-04-18",
        time_to="2026-04-18",
        expand_level="overview",
    )
    grouped = search_events(
        palace_path=palace_path,
        time_from="2026-04-18",
        time_to="2026-04-18",
        expand_level="grouped",
    )
    evidence = search_events(
        palace_path=palace_path,
        time_from="2026-04-18",
        time_to="2026-04-18",
        expand_level="evidence",
    )

    assert "evidence" not in overview["groups"][0]
    assert "progress" in grouped["groups"][0]
    assert "evidence" in evidence["groups"][0]
    assert len(evidence["groups"][0]["evidence"]) == 2


def test_search_events_task_group_merges_related_sessions(collection, palace_path):
    collection.add(
        ids=["drawer_task_a", "drawer_task_b"],
        documents=[
            "Implemented search_events MVP and verified grouped output.",
            "Exposed MCP search_events tool with evidence output for follow-up queries.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "session-a.jsonl",
                "source_session_id": "sess-a",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-20T09:00:00",
                "event_time_end": "2026-04-20T09:05:00",
                "event_at": "2026-04-20T09:05:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "backend",
                "source_file": "session-b.jsonl",
                "source_session_id": "sess-b",
                "record_kind": "memory",
                "added_by": "codex",
                "event_time_start": "2026-04-20T10:00:00",
                "event_time_end": "2026-04-20T10:05:00",
                "event_at": "2026-04-20T10:05:00",
                "timestamp_source": "message_timestamp",
            },
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-20",
        time_to="2026-04-20",
        query="search_events",
    )

    assert result["stats"]["matched_records"] == 2
    assert result["stats"]["returned_groups"] == 1
    assert result["groups"][0]["title"] == "Task search_events"
    assert result["groups"][0]["record_count"] == 2
    assert result["groups"][0]["session_count"] == 2


def test_search_events_builds_sidecar_index(collection, palace_path):
    collection.add(
        ids=["drawer_indexed"],
        documents=["Implemented event index sidecar rebuild path."],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "index.jsonl",
                "source_session_id": "sess-index",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-21T08:00:00",
                "event_time_end": "2026-04-21T08:05:00",
                "event_at": "2026-04-21T08:05:00",
                "timestamp_source": "message_timestamp",
            }
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-21",
        time_to="2026-04-21",
    )

    assert result["stats"]["index_backend"] == "sidecar"
    assert result["stats"]["candidate_records"] == 1
    assert Path(get_event_index_path(palace_path)).is_file()


def test_search_events_excludes_low_confidence_by_default(collection, palace_path):
    collection.add(
        ids=["drawer_low_conf"],
        documents=["Fallback only record without reliable event metadata."],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "fallback.txt",
                "record_kind": "memory",
                "added_by": "codex",
                "filed_at": "2026-04-20T08:00:00",
                "event_at": "2026-04-20T08:00:00",
                "timestamp_source": "ingest_time",
            }
        ],
    )

    default_result = search_events(
        palace_path=palace_path,
        time_from="2026-04-20",
        time_to="2026-04-20",
    )
    included_result = search_events(
        palace_path=palace_path,
        time_from="2026-04-20",
        time_to="2026-04-20",
        include_low_confidence=True,
    )

    assert default_result["stats"]["matched_records"] == 0
    assert included_result["stats"]["matched_records"] == 1


def test_search_events_uses_timezone_for_date_boundaries(collection, palace_path):
    collection.add(
        ids=["drawer_shanghai_day", "drawer_next_day"],
        documents=[
            "Shanghai-local 2026-04-27 work item.",
            "Shanghai-local 2026-04-28 work item.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "planning",
                "source_file": "tz-a.jsonl",
                "source_session_id": "sess-tz-a",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-26T20:00:00+00:00",
                "event_time_end": "2026-04-26T20:05:00+00:00",
                "event_at": "2026-04-26T20:05:00+00:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "planning",
                "source_file": "tz-b.jsonl",
                "source_session_id": "sess-tz-b",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-27T16:30:00+00:00",
                "event_time_end": "2026-04-27T16:35:00+00:00",
                "event_at": "2026-04-27T16:35:00+00:00",
                "timestamp_source": "message_timestamp",
            },
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-27",
        time_to="2026-04-27",
        timezone_name="Asia/Shanghai",
        group_by="session",
    )

    assert result["stats"]["matched_records"] == 1
    assert result["groups"][0]["title"] == "Session sess-tz-a"
    assert result["filters"]["timezone"] == "Asia/Shanghai"


def test_search_events_exact_session_filter_ignores_cross_session_mentions(collection, palace_path):
    target_session_id = "019dba1e-2268-7261-aeff-0f346f95d745"
    collection.add(
        ids=["drawer_target", "drawer_reference"],
        documents=[
            "Target session actual transcript content.",
            f"Today we discussed session {target_session_id} in a different conversation.",
        ],
        metadatas=[
            {
                "wing": "project",
                "room": "planning",
                "source_file": "target.jsonl",
                "source_session_id": target_session_id,
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-27T02:00:00+00:00",
                "event_time_end": "2026-04-27T02:05:00+00:00",
                "event_at": "2026-04-27T02:05:00+00:00",
                "timestamp_source": "message_timestamp",
            },
            {
                "wing": "project",
                "room": "planning",
                "source_file": "other.jsonl",
                "source_session_id": "other-session",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-27T03:00:00+00:00",
                "event_time_end": "2026-04-27T03:05:00+00:00",
                "event_at": "2026-04-27T03:05:00+00:00",
                "timestamp_source": "message_timestamp",
            },
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-27",
        time_to="2026-04-27",
        timezone_name="Asia/Shanghai",
        query=target_session_id,
        group_by="session",
        expand_level="evidence",
    )

    assert result["stats"]["matched_records"] == 1
    assert result["groups"][0]["title"] == f"Session {target_session_id}"
    assert result["groups"][0]["evidence"][0]["source_session_id"] == target_session_id


def test_print_event_search_outputs_group_summary(capsys, collection, palace_path):
    collection.add(
        ids=["drawer_print"],
        documents=["Implemented event search output formatting."],
        metadatas=[
            {
                "wing": "project",
                "room": "backend",
                "source_file": "print.jsonl",
                "source_session_id": "sess-print",
                "record_kind": "transcript",
                "added_by": "codex",
                "event_time_start": "2026-04-19T08:00:00",
                "event_time_end": "2026-04-19T08:05:00",
                "event_at": "2026-04-19T08:05:00",
                "timestamp_source": "message_timestamp",
            }
        ],
    )

    result = search_events(
        palace_path=palace_path,
        time_from="2026-04-19",
        time_to="2026-04-19",
    )
    print_event_search(result)
    out = capsys.readouterr().out
    assert "Event Search" in out
    assert "Session sess-print" in out
