import json
import os
import tempfile
import shutil
from pathlib import Path

import chromadb

from mempalace.convo_miner import mine_convos
from mempalace.palace import file_already_mined


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine_convos(tmpdir, palace_path, wing="test_convos")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    assert col.count() >= 2

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_short_files(capsys):
    """Files below MIN_CHUNK_SIZE get a sentinel so they are skipped on re-run."""
    tmpdir = tempfile.mkdtemp()
    try:
        # A file too short to produce any chunks
        with open(os.path.join(tmpdir, "tiny.txt"), "w") as f:
            f.write("hi")

        palace_path = os.path.join(tmpdir, "palace")

        # First run -- file is processed (sentinel written)
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()  # drain output

        # Verify sentinel was written (resolve path -- macOS /var -> /private/var)
        resolved_file = str(Path(tmpdir).resolve() / "tiny.txt")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        assert file_already_mined(col, resolved_file)

        # Second run -- file should be skipped
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_empty_chunk_files(capsys):
    """Files that normalize but produce 0 exchange chunks get a sentinel."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Content long enough to pass MIN_CHUNK_SIZE but with no exchange markers
        # (no "> " lines), so chunk_exchanges returns []
        with open(os.path.join(tmpdir, "no_exchanges.txt"), "w") as f:
            f.write("This is a plain paragraph without any exchange markers. " * 5)

        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_rebuilds_stale_drawers_after_schema_bump(capsys):
    """When stored drawers have an older normalize_version, the next mine
    silently purges them and refiles — no manual erase required.

    This is what makes the strip_noise upgrade apply to existing corpora:
    users just run `mempalace mine` again and old noise-filled drawers get
    replaced with clean ones."""
    from mempalace.palace import NORMALIZE_VERSION

    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What is memory?\nMemory is persistence.\n\n"
            "> Why does it matter?\nIt enables continuity.\n\n"
            "> How do we build it?\nWith structured storage.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        # First mine — stamps drawers with NORMALIZE_VERSION
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        first_pass = col.get(where={"source_file": resolved})
        first_ids = set(first_pass["ids"])
        assert first_ids, "first mine should produce drawers"
        stored_version = max(meta.get("normalize_version", 0) for meta in first_pass["metadatas"])
        assert stored_version >= NORMALIZE_VERSION

        # Simulate pre-v2 drawers: rewrite metadata to an older version,
        # and replace content with "noise" so we can see it get cleaned up.
        stale_metas = []
        for meta in first_pass["metadatas"]:
            stale = dict(meta)
            stale["normalize_version"] = 1
            stale_metas.append(stale)
        col.update(
            ids=list(first_pass["ids"]),
            documents=["STALE NOISE"] * len(first_pass["ids"]),
            metadatas=stale_metas,
        )
        # Add an extra orphan drawer that should also be purged.
        col.add(
            ids=["orphan_drawer"],
            documents=["OLD ORPHAN"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "default",
                    "source_file": resolved,
                    "chunk_index": 999,
                    "normalize_version": 1,
                }
            ],
        )
        del col, client

        # Second mine — version gate should trigger rebuild
        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert (
            "Files skipped (already filed): 0" in out
        ), "stale drawers should force a rebuild, not a skip"

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        rebuilt = col.get(where={"source_file": resolved})
        # Orphan is gone
        assert "orphan_drawer" not in rebuilt["ids"]
        # No stale content survived
        assert all("STALE NOISE" not in d for d in rebuilt["documents"])
        assert all("OLD ORPHAN" not in d for d in rebuilt["documents"])
        # All rebuilt drawers carry the current version
        for meta in rebuilt["metadatas"]:
            assert meta.get("normalize_version", 0) >= NORMALIZE_VERSION
        del col, client
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_updates_changed_codex_session():
    tmpdir = tempfile.mkdtemp()
    try:
        convo = Path(tmpdir) / "chat.jsonl"
        convo.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session_meta", "payload": {"id": "sess-1"}}),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-17T10:00:00Z",
                            "payload": {
                                "type": "user_message",
                                "message": "先看一下这段会话记录，并确认第一轮到底发生了哪些关键步骤。",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-17T10:01:00Z",
                            "payload": {
                                "type": "agent_message",
                                "message": "已经读取第一轮，并整理出背景、执行动作和当前结论。",
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(convo.resolve())
        first_pass = col.get(where={"source_file": resolved})
        transcript_rows = [
            (doc, meta)
            for doc, meta in zip(first_pass["documents"], first_pass["metadatas"])
            if (meta or {}).get("record_kind") == "transcript"
        ]
        assert len(transcript_rows) == 1
        assert transcript_rows[0][1]["event_at"] == "2026-04-17T10:01:00+00:00"
        assert transcript_rows[0][1]["source_message_start_idx"] == 0
        assert transcript_rows[0][1]["source_message_end_idx"] == 1

        with convo.open("a", encoding="utf-8") as f:
            f.write(
                "\n"
                + json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-04-17T10:02:00Z",
                        "payload": {
                            "type": "user_message",
                            "message": "继续补第二轮，把新增的决策、问题和交付物也一起补齐。",
                        },
                    }
                )
            )
            f.write(
                "\n"
                + json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-04-17T10:03:00Z",
                        "payload": {
                            "type": "agent_message",
                            "message": "第二轮也已记录，新增内容已经按照时间顺序写回 transcript。",
                        },
                    }
                )
            )

        mine_convos(tmpdir, palace_path, wing="test")
        second_pass = col.get(where={"source_file": resolved})
        transcript_rows = [
            (doc, meta)
            for doc, meta in zip(second_pass["documents"], second_pass["metadatas"])
            if (meta or {}).get("record_kind") == "transcript"
        ]
        docs = [doc for doc, _ in transcript_rows]
        metas = [meta for _, meta in transcript_rows]

        assert len(transcript_rows) == 2
        assert any("第二轮也已记录" in doc for doc in docs)
        assert any(meta["event_at"] == "2026-04-17T10:03:00+00:00" for meta in metas)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_general_mode_keeps_transcript_and_memories():
    tmpdir = tempfile.mkdtemp()
    try:
        convo = Path(tmpdir) / "chat.jsonl"
        convo.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session_meta", "payload": {"id": "sess-2"}}),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-17T12:00:00Z",
                            "payload": {
                                "type": "user_message",
                                "message": "We decided to ship a nightly backup because the old process kept failing.",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-17T12:01:00Z",
                            "payload": {
                                "type": "agent_message",
                                "message": "Good. I fixed the schedule and the workaround is no longer needed.",
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test", extract_mode="general")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(convo.resolve())
        results = col.get(where={"source_file": resolved})
        kinds = {(meta or {}).get("record_kind") for meta in results["metadatas"]}

        assert "transcript" in kinds
        assert "memory" in kinds
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_writes_task_hint_for_transcript_chunks():
    tmpdir = tempfile.mkdtemp()
    try:
        convo = Path(tmpdir) / "chat.jsonl"
        convo.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session_meta", "payload": {"id": "sess-task-hint"}}),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-18T12:00:00Z",
                            "payload": {
                                "type": "user_message",
                                "message": "Please implement search_events MCP support and evidence expansion.",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-04-18T12:01:00Z",
                            "payload": {
                                "type": "agent_message",
                                "message": "search_events CLI and MCP integration are now wired for review.",
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(convo.resolve())
        results = col.get(where={"source_file": resolved})
        transcript_hints = [
            (meta or {}).get("task_hint")
            for meta in results["metadatas"]
            if (meta or {}).get("record_kind") == "transcript"
        ]

        assert transcript_hints
        assert "search_events" in transcript_hints
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
