#!/usr/bin/env python3
"""
convo_miner.py - Mine conversations into the palace.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .event_search import derive_task_hint
from .normalize import normalize_with_metadata
from .palace import NORMALIZE_VERSION, SKIP_DIRS, get_collection, mine_lock

_HALL_KEYWORDS_CACHE = None

CONVO_EXTENSIONS = {".txt", ".md", ".json", ".jsonl"}
MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800
MAX_FILE_SIZE = 10 * 1024 * 1024
TAIL_REWIND_MESSAGES = 2
REGISTRY_ROOM = "_registry"
REGISTRY_MODE = "registry"
CONVO_NORMALIZE_VERSION = max(NORMALIZE_VERSION, 3)

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def _detect_hall_cached(content: str) -> str:
    global _HALL_KEYWORDS_CACHE
    if _HALL_KEYWORDS_CACHE is None:
        from .config import MempalaceConfig

        _HALL_KEYWORDS_CACHE = MempalaceConfig().hall_keywords
    content_lower = content[:3000].lower()
    scores = {}
    for hall, keywords in _HALL_KEYWORDS_CACHE.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            scores[hall] = score
    return max(scores, key=scores.get) if scores else "general"


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_from_epoch(epoch_value):
    try:
        return datetime.fromtimestamp(float(epoch_value)).astimezone().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _get_file_state(filepath: Path) -> dict:
    stat = filepath.stat()
    return {
        "source_size": stat.st_size,
        "source_mtime": float(stat.st_mtime),
        "source_created_at": _iso_from_epoch(stat.st_ctime),
        "source_modified_at": _iso_from_epoch(stat.st_mtime),
    }


def _registry_id(source_file: str) -> str:
    return f"_reg_{hashlib.sha256(source_file.encode('utf-8')).hexdigest()[:24]}"


def _drawer_id(source_file: str, room: str, record_kind: str, chunk_key: str) -> str:
    digest = hashlib.sha256(f"{source_file}|{room}|{record_kind}|{chunk_key}".encode("utf-8")).hexdigest()[:24]
    return f"drawer_{record_kind}_{digest}"


def _message_digest(message: dict) -> str:
    payload = {
        "role": message.get("role", ""),
        "text": message.get("text", ""),
        "event_time_start": message.get("event_time_start"),
        "event_time_end": message.get("event_time_end"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _transcript_digest(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def scan_convos(convo_dir: str) -> list:
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() not in CONVO_EXTENSIONS or filepath.is_symlink():
                continue
            try:
                if filepath.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            files.append(filepath)
    return files


def chunk_exchanges(content: str) -> list:
    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return _chunk_by_exchange(lines)
    return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: list) -> list:
    chunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1
            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            content = user_turn
            if ai_lines:
                content = f"{content}\n{' '.join(ai_lines)}"
            if len(content) <= CHUNK_SIZE:
                if len(content.strip()) > MIN_CHUNK_SIZE:
                    chunks.append({"content": content, "chunk_index": len(chunks)})
            else:
                remainder = content
                part_index = 0
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    if len(part.strip()) > MIN_CHUNK_SIZE:
                        chunks.append(
                            {
                                "content": part,
                                "chunk_index": len(chunks),
                                "chunk_key": f"legacy_{len(chunks)}_{part_index}",
                            }
                        )
                    part_index += 1
        else:
            i += 1
    return chunks


def _chunk_by_paragraph(content: str) -> list:
    chunks = []
    paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                chunks.append({"content": group, "chunk_index": len(chunks)})
        return chunks

    for paragraph in paragraphs:
        if len(paragraph) > MIN_CHUNK_SIZE:
            chunks.append({"content": paragraph, "chunk_index": len(chunks)})
    return chunks


def detect_convo_room(content: str) -> str:
    content_lower = content[:3000].lower()
    scores = {}
    for room, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            scores[room] = score
    return max(scores, key=scores.get) if scores else "general"


def _append_split_chunk(
    chunks: list,
    content: str,
    chunk_index_base: int,
    start_seq: int,
    end_seq: int,
    start_time,
    end_time,
    record_kind: str,
):
    remainder = content
    part_index = 0
    while remainder:
        part = remainder[:CHUNK_SIZE]
        remainder = remainder[CHUNK_SIZE:]
        if len(part.strip()) <= MIN_CHUNK_SIZE:
            part_index += 1
            continue
        chunks.append(
            {
                "content": part,
                "chunk_index": chunk_index_base + len(chunks),
                "chunk_key": f"{record_kind}_{start_seq}_{end_seq}_{part_index}",
                "source_message_start_idx": start_seq,
                "source_message_end_idx": end_seq,
                "event_time_start": start_time,
                "event_time_end": end_time or start_time,
                "event_at": end_time or start_time,
                "record_kind": record_kind,
            }
        )
        part_index += 1


def _chunk_message_exchanges(messages: list, chunk_index_base: int = 0) -> list:
    chunks = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") != "user":
            i += 1
            continue

        start_seq = _safe_int(message.get("seq"), i)
        end_seq = start_seq
        start_time = message.get("event_time_start") or message.get("event_time_end")
        end_time = message.get("event_time_end") or start_time
        user_turn = message.get("text", "").strip()
        i += 1

        assistant_parts = []
        while i < len(messages) and messages[i].get("role") != "user":
            text = messages[i].get("text", "").strip()
            if text:
                assistant_parts.append(text)
            end_seq = _safe_int(messages[i].get("seq"), end_seq)
            end_time = messages[i].get("event_time_end") or messages[i].get("event_time_start") or end_time
            i += 1

        content = f"> {user_turn}" if user_turn else ""
        if assistant_parts:
            content = content + "\n" + "\n".join(assistant_parts) if content else "\n".join(assistant_parts)
        if len(content.strip()) <= MIN_CHUNK_SIZE:
            continue
        _append_split_chunk(
            chunks,
            content,
            chunk_index_base,
            start_seq,
            end_seq,
            start_time,
            end_time,
            "transcript",
        )
    return chunks


def _extract_general_chunks(normalized_data: dict, chunk_index_base: int = 0) -> list:
    from .general_extractor import extract_memories

    transcript = normalized_data.get("transcript", "")
    if not transcript:
        return []
    memories = extract_memories(transcript)
    if not memories:
        return []

    messages = normalized_data.get("messages", [])
    if messages:
        start_seq = _safe_int(messages[0].get("seq"), 0)
        end_seq = _safe_int(messages[-1].get("seq"), start_seq)
        start_time = messages[0].get("event_time_start") or messages[0].get("event_time_end")
        end_time = messages[-1].get("event_time_end") or messages[-1].get("event_time_start") or start_time
        timestamp_source = "message_timestamp" if end_time or start_time else None
    else:
        start_seq = -1
        end_seq = -1
        start_time = None
        end_time = None
        timestamp_source = None

    chunks = []
    for idx, memory in enumerate(memories):
        chunk = dict(memory)
        chunk["chunk_index"] = chunk_index_base + idx
        chunk["chunk_key"] = f"memory_{start_seq}_{end_seq}_{idx}"
        chunk["source_message_start_idx"] = start_seq
        chunk["source_message_end_idx"] = end_seq
        chunk["event_time_start"] = start_time
        chunk["event_time_end"] = end_time
        chunk["event_at"] = end_time or start_time
        chunk["timestamp_source"] = timestamp_source
        chunk["record_kind"] = "memory"
        chunks.append(chunk)
    return chunks


def _stamp_fallback_chunks(chunks: list, file_state: dict, chunk_index_base: int = 0, record_kind: str = "transcript") -> list:
    event_at = file_state.get("source_modified_at")
    timestamp_source = "file_mtime" if event_at else "ingest_time"
    stamped = []
    for idx, chunk in enumerate(chunks):
        row = dict(chunk)
        row["chunk_index"] = chunk_index_base + idx
        row["chunk_key"] = row.get("chunk_key", f"{record_kind}_{chunk_index_base + idx}")
        row["source_message_start_idx"] = row.get("source_message_start_idx", -1)
        row["source_message_end_idx"] = row.get("source_message_end_idx", -1)
        row["event_time_start"] = row.get("event_time_start", event_at)
        row["event_time_end"] = row.get("event_time_end", event_at)
        row["event_at"] = row.get("event_at", event_at)
        row["timestamp_source"] = row.get("timestamp_source", timestamp_source)
        row["record_kind"] = row.get("record_kind", record_kind)
        stamped.append(row)
    return stamped


def _build_transcript_chunks(normalized_data: dict, file_state: dict, chunk_index_base: int = 0, messages_override=None) -> list:
    messages = messages_override if messages_override is not None else normalized_data.get("messages", [])
    if messages:
        chunks = _chunk_message_exchanges(messages, chunk_index_base=chunk_index_base)
        for chunk in chunks:
            chunk["timestamp_source"] = "message_timestamp" if chunk.get("event_at") else "file_mtime"
        return chunks
    transcript = normalized_data.get("transcript", "")
    return _stamp_fallback_chunks(chunk_exchanges(transcript), file_state, chunk_index_base)


def _load_source_state(collection, source_file: str):
    try:
        results = collection.get(where={"source_file": source_file}, include=["metadatas"])
    except Exception:
        return None, []
    registry = None
    drawers = []
    for row_id, metadata in zip(results.get("ids", []) or [], results.get("metadatas", []) or []):
        metadata = metadata or {}
        if metadata.get("room") == REGISTRY_ROOM and metadata.get("ingest_mode") == REGISTRY_MODE:
            registry = {"id": row_id, "metadata": metadata}
        else:
            drawers.append({"id": row_id, "metadata": metadata})
    return registry, drawers


def _delete_source(collection, source_file: str):
    try:
        collection.delete(where={"source_file": source_file})
    except Exception:
        pass


def _delete_tail_drawers(collection, drawers: list, overlap_start: int):
    ids = []
    for drawer in drawers:
        metadata = drawer.get("metadata") or {}
        end_idx = metadata.get("source_message_end_idx")
        if end_idx is None:
            continue
        if _safe_int(end_idx, -1) >= overlap_start:
            ids.append(drawer["id"])
    if ids:
        collection.delete(ids=ids)


def _last_chunk_index(drawers: list) -> int:
    last_index = -1
    for drawer in drawers:
        metadata = drawer.get("metadata") or {}
        last_index = max(last_index, _safe_int(metadata.get("chunk_index"), -1))
    return last_index


def _registry_matches_file(registry_meta: dict, file_state: dict, transcript_digest: str) -> bool:
    if not registry_meta:
        return False
    if _safe_int(registry_meta.get("normalize_version"), 0) < CONVO_NORMALIZE_VERSION:
        return False
    stored_size = _safe_int(registry_meta.get("source_size"), -1)
    stored_mtime = registry_meta.get("source_modified_at")
    return (
        stored_size == _safe_int(file_state.get("source_size"), -2)
        and stored_mtime == file_state.get("source_modified_at")
        and registry_meta.get("transcript_digest", "") == transcript_digest
    )


def _plan_exchange_update(registry_meta: dict, drawers: list, messages: list, transcript_digest: str, file_state: dict):
    if registry_meta is None:
        return {"mode": "full", "rewind_from": 0}
    if _safe_int(registry_meta.get("normalize_version"), 0) < CONVO_NORMALIZE_VERSION:
        return {"mode": "full", "rewind_from": 0}

    stored_size = _safe_int(registry_meta.get("source_size"), -1)
    current_size = _safe_int(file_state.get("source_size"), -1)
    stored_count = _safe_int(registry_meta.get("message_count"), 0)

    if current_size < stored_size or len(messages) < stored_count:
        return {"mode": "full", "rewind_from": 0}
    if len(messages) == stored_count:
        if registry_meta.get("transcript_digest", "") == transcript_digest:
            return {"mode": "noop", "rewind_from": 0}
        return {"mode": "full", "rewind_from": 0}

    stored_last_digest = registry_meta.get("last_message_digest", "")
    if stored_count > 0:
        boundary = messages[stored_count - 1]
        if not stored_last_digest or _message_digest(boundary) != stored_last_digest:
            return {"mode": "full", "rewind_from": 0}

    if any("source_message_end_idx" not in (drawer.get("metadata") or {}) for drawer in drawers):
        return {"mode": "full", "rewind_from": 0}

    rewind_from = max(0, stored_count - TAIL_REWIND_MESSAGES)
    return {"mode": "incremental", "rewind_from": rewind_from}


def _build_registry_metadata(
    source_file: str,
    wing: str,
    agent: str,
    file_state: dict,
    normalized_data: dict,
    transcript_digest: str,
    last_chunk_index: int,
    extract_mode: str,
):
    messages = normalized_data.get("messages", []) or []
    filed_at = datetime.now().isoformat()
    metadata = {
        "wing": wing,
        "room": REGISTRY_ROOM,
        "source_file": source_file,
        "added_by": agent,
        "filed_at": filed_at,
        "ingest_mode": REGISTRY_MODE,
        "extract_mode": extract_mode,
        "normalize_version": CONVO_NORMALIZE_VERSION,
        "source_size": _safe_int(file_state.get("source_size"), 0),
        "source_mtime": file_state.get("source_mtime"),
        "source_created_at": file_state.get("source_created_at"),
        "source_modified_at": file_state.get("source_modified_at"),
        "message_count": len(messages),
        "last_chunk_index": last_chunk_index,
        "transcript_digest": transcript_digest,
        "source_format": normalized_data.get("source_format", ""),
    }
    session_id = normalized_data.get("session_id")
    if session_id:
        metadata["source_session_id"] = session_id

    if messages:
        metadata["last_message_digest"] = _message_digest(messages[-1])
        start_time = messages[0].get("event_time_start") or messages[0].get("event_time_end")
        end_time = messages[-1].get("event_time_end") or messages[-1].get("event_time_start") or start_time
        metadata["event_time_start"] = start_time
        metadata["event_time_end"] = end_time
        metadata["event_at"] = end_time or file_state.get("source_modified_at") or filed_at
        metadata["timestamp_source"] = "message_timestamp" if end_time or start_time else "file_mtime"
    else:
        fallback = file_state.get("source_modified_at") or filed_at
        metadata["event_time_start"] = fallback
        metadata["event_time_end"] = fallback
        metadata["event_at"] = fallback
        metadata["timestamp_source"] = "file_mtime" if file_state.get("source_modified_at") else "ingest_time"
    return metadata


def _upsert_registry(collection, metadata: dict):
    source_file = metadata["source_file"]
    collection.upsert(
        documents=[f"[registry] {source_file}"],
        ids=[_registry_id(source_file)],
        metadatas=[metadata],
    )


def _upsert_chunks(collection, chunks: list, wing: str, agent: str, source_file: str, file_state: dict, normalized_data: dict):
    for chunk in chunks:
        room = chunk["room"]
        metadata = {
            "wing": wing,
            "room": room,
            "hall": _detect_hall_cached(chunk["content"]),
            "source_file": source_file,
            "chunk_index": chunk["chunk_index"],
            "chunk_key": chunk["chunk_key"],
            "added_by": agent,
            "filed_at": datetime.now().isoformat(),
            "ingest_mode": "convos",
            "record_kind": chunk.get("record_kind", "transcript"),
            "extract_mode": chunk.get("extract_mode", "exchange"),
            "normalize_version": CONVO_NORMALIZE_VERSION,
            "source_message_start_idx": chunk.get("source_message_start_idx", -1),
            "source_message_end_idx": chunk.get("source_message_end_idx", -1),
            "event_time_start": chunk.get("event_time_start"),
            "event_time_end": chunk.get("event_time_end"),
            "event_at": chunk.get("event_at") or file_state.get("source_modified_at"),
            "timestamp_source": chunk.get("timestamp_source", "file_mtime"),
            "source_mtime": file_state.get("source_mtime"),
            "source_created_at": file_state.get("source_created_at"),
            "source_modified_at": file_state.get("source_modified_at"),
            "source_format": normalized_data.get("source_format", ""),
        }
        session_id = normalized_data.get("session_id")
        if session_id:
            metadata["source_session_id"] = session_id
        task_hint = derive_task_hint(chunk["content"])
        if task_hint:
            metadata["task_hint"] = task_hint
        collection.upsert(
            documents=[chunk["content"]],
            ids=[_drawer_id(source_file, room, metadata["record_kind"], chunk["chunk_key"])],
            metadatas=[metadata],
        )


def _prepare_transcript_chunks(normalized_data: dict, file_state: dict, room: str, extract_mode: str, chunk_index_base: int = 0, messages_override=None):
    chunks = _build_transcript_chunks(
        normalized_data,
        file_state,
        chunk_index_base=chunk_index_base,
        messages_override=messages_override,
    )
    for chunk in chunks:
        chunk["room"] = room
        chunk["extract_mode"] = extract_mode
    return chunks


def _prepare_memory_chunks(normalized_data: dict, file_state: dict, chunk_index_base: int):
    chunks = _extract_general_chunks(normalized_data, chunk_index_base=chunk_index_base)
    if not chunks:
        return []
    stamped = _stamp_fallback_chunks(
        chunks,
        file_state,
        chunk_index_base=chunk_index_base,
        record_kind="memory",
    )
    for chunk in stamped:
        chunk["room"] = chunk.get("memory_type", "general")
        chunk["extract_mode"] = "general"
    return stamped


def mine_convos(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
):
    convo_path = Path(convo_dir).expanduser().resolve()
    if not wing:
        wing = convo_path.name.lower().replace(" ", "_").replace("-", "_")

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine - Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN - nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = get_collection(palace_path) if not dry_run else None
    total_drawers = 0
    files_skipped = 0
    room_counts = defaultdict(int)

    for index, filepath in enumerate(files, 1):
        source_file = str(filepath)
        file_state = _get_file_state(filepath)

        try:
            normalized_data = normalize_with_metadata(source_file)
        except (OSError, ValueError):
            normalized_data = {
                "transcript": "",
                "messages": [],
                "source_format": "error",
                "session_id": None,
            }

        transcript = normalized_data.get("transcript", "")
        transcript_digest = _transcript_digest(transcript)
        transcript_room = detect_convo_room(transcript) if transcript else "general"
        transcript_chunks = []
        memory_chunks = []

        if transcript and len(transcript.strip()) >= MIN_CHUNK_SIZE:
            transcript_chunks = _prepare_transcript_chunks(
                normalized_data,
                file_state,
                room=transcript_room,
                extract_mode=extract_mode,
            )
            if extract_mode == "general":
                memory_chunks = _prepare_memory_chunks(normalized_data, file_state, len(transcript_chunks))

        planned_chunks = transcript_chunks + memory_chunks

        if dry_run:
            if planned_chunks:
                print(f"    [DRY RUN] {filepath.name} -> {len(planned_chunks)} drawers")
                for chunk in planned_chunks:
                    room_counts[chunk["room"]] += 1
                total_drawers += len(planned_chunks)
            else:
                print(f"    [DRY RUN] {filepath.name} -> registry only")
            continue

        with mine_lock(source_file):
            registry, existing_drawers = _load_source_state(collection, source_file)
            registry_meta = registry["metadata"] if registry else None
            surviving_drawers = []

            if _registry_matches_file(registry_meta, file_state, transcript_digest):
                files_skipped += 1
                continue

            if extract_mode == "general":
                plan = {"mode": "full", "rewind_from": 0}
            else:
                plan = _plan_exchange_update(
                    registry_meta,
                    existing_drawers,
                    normalized_data.get("messages", []) or [],
                    transcript_digest,
                    file_state,
                )

            if plan["mode"] == "noop":
                files_skipped += 1
                continue

            if plan["mode"] == "full":
                _delete_source(collection, source_file)
                chunks_to_write = planned_chunks
            else:
                overlap_start = plan["rewind_from"]
                _delete_tail_drawers(collection, existing_drawers, overlap_start)
                surviving_drawers = [
                    drawer
                    for drawer in existing_drawers
                    if _safe_int((drawer.get("metadata") or {}).get("source_message_end_idx"), -1) < overlap_start
                ]
                chunk_index_base = _last_chunk_index(surviving_drawers) + 1
                partial_messages = [
                    message
                    for message in normalized_data.get("messages", []) or []
                    if _safe_int(message.get("seq"), -1) >= overlap_start
                ]
                chunks_to_write = _prepare_transcript_chunks(
                    normalized_data,
                    file_state,
                    room=transcript_room,
                    extract_mode=extract_mode,
                    chunk_index_base=chunk_index_base,
                    messages_override=partial_messages,
                )

            if chunks_to_write:
                _upsert_chunks(collection, chunks_to_write, wing, agent, source_file, file_state, normalized_data)
                for chunk in chunks_to_write:
                    room_counts[chunk["room"]] += 1
            else:
                _delete_source(collection, source_file)

            registry_metadata = _build_registry_metadata(
                source_file=source_file,
                wing=wing,
                agent=agent,
                file_state=file_state,
                normalized_data=normalized_data,
                transcript_digest=transcript_digest,
                last_chunk_index=max(
                    _last_chunk_index(surviving_drawers),
                    _last_chunk_index([{"metadata": {"chunk_index": chunk["chunk_index"]}} for chunk in chunks_to_write]),
                ),
                extract_mode=extract_mode,
            )
            _upsert_registry(collection, registry_metadata)

            total_drawers += len(chunks_to_write)
            print(f"  + [{index:4}/{len(files)}] {filepath.name[:50]:50} {len(chunks_to_write)}")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    if room_counts:
        print("\n  By room:")
        for room, count in sorted(room_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"    {room:20} {count} drawers")
    print('\n  Next: mempalace search "what you are looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
