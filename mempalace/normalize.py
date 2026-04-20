#!/usr/bin/env python3
"""
normalize.py - Convert chat export formats to MemPalace transcript format.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SLACK_PROVENANCE_FOOTER = (
    "\n[source: slack-export | multi-party chat - speaker roles are positional, not verified]"
)

_NOISE_TAGS = (
    "system-reminder",
    "command-message",
    "command-name",
    "task-notification",
    "user-prompt-submit-hook",
    "hook_output",
)


def _tag_pattern(name: str) -> "re.Pattern[str]":
    return re.compile(
        rf"(?m)^(?:> )?<{name}(?:\s[^>]*)?>"
        rf"(?:(?!\n\s*\n)[\s\S])*?"
        rf"</{name}>[ \t]*\n?"
    )


_NOISE_TAG_PATTERNS = [_tag_pattern(tag) for tag in _NOISE_TAGS]
_NOISE_LINE_PREFIXES = (
    "CURRENT TIME:",
    "VERIFIED FACTS (do not contradict)",
    "AGENT SPECIALIZATION:",
    "Checking verified facts...",
    "Injecting timestamp...",
    "Starting background pipeline...",
    "Checking emotional weights...",
    "Auto-save reminder...",
    "Checking pipeline...",
    "MemPalace auto-save checkpoint.",
)
_NOISE_LINE_PATTERNS = [
    re.compile(rf"(?m)^(?:> )?{re.escape(prefix)}.*\n?") for prefix in _NOISE_LINE_PREFIXES
]
_HOOK_LINE_RE = re.compile(
    r"(?m)^(?:> )?Ran \d+ "
    r"(?:Stop|PreCompact|PreToolUse|PostToolUse|UserPromptSubmit|Notification|SessionStart|SessionEnd) "
    r"hook[s]?.*\n?"
)
_COLLAPSED_LINES_RE = re.compile(r"(?m)^(?:> )?…\s*\+\d+ lines.*\n?")

_TOOL_RESULT_MAX_LINES_BASH = 20
_TOOL_RESULT_MAX_MATCHES = 20
_TOOL_RESULT_MAX_BYTES = 2048


def strip_noise(text: str) -> str:
    for pat in _NOISE_TAG_PATTERNS:
        text = pat.sub("", text)
    for pat in _NOISE_LINE_PATTERNS:
        text = pat.sub("", text)
    text = _HOOK_LINE_RE.sub("", text)
    text = _COLLAPSED_LINES_RE.sub("", text)
    text = re.sub(r"\s*\[\d+\s+tokens?\]\s*\(ctrl\+o to expand\)", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize(filepath: str) -> str:
    return normalize_with_metadata(filepath)["transcript"]


def normalize_with_metadata(filepath: str) -> dict:
    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}")
    if file_size > 500 * 1024 * 1024:
        raise IOError(f"File too large ({file_size // (1024 * 1024)} MB): {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}")

    if not content.strip():
        return {
            "transcript": content,
            "messages": [],
            "source_format": "empty",
            "session_id": None,
        }

    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return {
            "transcript": content,
            "messages": _transcript_to_messages(content),
            "source_format": "transcript",
            "session_id": None,
        }

    ext = Path(filepath).suffix.lower()
    if ext in (".json", ".jsonl") or content.strip()[:1] in ("{", "["):
        normalized = _try_normalize_json_structured(content)
        if normalized:
            return normalized

    return {
        "transcript": content,
        "messages": [],
        "source_format": "plain_text",
        "session_id": None,
    }


def _try_normalize_json(content: str) -> Optional[str]:
    normalized = _try_normalize_json_structured(content)
    if normalized:
        return normalized["transcript"]
    return None


def _try_normalize_json_structured(content: str) -> Optional[dict]:
    normalized = _try_claude_code_jsonl_structured(content)
    if normalized:
        return normalized

    normalized = _try_codex_jsonl_structured(content)
    if normalized:
        return normalized

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    for parser in (_try_claude_ai_json, _try_chatgpt_json, _try_slack_json):
        normalized = parser(data)
        if normalized:
            return {
                "transcript": normalized,
                "messages": _transcript_to_messages(normalized),
                "source_format": "json_transcript",
                "session_id": None,
            }

    return None


def _try_claude_code_jsonl(content: str) -> Optional[str]:
    normalized = _try_claude_code_jsonl_structured(content)
    if normalized:
        return normalized["transcript"]
    return None


def _try_claude_code_jsonl_structured(content: str) -> Optional[dict]:
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages = []
    session_id = None

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue

        msg_content = message.get("content", "")
        entry_timestamp = _coerce_iso_timestamp(
            entry.get("timestamp") or message.get("created_at") or message.get("updated_at")
        )
        session_id = session_id or entry.get("sessionId") or entry.get("session_id")

        if msg_type in ("human", "user"):
            text = _extract_content(msg_content, include_tool_blocks=False)
            if text:
                text = strip_noise(text)
            if text:
                _append_or_merge_message(messages, role="user", text=text, timestamp=entry_timestamp)
        elif msg_type == "assistant":
            text = _extract_content(msg_content, include_tool_blocks=False)
            if text:
                text = strip_noise(text)
            if text:
                _append_or_merge_message(
                    messages,
                    role="assistant",
                    text=text,
                    timestamp=entry_timestamp,
                    merge_same_role=True,
                )

    if len(messages) >= 2:
        return _build_structured_result(messages, "claude_code_jsonl", session_id=session_id)
    return None


def _try_codex_jsonl(content: str) -> Optional[str]:
    normalized = _try_codex_jsonl_structured(content)
    if normalized:
        return normalized["transcript"]
    return None


def _try_codex_jsonl_structured(content: str) -> Optional[dict]:
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages = []
    has_session_meta = False
    session_id = None

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry_type == "session_meta":
            has_session_meta = True
            payload = entry.get("payload", {})
            if isinstance(payload, dict):
                session_id = session_id or payload.get("id") or payload.get("session_id")
            continue

        if entry_type != "event_msg":
            continue

        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type", "")
        msg = payload.get("message")
        if not isinstance(msg, str):
            continue
        text = msg.strip()
        if not text:
            continue

        entry_timestamp = _coerce_iso_timestamp(entry.get("timestamp")) or _coerce_iso_timestamp(
            payload.get("timestamp")
        )
        if payload_type == "user_message":
            _append_or_merge_message(messages, role="user", text=text, timestamp=entry_timestamp)
        elif payload_type == "agent_message":
            _append_or_merge_message(messages, role="assistant", text=text, timestamp=entry_timestamp)

    if len(messages) >= 2 and has_session_meta:
        return _build_structured_result(messages, "codex_jsonl", session_id=session_id)
    return None


def _coerce_iso_timestamp(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return datetime.fromtimestamp(float(text), timezone.utc).isoformat()
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _append_or_merge_message(
    messages: list,
    role: str,
    text: str,
    timestamp: Optional[str],
    merge_same_role: bool = False,
):
    if not text:
        return
    if merge_same_role and messages and messages[-1]["role"] == role:
        messages[-1]["text"] = messages[-1]["text"] + "\n" + text
        if timestamp:
            if not messages[-1].get("event_time_start"):
                messages[-1]["event_time_start"] = timestamp
            messages[-1]["event_time_end"] = timestamp
        return

    messages.append(
        {
            "role": role,
            "text": text,
            "event_time_start": timestamp,
            "event_time_end": timestamp,
            "seq": len(messages),
        }
    )


def _build_structured_result(messages: list, source_format: str, session_id: Optional[str] = None) -> dict:
    transcript = _messages_to_transcript([(m["role"], m["text"]) for m in messages])
    return {
        "transcript": transcript,
        "messages": messages,
        "source_format": source_format,
        "session_id": session_id,
    }


def _transcript_to_messages(content: str) -> list:
    messages = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_lines = [line.strip()[1:].lstrip()]
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                user_lines.append(lines[i].strip()[1:].lstrip())
                i += 1
            user_text = "\n".join(part for part in user_lines if part).strip()
            if user_text:
                _append_or_merge_message(messages, role="user", text=user_text, timestamp=None)

            assistant_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">"):
                    break
                if next_line.strip():
                    assistant_lines.append(next_line)
                i += 1
            assistant_text = "\n".join(assistant_lines).strip()
            if assistant_text:
                _append_or_merge_message(
                    messages,
                    role="assistant",
                    text=assistant_text,
                    timestamp=None,
                    merge_same_role=True,
                )
            continue
        i += 1
    return messages


def _try_claude_ai_json(data) -> Optional[str]:
    if isinstance(data, dict):
        data = data.get("messages", data.get("chat_messages", []))
    if not isinstance(data, list):
        return None

    if data and isinstance(data[0], dict) and ("chat_messages" in data[0] or "messages" in data[0]):
        transcripts = []
        for convo in data:
            if not isinstance(convo, dict):
                continue
            chat_msgs = convo.get("chat_messages") or convo.get("messages", [])
            messages = _collect_claude_messages(chat_msgs)
            if len(messages) >= 2:
                transcripts.append(_messages_to_transcript(messages))
        if transcripts:
            return "\n\n".join(transcripts)
        return None

    messages = _collect_claude_messages(data)
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _collect_claude_messages(items) -> list:
    messages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("sender", "")
        text = _extract_content(item.get("content", "")) or (item.get("text") or "").strip()
        if role in ("user", "human") and text:
            messages.append(("user", text))
        elif role in ("assistant", "ai") and text:
            messages.append(("assistant", text))
    return messages


def _try_chatgpt_json(data) -> Optional[str]:
    if not isinstance(data, dict) or "mapping" not in data:
        return None
    mapping = data["mapping"]
    messages = []
    root_id = None
    fallback_root = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            if node.get("message") is None:
                root_id = node_id
                break
            if fallback_root is None:
                fallback_root = node_id
    if not root_id:
        root_id = fallback_root
    if root_id:
        current_id = root_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message")
            if msg:
                role = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = " ".join(str(part) for part in parts if isinstance(part, str) and part).strip()
                if role == "user" and text:
                    messages.append(("user", text))
                elif role == "assistant" and text:
                    messages.append(("assistant", text))
            children = node.get("children", [])
            current_id = children[0] if children else None
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_slack_json(data) -> Optional[str]:
    if not isinstance(data, list):
        return None
    messages = []
    seen_users = {}
    last_role = None
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        raw_user_id = item.get("user", item.get("username", ""))
        user_id = re.sub(r"[\[\]\n\r\x00-\x1f]", "_", raw_user_id).strip()
        text = item.get("text", "").strip()
        if not text or not user_id:
            continue
        if user_id not in seen_users:
            if not seen_users:
                seen_users[user_id] = "user"
            elif last_role == "user":
                seen_users[user_id] = "assistant"
            else:
                seen_users[user_id] = "user"
        last_role = seen_users[user_id]
        messages.append((seen_users[user_id], f"[{user_id}] {text}"))
    if len(messages) >= 2:
        return _messages_to_transcript(messages) + _SLACK_PROVENANCE_FOOTER
    return None


def _extract_content(content, tool_use_map: dict = None, include_tool_blocks: bool = True) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    parts.append(item.get("text", ""))
                elif include_tool_blocks and block_type == "tool_use":
                    parts.append(_format_tool_use(item))
                elif include_tool_blocks and block_type == "tool_result":
                    tid = item.get("tool_use_id", "")
                    tname = (tool_use_map or {}).get(tid, "Unknown")
                    result_content = item.get("content", "")
                    formatted = _format_tool_result(result_content, tname)
                    if formatted:
                        parts.append(formatted)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        return content.get("text", "").strip()
    return ""


def _format_tool_use(block: dict) -> str:
    name = block.get("name", "Unknown")
    inp = block.get("input", {})

    if name == "Bash":
        cmd = inp.get("command", "")
        if len(cmd) > 200:
            cmd = cmd[:200] + "..."
        return f"[Bash] {cmd}"

    if name == "Read":
        path = inp.get("file_path", "?")
        offset = inp.get("offset")
        limit = inp.get("limit")
        if offset is not None and limit is not None:
            try:
                return f"[Read {path}:{offset}-{int(offset) + int(limit)}]"
            except (TypeError, ValueError):
                return f"[Read {path}:{offset}+{limit}]"
        return f"[Read {path}]"

    if name == "Grep":
        pattern = inp.get("pattern", "")
        target = inp.get("path") or inp.get("glob") or ""
        return f"[Grep] {pattern} in {target}"

    if name == "Glob":
        pattern = inp.get("pattern", "")
        return f"[Glob] {pattern}"

    if name in ("Edit", "Write"):
        path = inp.get("file_path", "?")
        return f"[{name} {path}]"

    summary = json.dumps(inp, separators=(",", ":"))
    if len(summary) > 200:
        summary = summary[:200] + "..."
    return f"[{name}] {summary}"


def _format_tool_result(content, tool_name: str) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts)
    else:
        text = str(content) if content else ""

    text = text.strip()
    if not text:
        return ""

    if tool_name in ("Read", "Edit", "Write"):
        return ""

    lines = text.split("\n")
    if tool_name == "Bash":
        if len(lines) <= _TOOL_RESULT_MAX_LINES_BASH * 2:
            return "→ " + "\n→ ".join(lines)
        head = lines[:_TOOL_RESULT_MAX_LINES_BASH]
        tail = lines[-_TOOL_RESULT_MAX_LINES_BASH :]
        omitted = len(lines) - (_TOOL_RESULT_MAX_LINES_BASH * 2)
        return (
            "→ "
            + "\n→ ".join(head)
            + f"\n→ ... [{omitted} lines omitted] ..."
            + "\n→ "
            + "\n→ ".join(tail)
        )

    if tool_name in ("Grep", "Glob"):
        if len(lines) <= _TOOL_RESULT_MAX_MATCHES:
            return "→ " + "\n→ ".join(lines)
        kept = lines[:_TOOL_RESULT_MAX_MATCHES]
        remaining = len(lines) - _TOOL_RESULT_MAX_MATCHES
        return "→ " + "\n→ ".join(kept) + f"\n→ ... [{remaining} more matches]"

    if len(text) > _TOOL_RESULT_MAX_BYTES:
        return "→ " + text[:_TOOL_RESULT_MAX_BYTES] + f"... [truncated, {len(text)} chars]"
    return "→ " + text


def _messages_to_transcript(messages: list, spellcheck: bool = True) -> str:
    if spellcheck:
        try:
            from mempalace.spellcheck import spellcheck_user_text

            _fix = spellcheck_user_text
        except ImportError:
            _fix = None
    else:
        _fix = None

    lines = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            if _fix is not None:
                text = _fix(text)
            lines.append(f"> {text}")
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                lines.append(messages[i + 1][1])
                i += 2
            else:
                i += 1
        else:
            lines.append(text)
            i += 1
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python normalize.py <filepath>")
        sys.exit(1)
    path = sys.argv[1]
    result = normalize(path)
    quote_count = sum(1 for line in result.split("\n") if line.strip().startswith(">"))
    print(f"\nFile: {os.path.basename(path)}")
    print(f"Normalized: {len(result)} chars | {quote_count} user turns detected")
    print("\n--- Preview (first 20 lines) ---")
    print("\n".join(result.split("\n")[:20]))
