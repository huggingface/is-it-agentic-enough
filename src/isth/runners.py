"""Agent-runner abstraction.

The harness drives a headless coding agent and parses its event stream for
token / tool-call accounting and, later, for the analysis reports. Different
agents speak different wire formats, so each runner is responsible for three
things:

- :meth:`Runner.build_cmd` — the subprocess command for one cell.
- :meth:`Runner.env` — environment tweaks (PATH, token hygiene).
- :meth:`Runner.normalize` — translate one raw stdout JSON line into zero or
  more events in the *canonical* schema (Claude Code's ``stream-json``).

By normalizing every runner's output to the canonical schema **at write time**,
the rest of the harness — the live accounting loop in ``run_task.run`` and the
whole read-side (:mod:`isth.analyze`, :mod:`isth.compare`, :mod:`isth.explain`,
:func:`isth.log.summarize_event`) — stays runner-agnostic, and Pi runs land in
the same tables as Claude runs.

The canonical event shapes consumed downstream are:

    {"type": "assistant", "message": {"content": [<block>, ...],
                                       "usage": {"input_tokens", "output_tokens",
                                                 "cache_read_input_tokens",
                                                 "cache_creation_input_tokens"}}}
    {"type": "user", "message": {"content": [{"type": "tool_result",
                                              "tool_use_id", "content", "is_error"}]}}

where ``<block>`` is ``{"type": "text", "text"}`` or
``{"type": "tool_use", "id", "name", "input"}``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _newest_jsonl(d: Path | None) -> Path | None:
    if not d or not d.exists():
        return None
    files = [p for p in d.rglob("*.jsonl") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


class Runner:
    """Base class. ``name`` is the public runner id used on the CLI."""

    name: str

    def build_cmd(
        self,
        prompt: str,
        ws: Path,
        cfg_dir: Path,
        variant: str,
        model: str | None,
        provider: str | None,
        session_dir: Path | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def env(self, venv_python: Path, session_dir: Path | None = None) -> dict[str, str]:
        """Base env: prepend the per-commit venv to PATH and strip HF tokens so
        the agent's task work (model downloads) stays anonymous and comparable
        across runs."""
        env = dict(os.environ)
        env["PATH"] = f"{venv_python.parent}:{env.get('PATH', '')}"
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(k, None)
        return env

    def normalize(self, event: dict) -> list[dict]:
        """Translate one raw stdout JSON object into canonical events."""
        raise NotImplementedError

    def collect_session(self, session_dir: Path, ws: Path) -> Path | None:
        """Return the agent's *native* session file written during the run (for
        upload to the Hub agent-traces viewer), or None if not found.
        ``session_dir`` is the dir handed to :meth:`build_cmd`/:meth:`env`."""
        return _newest_jsonl(session_dir)


class ClaudeRunner(Runner):
    name = "claude"

    def build_cmd(self, prompt, ws, cfg_dir, variant, model, provider, session_dir=None):
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--add-dir",
            str(ws),
        ]
        # When not capturing sessions, stay ephemeral (historical behavior).
        # When capturing, let Claude Code persist to its default project store
        # (~/.claude/projects/<escaped-cwd>/…); collect_session finds it by cwd.
        # We deliberately do NOT relocate CLAUDE_CONFIG_DIR — that would lose
        # the user's Claude auth.
        if session_dir is None:
            cmd.append("--no-session-persistence")
        if model:
            cmd.extend(["--model", model])
        if variant == "skill":
            cmd.extend(["--plugin-dir", str(cfg_dir / "plugin")])
        return cmd

    def normalize(self, event: dict) -> list[dict]:
        # Claude already emits the canonical schema.
        return [event]

    def collect_session(self, session_dir: Path, ws: Path) -> Path | None:
        # Claude Code stores sessions under ~/.claude/projects/<escaped-cwd>/.
        # The escaping replaces path separators (and other chars) with '-'.
        projects = Path.home() / ".claude" / "projects"
        escaped = str(ws).replace("/", "-")
        candidates = [projects / escaped, *projects.glob(f"*{ws.name}*")]
        newest = None
        for d in candidates:
            cur = _newest_jsonl(d)
            if cur and (newest is None or cur.stat().st_mtime > newest.stat().st_mtime):
                newest = cur
        return newest


# Pi tool name -> canonical (Claude) tool name. Pi tools are lowercase; the
# read-side and summarizer key off Claude's capitalized names.
_PI_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "list": "Glob",
    "glob": "Glob",
    "grep": "Grep",
}

# Pi tool-arg key -> canonical key, per tool. Only keys the downstream cares
# about (summaries, doc-detection) need mapping; unknown keys pass through.
_PI_ARG_KEYS = {
    "Bash": {"command": "command"},
    "Read": {"path": "file_path", "file_path": "file_path"},
    "Write": {"path": "file_path", "file_path": "file_path"},
    "Edit": {"path": "file_path", "file_path": "file_path"},
    "Grep": {"pattern": "pattern", "query": "pattern"},
    "Glob": {"pattern": "pattern"},
}


def _map_tool(name: str) -> str:
    return _PI_TOOL_NAMES.get(name, name)


def _result_to_content(result):
    """Coerce a Pi ``tool_execution_end.result`` into a canonical tool_result
    ``content`` (a string, or a ``[{type:text,text}]`` list — both of which the
    read-side already understands)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        # Already a list of content blocks (Pi tool results are
        # ``[{type:"text", text:...}]``) — keep as-is.
        if all(isinstance(b, dict) and "text" in b for b in result):
            return result
        return json.dumps(result)
    if isinstance(result, dict):
        # ToolResultMessage-like: prefer its inner content if present.
        inner = result.get("content")
        if inner is not None:
            return _result_to_content(inner)
        if "text" in result:
            return result["text"]
        return json.dumps(result)
    return str(result)


def _map_args(canonical_name: str, args: dict | None) -> dict:
    args = args or {}
    mapping = _PI_ARG_KEYS.get(canonical_name, {})
    if not mapping:
        return dict(args)
    out = dict(args)
    for src, dst in mapping.items():
        if src in args and dst not in out:
            out[dst] = args[src]
    return out


class PiRunner(Runner):
    """Drives the ``pi`` CLI (``@mariozechner/pi-coding-agent``) against any
    provider Pi knows — in particular ``huggingface`` for HF inference
    providers. Translates Pi's ``--mode json`` event stream to the canonical
    schema."""

    name = "pi"

    def build_cmd(self, prompt, ws, cfg_dir, variant, model, provider, session_dir=None):
        cmd = [
            "pi",
            "-p",
            prompt,
            "--mode",
            "json",
        ]
        # Persist the native session into session_dir when capturing (for Hub
        # upload); otherwise stay ephemeral.
        if session_dir is not None:
            cmd.extend(["--session-dir", str(session_dir)])
        else:
            cmd.append("--no-session")
        if provider:
            cmd.extend(["--provider", provider])
        if model:
            cmd.extend(["--model", model])
        # Give Pi the HF key for its OWN model calls only (the task env stays
        # token-free; see env()). Fail fast with a clear message if missing.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if (provider or "") == "huggingface":
            if not token:
                raise SystemExit(
                    "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to run the "
                    "pi runner against the huggingface provider."
                )
            cmd.extend(["--api-key", token])
        if variant == "skill":
            cmd.extend(["--skill", str(cfg_dir / "plugin" / "skills" / "transformers")])
        return cmd

    # --- event normalization -------------------------------------------------
    # NOTE: the exact assistant-content / usage shape is confirmed against a
    # real `pi --mode json` capture in the smoke test; adjust the field reads
    # below if Pi's schema differs.

    def normalize(self, event: dict) -> list[dict]:
        etype = event.get("type")
        if etype == "message_end":
            msg = event.get("message") or {}
            if msg.get("role") != "assistant":
                return []
            return self._assistant_event(msg)
        if etype == "tool_execution_end":
            return self._tool_result_event(event)
        return []

    def _assistant_event(self, msg: dict) -> list[dict]:
        blocks: list[dict] = []
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt in ("text", "thinking"):
                    txt = b.get("text") or b.get("thinking") or ""
                    if txt:
                        blocks.append({"type": "text", "text": txt})
                elif bt in ("toolCall", "tool_call", "tool_use"):
                    name = _map_tool(b.get("name") or b.get("toolName") or "")
                    raw_args = b.get("arguments") or b.get("args") or b.get("input") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": b.get("id") or b.get("toolCallId") or "",
                            "name": name,
                            "input": _map_args(name, raw_args),
                        }
                    )
        # Some assistant messages also carry tool calls in a separate field.
        for tc in msg.get("toolCalls") or []:
            name = _map_tool(tc.get("name") or tc.get("toolName") or "")
            raw_args = tc.get("arguments") or tc.get("args") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or tc.get("toolCallId") or "",
                    "name": name,
                    "input": _map_args(name, raw_args),
                }
            )
        usage = msg.get("usage") or {}
        return [
            {
                "type": "assistant",
                "message": {
                    "content": blocks,
                    "usage": {
                        "input_tokens": int(usage.get("input") or 0),
                        "output_tokens": int(usage.get("output") or 0),
                        "cache_read_input_tokens": int(usage.get("cacheRead") or 0),
                        "cache_creation_input_tokens": int(usage.get("cacheWrite") or 0),
                    },
                },
            }
        ]

    def _tool_result_event(self, event: dict) -> list[dict]:
        content = _result_to_content(event.get("result"))
        return [
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": event.get("toolCallId") or "",
                            "content": content,
                            "is_error": bool(event.get("isError")),
                        }
                    ]
                },
            }
        ]


_RUNNERS = {r.name: r for r in (ClaudeRunner(), PiRunner())}


def get_runner(name: str | None) -> Runner:
    runner = _RUNNERS.get(name or "claude")
    if runner is None:
        raise SystemExit(f"Unknown runner: {name!r} (choose from {sorted(_RUNNERS)})")
    return runner
