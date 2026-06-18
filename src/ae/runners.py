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
read-side (:mod:`ae.runs`, :func:`ae.log.summarize_event`, the report builder) —
stays runner-agnostic.

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
import random
import sys
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
        assets: dict,
        model: str | None,
        session_dir: Path | None = None,
    ) -> list[str]:
        """Build the subprocess command. ``assets`` is the profile's per-tier
        extras: ``skill_dir`` (Pi ``--skill``); empty for tiers/profiles with no
        agent assets."""
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
    """Drives the ``pi`` CLI (``@mariozechner/pi-coding-agent``) against
    Hugging Face inference providers (the only provider we use it with).
    Translates Pi's ``--mode json`` event stream to the canonical schema."""

    name = "pi"
    PROVIDER = "huggingface"

    def build_cmd(self, prompt, ws, assets, model, session_dir=None):
        cmd = [
            "pi",
            "-p",
            prompt,
            "--mode",
            "json",
            "--provider",
            self.PROVIDER,
        ]
        # Always persist the native session into session_dir (for Hub upload).
        if session_dir is not None:
            cmd.extend(["--session-dir", str(session_dir)])
        if model:
            cmd.extend(["--model", model])
        # Give Pi the HF key for its OWN model calls only (the task env stays
        # token-free; see env()). Fail fast with a clear message if missing.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise SystemExit(
                "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to run the pi runner."
            )
        cmd.extend(["--api-key", token])
        if assets.get("skill_dir"):
            cmd.extend(["--skill", str(assets["skill_dir"])])
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


class MockRunner(Runner):
    """A fake agent for fast UI / end-to-end testing (pair with the ``mock`` profile).

    Instead of driving a real CLI, it synthesizes a randomized — but
    deterministic per cell — canonical stream-json transcript and replays it
    through a one-shot subprocess, so the real ``run_task`` parse → write →
    trace → cleanup path is exercised. A whole suite finishes in seconds.

    Behavior is biased by tier so the report tells a story: the ``skill`` tier
    "adopts" the CLI more than ``bare``. Approaches/markers/errors/match/tokens
    all vary across cells.
    """

    name = "mock"

    # Per-tier weights over [cli, pipeline, no-tool, error] — skill favors the CLI.
    _TIER_WEIGHTS = {
        "bare": [0.15, 0.6, 0.1, 0.15],
        "clone": [0.45, 0.4, 0.05, 0.1],
        "skill": [0.85, 0.1, 0.0, 0.05],
    }

    def _events(self, ws: Path) -> tuple[list[dict], float]:
        # Cell coordinates live in the workspace name: {binding}__{tier}__{task}__run{N}
        parts = ws.name.split("__")
        tier = parts[1] if len(parts) >= 4 else "bare"
        task = parts[2] if len(parts) >= 4 else "task"
        rng = random.Random(ws.name)  # deterministic per cell, varied across cells

        expected = None
        try:
            # The mock runner pairs with the mock profile, which reuses the
            # transformers task suite — read `expected` from there.
            from .profiles.transformers import tasks as _tasks

            expected = (_tasks().get(task) or {}).get("expected")
        except Exception:
            pass

        approach = rng.choices(
            ["cli", "pipeline", "none", "error"],
            weights=self._TIER_WEIGHTS.get(tier, self._TIER_WEIGHTS["bare"]),
        )[0]
        sub = rng.choice(["classify", "ner", "tokenize", "caption", "transcribe", "summarize"])

        events: list[dict] = []
        tid = 0

        def _assistant(name: str, inp: dict) -> None:
            nonlocal tid
            tid += 1
            events.append({
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": f"t{tid}", "name": name, "input": inp}],
                    "usage": {
                        "input_tokens": rng.randint(400, 4000),
                        "output_tokens": rng.randint(20, 600),
                        "cache_read_input_tokens": rng.randint(0, 3000),
                        "cache_creation_input_tokens": rng.randint(0, 500),
                    },
                },
            })

        def _result(content: str, is_error: bool = False) -> None:
            events.append({
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result", "tool_use_id": f"t{tid}",
                    "content": content, "is_error": is_error,
                }]},
            })

        # Occasionally consult docs first (fires ran-help / agentic-exemplar markers).
        if tier in ("clone", "skill") and rng.random() < 0.3:
            _assistant("Read", {"file_path": "/repo/src/transformers/cli/agentic/text.py"})
            _result("def classify(...): ...")
        if rng.random() < 0.2:
            _assistant("Bash", {"command": "transformers --help"})
            _result("Usage: transformers [OPTIONS] COMMAND ...")

        if approach == "cli":
            _assistant("Bash", {"command": f'transformers --format json {sub} --text "x" --model some/model'})
            _result(json.dumps({"label": expected or "RESULT", "score": 0.99}))
        elif approach == "pipeline":
            _assistant("Bash", {"command": "python3 -c 'from transformers import pipeline; p=pipeline(); print(p(\"x\"))'"})
            _result(f"[{{'label': '{expected or 'RESULT'}', 'score': 0.98}}]")
        elif approach == "error":
            _assistant("Bash", {"command": "python3 -c 'from transformers import pipeline; pipeline(\"bad-task\")'"})
            _result("Traceback (most recent call last):\nKeyError: 'bad-task'", is_error=True)
            if rng.random() < 0.7:  # recover via CLI
                _assistant("Bash", {"command": f'transformers {sub} --text "x"'})
                _result(json.dumps({"label": expected or "RESULT"}))

        hit = bool(expected) and rng.random() < 0.75 and approach != "none"
        final = f"The result is {expected}." if hit else "Done — see the output above."
        events.append({"type": "result", "result": final, "is_error": approach == "error" and rng.random() < 0.2})

        # Log-spread elapsed (~0.01–1s) so the time distribution/median look alive
        # without slowing the suite much.
        return events, round(10 ** rng.uniform(-2, 0.0), 3)

    def build_cmd(self, prompt, ws, assets, model, session_dir=None):
        events, sleep = self._events(ws)
        script = (
            "import sys, json, time\n"
            "time.sleep(float(sys.argv[2]))\n"
            "for e in json.loads(sys.argv[1]):\n"
            "    print(json.dumps(e), flush=True)\n"
        )
        return [sys.executable, "-c", script, json.dumps(events), f"{sleep:.3f}"]

    def normalize(self, event: dict) -> list[dict]:
        return [event]  # already canonical

    def collect_session(self, session_dir: Path, ws: Path) -> Path | None:
        # Synthesize a tiny native-session file so traces/ (and the upload/report
        # UI) is populated too.
        p = session_dir / "mock-session.jsonl"
        p.write_text(
            json.dumps({"type": "session", "mock": True, "cwd": str(ws)}) + "\n"
            + json.dumps({"type": "message", "role": "assistant", "content": "mock run"}) + "\n"
        )
        return p


_RUNNERS = {r.name: r for r in (PiRunner(), MockRunner())}


def get_runner(name: str | None) -> Runner:
    runner = _RUNNERS.get(name or "pi")
    if runner is None:
        raise SystemExit(f"Unknown runner: {name!r} (choose from {sorted(_RUNNERS)})")
    return runner
