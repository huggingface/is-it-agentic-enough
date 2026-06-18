"""Behavior markers: profile-defined regexes that flag whether a run exhibited
some behavior, so adoption can be tracked across bindings.

This replaces the old hard-wired CLI-vs-Python bucketing. A profile declares a
list of :class:`Marker`s (see ``Profile.markers``); each is an independent,
possibly-overlapping flag — a run can fire several or none. Reports show, per
cell and per binding, ``fired/total`` for each marker, i.e. "did adoption of
behavior X move across commits / model growth."

A marker searches one **scope** of the run, assembled from the parsed run:

- ``commands`` — the shell commands the agent ran (Bash inputs)
- ``wrote``    — contents (and paths) the agent ``Write``-wrote
- ``reads``    — paths the agent ``Read`` / ``Grep`` / ``Glob``'d
- ``final``    — the agent's final answer
- ``any``      — all of the above plus tool-result text

Generic profiles that declare no markers simply get no marker columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCOPES = ("commands", "wrote", "reads", "final", "any")


@dataclass(frozen=True)
class Marker:
    name: str
    pattern: str
    description: str            # one sentence explaining what this marker detects
    scope: str = "commands"


def run_corpus(run) -> dict[str, str]:
    """Assemble the per-scope search corpus from an ``analyze.Run`` (duck-typed:
    uses ``.tool_calls``, ``.tool_results``, ``.final``)."""
    cmds: list[str] = []
    wrote: list[str] = []
    reads: list[str] = []
    for name, inp in run.tool_calls:
        if name == "Bash":
            cmds.append(inp.get("command") or "")
        elif name == "Write":
            wrote.append(inp.get("file_path") or "")
            wrote.append(str(inp.get("content") or ""))
        elif name in ("Read", "Grep", "Glob"):
            reads.append(inp.get("file_path") or inp.get("pattern") or inp.get("path") or "")
    commands = "\n".join(cmds)
    wrote_s = "\n".join(wrote)
    reads_s = "\n".join(reads)
    final = run.final or ""
    any_s = "\n".join([commands, wrote_s, reads_s, "\n".join(run.tool_results), final])
    return {"commands": commands, "wrote": wrote_s, "reads": reads_s, "final": final, "any": any_s}


def fired(markers: list[Marker], run) -> dict[str, bool]:
    """Return ``{marker_name: did_it_fire}`` for one run."""
    corpus = run_corpus(run)
    out: dict[str, bool] = {}
    for m in markers:
        hay = corpus.get(m.scope, corpus["any"])
        out[m.name] = re.search(m.pattern, hay) is not None
    return out
