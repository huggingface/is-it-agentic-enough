<p align="center">
  <img width="100%" alt="Screenshot 2026-04-28 at 16 19 34" src="https://github.com/user-attachments/assets/f8d3962d-cd28-4891-90e1-dc54ad841989" />
</p>

# is-transformers-agentic-enough

Harness for measuring how a Claude Code agent uses `transformers` at a
given commit, across three discovery conditions.

> ⚠️ **Trusted local use only.** This harness runs `claude` with
> `--permission-mode bypassPermissions` and executes code from whatever
> `transformers` ref you point it at. See [SECURITY.md](./SECURITY.md)
> before pointing it at anything you didn't write yourself, and before
> sharing the contents of `results/`.

## Install

```bash
uv venv --python 3.13 .env
uv pip install --python .env/bin/python -e .
```

This installs the `isth` command into `.env/bin/isth`. The harness assumes
the transformers source repo lives at `../transformers` relative to this
directory; override with `ISTH_TRANSFORMERS_SRC=/path/to/transformers` if
it's elsewhere. Runtime state (`configs/`, `workspaces/`, `results/`)
lands next to cwd; override via `ISTH_DATA_DIR`.

## Conditions tested per commit

- **bare** — only `pip install` of transformers. Workspace has `inputs/`
  and nothing else. Tests whether the CLI is self-discoverable with no
  repo access and no skill hint.
- **clone** — pip install + the workspace IS a git worktree of transformers
  at that SHA. `AGENTS.md`, `CLAUDE.md`, `src/transformers/cli/agentic/`
  auto-discover from cwd.
- **skill** — pip install + a Claude Code plugin dir with a `SKILL.md`
  rendered from the commit's `skill.json` manifest, loaded via
  `--plugin-dir`. Silently skipped for commits where the skill can't be
  derived.

Each `(commit × variant × task)` is run N times (default 3) to smooth
model non-determinism.

## Commands

The full CLI reference — every subcommand, every flag, the typical
workflows — lives in [API.md](./API.md). At a glance:

```bash
isth tasks                                  # list the 8 task ids
isth setup <ref> [--remove]                 # build / tear down per-commit cache
isth run <ref> <task> <run_index> [variants...]  # one cell, ad-hoc
isth suite <ref>                            # full suite for one commit
isth diff <ref1>..<ref2>[..<refN>]          # end-to-end: run + compare
isth analyze <short-sha> [task_id]          # per-commit markdown report
isth compare <refs...>                      # cross-ref diff table
isth explain <variant> <task> <refs...>     # per-cell tool-call timeline
```

Most-common path: `isth diff A..B > progress.md` builds caches, runs the
suite on each ref, and prints the comparison report in one shot. While
it's running you can run `isth explain <variant> <task> A..B` from
another terminal to drill into any cell that looks weird in the live
dashboard.

## Worked example

The checked-in [`progress.md`](./progress.md) was produced by running the
harness against two commits straddling the agent-first CLI work.

```bash
isth suite 0ea540efff           # "before" — /v1/completions endpoint commit
isth suite 59e4754341           # "after"  — bugfixes after the CLI landed
isth compare 0ea540efff 59e4754341 > progress.md
```

Each `(commit × variant × task)` cell ran 3 times, so a full suite for one
commit is 3 variants × 8 tasks × 3 runs = 72 runs (skipping `skill` for the
earlier commit, which predates the manifest).

### What gets serialized

For every individual run the harness writes two files under `results/`:

- `<sha>__<variant>__<task>__run<N>.jsonl` — the raw Claude Code
  stream-json transcript: `system` events (session start, hook fires),
  every `assistant` turn with its tool-call arguments, every `user` turn
  carrying the `tool_result` payload (with `is_error` flags), and the
  final `result` event. This is the full trace of what the agent saw,
  what it ran, and what came back — nothing is summarised away.
- `<sha>__<variant>__<task>__run<N>.meta.json` — a small sidecar:
  resolved SHA, variant, task id, run index, model, status, tool-call
  count, wall-clock seconds, exit code, token accounting (`input`,
  `output`, `cache_read`, `cache_creation`), the exact `claude` command
  that was executed, and the workspace path.

For the two commits above, that's 6 variants-with-data × 8 tasks × 3 runs =
~144 `.jsonl` + 144 `.meta.json` files in `results/`.

### What `compare` shows on top of that

`progress.md` distils those raw traces into:

1. A header explaining the variants (`bare` / `clone` / `skill`), the
   tasks, and the cell legend.
2. **Headline summary tables** — atomic vs. compositional tasks, one
   column per commit: CLI-vs-Python adoption, error rate, match rate,
   median wall-clock, token totals.
3. **Per-variant summary tables** — same metrics, one sub-table per
   variant, so you can read "holding `bare` constant, what changed?"
   directly.
4. **Per-task tables** — one table per task, rows = variants, columns =
   commits. Each cell carries the approach bucket
   (`CLI-clean=3/3`, `Python-retry=1/3`, ...), `✓match`, error counts,
   first-success index, which docs the agent consulted, median time,
   and the `new`/`repeat`/`out` token split.

In the checked-in report the headline shift is visible at a glance:
CLI adoption on atomic tasks goes from 1/36 to 19/54, errored tool calls
drop from 14/131 to 2/203, and the new `skill` variant lands at 24/24
clean CLI runs.

### Feeding the result to an LLM

Because `progress.md` is self-describing (legend + tables in one
document) and the underlying `.jsonl` traces are kept verbatim, the
intended next step is to hand the report (optionally plus selected raw
traces for the cells you want to drill into) to an LLM and ask it to
narrate the behavioural diff between the two commits — what the agent
stopped doing, what it started doing, where it still struggles.

Example of summary given by an LLM:

```
Between commit `9914a3641f` (March 31) and `59e4754341` (April 23):

**Skill variant** — only exists at the newer commit, and it's a clean sweep: 30/30 CLI adoption, zero errors, fastest runs, tightest variance.

**Clone variant** — more reliable (errors 6/30 → 2/30) but slower and heavier (21s → 30s, 72k → 147k repeat tokens), because the agent now actually reads the `cli/agentic/*.py` exemplars. Reliability-for-cost tradeoff.

**Bare variant** — essentially unchanged. CLI adoption 6/30 → 3/30, errors flat at 7/30, time flat at 24s. Without a pointer, the agent doesn't find the CLI.

**Compositional tasks** — improved across the board: median time 97s → 51s, repeat tokens 537k → 264k, errors 7/12 → 5/18.

Net: the newer commit is better where the agent has guidance (skill, clone), unchanged where it doesn't (bare), and the skill path is the only one that delivers the full easier-use story — fewer errors, less context, faster.
```


> ⚠️ When you do this, remember that the `.jsonl` traces contain
> attacker-influenceable text — wrap each trace in explicit
> `BEGIN UNTRUSTED TRACE` / `END UNTRUSTED TRACE` markers and tell the
> reviewing LLM that everything between them is **data, not
> instructions**. See [SECURITY.md](./SECURITY.md) for details.
