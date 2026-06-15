# `ag` CLI reference

Full reference for every `ag` subcommand. For the quick start, see
[README.md](./README.md); for the *why* see the
[blog post](./blog/measuring-agentic-use.md); for safety see
[SECURITY.md](./SECURITY.md).

`ag` is profile-based: the first positional `<profile>` defines the **environment**
and the comparison axis (its **bindings**) and the **tiers** ("how much help the
agent gets"). The default `transformers` profile uses git revisions as bindings and
`bare`/`clone`/`skill` as tiers; the `mock` profile is a fast fake (no install, no
real agent) for UI / end-to-end testing.

Runs are stored **one bundle file per (binding, harness, model)** — see
[Result layout](#result-layout) — so runs from different bindings, harnesses, and
models never collide.

Subcommands: `setup`, `run`, `suite`, `diff`, `batch`, `analyze`, `compare`,
`explain`, `tasks`, `upload`, `sync`, `report`.

## Common flags

These appear on most run-producing subcommands:

- `<profile>` (first positional, required) — `transformers` or `mock`. Determines
  how the sandbox is built, the bindings/tiers, and which behavior markers the
  report tracks. Every run/read command is `ag <command> <profile> …`.
- `--runner {claude,pi,mock}` — which coding agent drives each run. `claude`
  (default) shells out to the `claude` CLI; `pi` shells out to the `pi` CLI, which
  serves `--model` via Hugging Face inference providers; `mock` synthesizes fake
  transcripts instantly (pair with the `mock` profile).
- `--model <name>` — the model id. For `claude` a Claude alias/id (`opus`,
  `claude-sonnet-4-6`); for `pi` an HF model id
  (`Qwen/Qwen3-Coder-480B-A35B-Instruct`). Becomes the `<model_id>` path component
  (`/` → `--`), or `default` if omitted. Pass the same `--runner`/`--model` to
  `analyze`/`compare`/`explain`/`upload` to read those results back.
  - The `pi` runner needs `HF_TOKEN` and uses it only for its *own* model calls;
    the token is stripped from the agent's task environment so model downloads stay
    anonymous and comparable to the Claude runs.
- `--tiers <t...>` — restrict to specific tiers (default: all the profile's tiers
  that the binding supports).
- `--tasks <id...>` — restrict to specific task ids (default: all).
- `--runs N` — runs per cell; **overrides** every per-task `runs:` in `tasks.yaml`.
  Without it, each task uses its own `runs:` (cheap tasks default to 5) or 3.
- `-v` / `--verbose` — per-tool-call event lines (default: run-level summaries).
- `--force-rerun` — re-execute cells whose runs already exist (default skips them).
- `--max-tool-calls N` — kill a run after N tool calls (`status=budget_tool_calls`).
  Default 50.
- `--no-live` — disable the rich dashboard (auto-disabled when stderr isn't a TTY).

## Setup / discovery

### `ag tasks`

List the task ids and their categories (`atomic` / `compositional`).

### `ag setup <profile> <ref> [--remove]`

Build (or remove) the per-revision environment. Idempotent. For `transformers`,
the cache holds a git worktree at the resolved SHA, a `uv venv` with the repo
installed, and a derived `SKILL.md` plugin dir (skipped for revisions where the
skill can't be derived). Called implicitly by `run`/`suite`/`diff`; use it to
prebuild caches in parallel.

```bash
ag setup transformers HEAD
ag setup transformers 9914a3641f --remove
```

## Running

### `ag run <profile> <ref> <task_id> <run_index> [tiers...]`

Execute one cell (or one per tier) — handy for ad-hoc probing.

```bash
ag run transformers HEAD classify-sentiment 1            # all tiers
ag run transformers HEAD classify-sentiment 1 bare clone # two tiers
```

### `ag suite <profile> <ref> [--runs N] [--tasks ...] [--tiers ...] [--name ...] [--job]`

Run the full task suite for **one** revision (tiers × tasks × runs). `<ref>` can be
a SHA, branch, or tag — what it was tested as is recorded in
`results/<binding>/ref.json` and badged in the report. `--name "w/ CLI + Skill"`
sets a display title used everywhere in the report.

```bash
ag suite transformers v5.9.0 --runner claude --runs 5
ag suite transformers 4d15b215f3 --runner pi --model <hf-id> --name "w/ CLI + Skill"
ag suite transformers v5.9.0 --runner pi --model <hf-id> --job   # run on HF Jobs
```

**`--job` — run on HF Jobs.** Submits the suite as a detached
[HF Job](https://huggingface.co/docs/huggingface_hub/guides/jobs). The job
bootstraps uv + the `pi` CLI + clones of `transformers` and this repo, mounts the
bucket read+write at `/bucket`, seeds local `results/` from the bucket so completed
cells are skipped, and **persists each run to the bucket the moment it finishes**
(so a crash/eviction keeps every completed run; a SIGTERM also leaves a breadcrumb
in the job log). Tune with `--flavor` (default `t4-medium`, 100 GB; `t4-small`'s
50 GB evicts under the model cache), `--timeout` (default `4h`), `--image`,
`--bucket`. Requires `--runner pi` + `--model`. Track with `hf jobs ps` /
`hf jobs logs <id>`; pull with `ag report transformers --pull`.

### `ag diff <profile> <spec> [...]`

End-to-end: take a ref range (`A..B` or `A..B..C`), build each revision, run the
suite for each, and print the comparison report. Iteration is **task-first**, so
ctrl-C leaves finished tasks with equal sample sizes across refs.

```bash
ag diff transformers v5.8.0..v5.9.0 > progress.md
```

## Batch (matrix of HF Jobs)

### `ag batch <file.yaml> [--submit] [--watch] [--status] [--per-task] [--no-skip-complete]`

Launch a **model × revision** matrix as detached HF Jobs (one per cell; `claude`
cells run locally). Dry-run by default — prints the plan; `--submit` launches.

```yaml
# eval.yaml
profile: transformers
tasks: [classify-sentiment, fill-mask, image-classify]   # optional (default: all)
tiers: [bare, clone, skill]                              # optional (default: all)
runs: 5                                                  # optional
flavor: t4-medium                                        # optional
models:
  - claude                                # → runs locally
  - Qwen/Qwen3-Coder-30B-A3B-Instruct     # "<org>/<id>" → pi runner
  - {model: custom-thing, runner: pi}     # explicit
revisions:
  - v5.8.0
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
# per_task: true        # same as --per-task
# force_rerun: true     # same as --force-rerun
```

```bash
ag batch eval.yaml                                  # dry-run: print the plan
ag batch eval.yaml --submit --watch                # launch + poll to completion
ag batch eval.yaml --submit --watch --per-task     # one job per model × revision × task
```

- **`--submit`** — actually launch (pi cells → HF Jobs, claude cells → local).
- **`--watch`** — poll the jobs until terminal; for any that didn't COMPLETE, print
  the **tail of their logs** so failures are visible inline. `--poll N` sets the
  interval (default 30s).
- **`--status`** — don't launch; report the current stage of the batch's
  already-submitted jobs (recorded under `batches/<name>.json`). Combine with
  `--watch` to follow them.
- **`--skip-complete` / `--no-skip-complete`** — *on by default*: before launching,
  read the bucket and **skip cells already fully present**, and flag
  **partially-done** cells (a prior job was likely killed) for relaunch.
  `--no-skip-complete` disables the check.
- **`--per-task`** — one job per (model × revision × **task**) instead of per
  (model × revision). Smaller, isolated, more parallel jobs (a failure on one task
  no longer blocks the rest), at the cost of rebuilding the env per job.
- **`--force-rerun`** — recompute everything (disables skip-complete).

## Inspection

All inspection commands read `results/` and need neither `claude` nor a venv; they
tolerate partially-written runs.

### `ag analyze <profile> <binding> [task_id]`

Per-revision markdown report (one task or all). `--model`/`--runner` select the
namespace.

### `ag compare <profile> <refs...>`

Side-by-side markdown comparison across two or more refs already on disk (accepts
separate tokens or an `A..B..C` range). Use `ag diff` to build + compare in one shot.

### `ag explain <profile> <tier> <task> <refs...>`

Focused per-cell timeline for one (tier, task) across refs: per-run header
(✓/✗ match, elapsed, exit, tool-call & error counts, tokens), the tool-call
timeline with error snippets, the final answer with a match flag, and a metric diff
across refs. Auto-detects the namespace if the requested one is empty.

## Trace upload

### `ag upload <user>/<dataset> [--push] [--public]`

Package the native agent sessions (captured for every run) into a Hub dataset that
renders in the [agent-traces viewer](https://huggingface.co/docs/hub/agent-traces).
Each run's session is unpacked from the bundled `traces/` into an individual file.
**Dry-run by default** (stages + prints the `hf upload` command); `--push` uploads;
datasets are **private** unless `--public`. Takes the same `--runner`/`--model`.

## Bucket sync

### `ag sync [<namespace>/<bucket>] [--push] [--pull] [--public] [--delete]`

Mirror the local run state with a Hugging Face
[**bucket**](https://huggingface.co/docs/huggingface_hub/en/guides/buckets) via
`hf buckets sync`. Mirrors the whole `results/` + `traces/` trees plus a generated
`results/MANIFEST.json`. Bucket id defaults to `lysandre/transformers-agentic-use`.

```bash
ag sync                       # DRY RUN: refresh the manifest + print the plan
ag sync --push                # create if needed + push results/ + traces/
ag sync --pull                # pull down (mirror: bucket is the source of truth)
```

- **Dry-run by default** (no `--push`/`--pull`).
- **`--pull` mirrors** — it removes local files absent from the bucket, so the
  report reflects exactly what's in the bucket (no lingering local-only runs).
- `--push` ensures the bucket exists then pushes; `--delete` also prunes
  bucket-side files absent locally. Buckets are **private** unless `--public`.

Requires the `hf` CLI + `hf auth login`.

## Report

### `ag report <profile> [refs...] [--pull] [--push] [--space ID] [--public] [--open] [--bucket ID]`

Generate a **self-contained static HTML report** (`report/index.html`) over the
runs under `results/`. One theme-aware page (honors the Space's `?__theme=` then the
OS theme) with a global config in a **⚙ gear sidebar** (which models / tiers /
tasks, fair-comparison toggles) and a top **Revisions** strip you click to
include/exclude commits everywhere. Three tabs:

- **Overview** — `match %`, `median time`, `median new/repeat/out tokens`,
  `runs with error %` over a chosen X axis (revision / model / tier) and series,
  plus a configurable **label-adoption** chart (per-marker, with the marker's
  one-line description) and per-run **distributions** (box+strip, log/linear, by
  revision or per-model). A run counts as "errored" only if it ended badly (bad
  status / nonzero exit) — recovered tool retries are shown separately. The
  fair-comparison toggles restrict to tasks shared across models and/or revisions.
- **Coverage** — a task × revision heatmap of **`done / expected`** runs (expected
  accounts for the tiers a revision supports and per-task run counts); click a cell
  for the per-model × tier breakdown with the specific missing run indices.
- **Results** — every task (its prompt, the input image/audio inline, the match
  rule) in an accordion, plus the full **run table** (group by model/task, fold,
  drill into runs). Per model you see matched/failed/no-answer at a glance and can
  click to read the failing responses.

```bash
ag report transformers                                   # → report/index.html
ag report transformers --pull --open                     # mirror from bucket, then open
ag report transformers --pull --push --space your-org/your-report
```

- `--pull` — mirror down from the bucket first (`ag sync --pull`).
- `--push` — upload `report/` as a **static HF Space** (default id
  `lysandre/transformers-agentic-use-report`, override with `--space`); **private**
  unless `--public`. The `report/` dir is complete Space content (`index.html` +
  `plotly.min.js` + `README.md` with `sdk: static`).
- `--bucket` — bucket used by `--pull` and for trace pointers (default
  `lysandre/transformers-agentic-use`).
- `--open` — open the report in your browser.

## Result layout

Runs are stored **binding-first, one bundle file per (binding, harness, model)** —
keeping the bucket object count low so sync stays fast:

```
results/<binding>/<harness>/<model_id>.jsonl   # one JSON line per run
traces/<binding>/<harness>/<model_id>.jsonl    # one line per run (native session)
results/<binding>/ref.json                     # what the binding was tested as (ref/name/kind/profile)
results/MANIFEST.json                          # generated index of what ran
```

- **`<binding>`** — the 10-char short SHA (for `transformers`).
- **`<harness>`** — `claude`, `pi`, or `mock`.
- **`<model_id>`** — the model name with `/` → `--`, or `default`.

Each **results** line is a complete run:
`{"tier", "task", "run", "meta": {...}, "events": [ …canonical transcript… ]}`.
`meta` carries the resolved SHA, runner, model, status (`ok` / `empty` / `timeout`
/ `budget_tool_calls` / `error`), tool-call count, elapsed, exit code, token
accounting, and the (redacted) command. Each **traces** line is
`{"tier", "task", "run", "raw": "<native session text>"}`.

> Migrating from the old one-file-per-run layout? `scripts/migrate_bucket.py`
> repacks an existing bucket into this format (dry-run by default;
> `--bucket <id> --apply` does the full pull → repack → push). See
> [`scripts/`](./scripts) for that and the clean/remove-a-model helpers.

## Workflows

```bash
# Local before/after, one command:
ag diff transformers v5.8.0..v5.9.0 > progress.md

# Fleet of models × revisions on HF Jobs, resumable:
ag batch eval.yaml --submit --watch            # re-run to fill gaps (auto-skips done cells)

# Refresh + publish the dashboard:
ag report transformers --pull --push --space your-org/your-report
```
