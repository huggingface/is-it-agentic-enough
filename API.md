# `agent-eval` CLI reference

Full reference for every `agent-eval` subcommand. For the quick start, see
[README.md](./README.md); for the *why* see the
[blog post](./blog/measuring-agentic-use.md); for safety see
[SECURITY.md](./SECURITY.md).

`agent-eval` is profile-based: a `<profile>` defines the **environment** and the
comparison axis (its **bindings**), the **tiers** ("how much help the agent
gets"), and the **task suite**. The default `transformers` profile uses git
revisions as bindings and `bare`/`clone`/`skill` as tiers; the `mock` profile is a
fast fake (no install, no real agent) for UI / end-to-end testing.

The shape of the tool is deliberately narrow:

- **Launching is YAML-only** — `agent-eval batch <file.yaml>` expands a
  model × revision matrix and runs each cell on **Hugging Face Jobs** (the open-model
  `pi` runner; a `mock` cell runs locally). There is no per-task / per-ref CLI
  launcher.
- **Viewing is the report** — results are explored only in the static web report
  that `agent-eval report` builds (and publishes as a Hugging Face Space).

Runs are stored **one shard file per (binding, harness, model, task)** — see
[Result layout](#result-layout) — so runs from different bindings, harnesses,
models, and tasks never collide.

Subcommands: `setup`, `batch`, `upload`, `sync`, `report`. (`suite` exists as the
per-revision worker the Job bootstrap runs inside the container; it is not part of
the user-facing surface and is hidden from `--help`.)

## `agent-eval setup <profile> <ref> [--remove]`

Build (or remove) the per-revision environment. Idempotent. For `transformers`,
the cache holds a git worktree at the resolved SHA, a `uv venv` with the repo
installed, and a derived `SKILL.md` for the `skill` tier (skipped for revisions
where the skill can't be derived). Useful to prebuild caches; jobs build their own
environment from scratch inside the container.

```bash
agent-eval setup transformers HEAD
agent-eval setup transformers 9914a3641f --remove
```

## `agent-eval batch <file.yaml> [--submit] [--watch] [--status] [--per-task] [--no-skip-complete] [--force-rerun]`

The launcher. Read a YAML file declaring the **model × revision** matrix, expand it,
and run each cell as a detached HF Job (a `mock` cell runs locally). **Dry-run by
default** — prints the plan; `--submit` actually launches.

```yaml
# eval.yaml
profile: transformers
runner: pi                                # default runner for bare model ids (pi | mock)
tasks: [classify-sentiment, fill-mask, image-classify]   # optional (default: the profile's tasks)
tiers: [bare, clone, skill]                              # optional (default: all)
runs: 5                                                  # optional (overrides per-task `runs:`)
flavor: t4-medium                                        # optional HF Jobs hardware
timeout: 4h                                              # optional job max duration
image: node:22-bookworm                                  # optional docker image
bucket: lysandre/transformers-agentic-use                # optional results bucket
models:
  - Qwen/Qwen3-Coder-30B-A3B-Instruct     # "<org>/<id>" → pi runner (HF-served)
  - {model: custom-thing, runner: pi}     # explicit
  - {model: smoke, runner: mock}          # mock → runs locally (UI / pipeline testing)
revisions:
  - v5.8.0
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
# per_task: true        # same as --per-task
# force_rerun: true     # same as --force-rerun
```

```bash
agent-eval batch eval.yaml                                  # dry-run: print the plan
agent-eval batch eval.yaml --submit --watch                # launch + poll to completion
agent-eval batch eval.yaml --submit --watch --per-task     # one job per model × revision × task
agent-eval batch eval.yaml --status --watch                # follow already-submitted jobs
```

- **`--submit`** — actually launch (pi cells → HF Jobs, mock cells → local).
- **`--watch`** — poll the jobs until terminal; for any that didn't COMPLETE, print
  the **tail of their logs** so failures are visible inline. `--poll N` sets the
  interval (default 30s).
- **`--status`** — don't launch; report the current stage of the batch's
  already-submitted jobs (recorded under `batches/<name>.json`). Combine with
  `--watch` to follow them.
- **`--skip-complete` / `--no-skip-complete`** — *on by default*: before launching,
  read the bucket (or the local mirror) and **skip cells already fully present**,
  and flag **partially-done** cells (a prior job was likely killed) for relaunch.
- **`--per-task`** — one job per (model × revision × **task**) instead of per
  (model × revision). Smaller, isolated, more parallel jobs (a failure on one task
  no longer blocks the rest), at the cost of rebuilding the env per job.
- **`--force-rerun`** — recompute everything (disables skip-complete).

**Each pi cell runs on HF Jobs.** The job bootstraps uv + the `pi` CLI + clones of
`transformers` and this repo, mounts the bucket read+write at `/bucket`, seeds local
`results/` from the bucket so completed cells are skipped, and **persists each run
to the bucket the moment it finishes** (so a crash/eviction keeps every completed
run). The `pi` runner needs `HF_TOKEN`, and uses it only for its *own* model calls —
the token is stripped from the agent's task environment so model downloads stay
anonymous and comparable.

## `agent-eval upload <user>/<dataset> [--push] [--public] [--runner R] [--model M]`

Package the native agent sessions (captured for every run, one file each under
`traces/`) into a Hub dataset that renders in the
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces).
**Dry-run by default** (stages + prints the `hf upload` command); `--push` uploads;
datasets are **private** unless `--public`. `--runner` (default `pi`) / `--model`
pick the `<harness>/<model_id>` namespace to upload.

## `agent-eval sync [<namespace>/<bucket>] [--push] [--pull] [--public] [--delete]`

Mirror the local run state with a Hugging Face
[**bucket**](https://huggingface.co/docs/huggingface_hub/en/guides/buckets) via
`hf buckets sync`. Mirrors the whole `results/` + `traces/` trees plus a generated
`results/MANIFEST.json`. Bucket id defaults to `lysandre/transformers-agentic-use`.

```bash
agent-eval sync                       # DRY RUN: refresh the manifest + print the plan
agent-eval sync --push                # create if needed + push results/ + traces/
agent-eval sync --pull                # pull down (mirror: bucket is the source of truth)
```

- **Dry-run by default** (no `--push`/`--pull`).
- **`--pull` mirrors** — it removes local files absent from the bucket, so the
  report reflects exactly what's in the bucket (no lingering local-only runs).
- `--push` ensures the bucket exists then pushes; `--delete` also prunes
  bucket-side files absent locally. Buckets are **private** unless `--public`.

Requires the `hf` CLI + `hf auth login`.

## `agent-eval report <profile> [refs...] [--pull] [--push] [--space ID] [--public] [--open] [--bucket ID]`

Generate a **self-contained static HTML report** (`report/index.html`) over the
runs under `results/` — the only way to view results. One theme-aware page (honors
the Space's `?__theme=` then the OS theme) with a global config in a **⚙ gear
sidebar** (which models / tiers / tasks, fair-comparison toggles) and a top
**Revisions** strip you click to include/exclude commits everywhere. Three tabs:

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
agent-eval report transformers                                   # → report/index.html
agent-eval report transformers --pull --open                     # mirror from bucket, then open
agent-eval report transformers --pull --push --space your-org/your-report
```

- `--pull` — mirror down from the bucket first (`agent-eval sync --pull`).
- `--push` — upload `report/` as a **static HF Space** (default id
  `lysandre/transformers-agentic-use-report`, override with `--space`); **private**
  unless `--public`. The `report/` dir is complete Space content (`index.html` +
  `data.js` + `plotly.min.js` + `README.md` with `sdk: static`).
- `--bucket` — bucket used by `--pull` and for trace pointers (default
  `lysandre/transformers-agentic-use`).
- `--open` — open the report in your browser.

## Result layout

Runs are stored **binding-first, one shard file per (binding, harness, model, task)** —
small enough an object count to keep sync fast, but sharded by task so concurrent
jobs never write the same object:

```
results/<binding>/<harness>/<model_id>/<task>.jsonl                 # one JSON line per run
traces/<binding>/<harness>/<model_id>/<tier>__<task>__run<N>.jsonl  # one native session per file
results/<binding>/ref.json                                         # what the binding was tested as (ref/name/kind/profile)
results/MANIFEST.json                                              # generated index of what ran
```

- **`<binding>`** — the 10-char short SHA (for `transformers`).
- **`<harness>`** — `pi` or `mock`.
- **`<model_id>`** — the model name with `/` → `--`, or `default`.
- **`<task>`** — the task id (e.g. `classify-sentiment`); **`<tier>`** is `bare`/`clone`/`skill`.

Each **results** line is a complete run:
`{"tier", "task", "run", "meta": {...}, "events": [ …canonical transcript… ]}`.
`meta` carries the resolved SHA, runner, model, status (`ok` / `empty` / `timeout`
/ `budget_tool_calls` / `error`), tool-call count, elapsed, exit code, token
accounting, and the (redacted) command. Each **traces** file is one run's *native*
agent session, stored verbatim — one session per file so the Hub
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces) auto-detects
and renders it in place (in a dataset or the bucket).

**Why shard by task.** `agent-eval batch --per-task` launches one HF Job per
(model, revision, task), so without sharding many jobs would write the same
`<model>.jsonl` object at once. Object storage has no atomic compare-and-swap and
every write is a whole-file overwrite, so the last writer silently clobbered the
rest — completed runs vanished. A per-task shard is owned by exactly one job, so
writers never collide (no locks, eviction-safe). The read side merges a model's
shards back together, and `hf buckets sync` is incremental, so only newly-written
shards come down on a pull.

## Tasks

The task suite is **defined by the profile**, not the CLI — different profiles run
different tasks. The `transformers` suite lives in
[`src/ae/data/transformers.yaml`](./src/ae/data/transformers.yaml); the `mock`
profile reuses it. Each task is:

```yaml
- id: classify-sentiment        # required: unique id
  category: atomic              # atomic | compositional
  prompt: |                     # required: the developer-style instruction
    Using distilbert/…-sst-2-english, classify the sentiment of "…".
  expected: positive            # optional: scored against the final answer
  match: substring              # optional: substring (default) | exact | regex
  runs: 5                       # optional: samples per cell (default 3)
```

`match: judge` (LLM grading) is reserved but not implemented.

## Workflows

```bash
# Fleet of open models × revisions on HF Jobs, resumable:
agent-eval batch eval.yaml --submit --watch            # re-run to fill gaps (auto-skips done cells)

# Follow in-flight jobs and see failures inline:
agent-eval batch eval.yaml --status --watch

# Refresh from the bucket + publish the report:
agent-eval report transformers --pull --push --space your-org/your-report
```
