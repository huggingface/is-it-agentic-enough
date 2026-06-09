# `ag` CLI reference

Full reference for every `ag` subcommand. For a high-level overview of the
harness — profiles, tiers, markers, matching — see [README.md](./README.md);
for safety properties see [SECURITY.md](./SECURITY.md).

`ag` is profile-based: the first positional `<profile>` defines the **environment** and the
comparison axis (its **bindings**), and the **tiers** ("how much help the agent
gets"). The default `transformers` profile uses git revisions as bindings and
`bare`/`clone`/`skill` as tiers; the `mock` profile is a fast fake for UI / E2E
testing. Runs are stored binding-first as
`results/<binding>/<harness>/<model_id>/<tier>__<task>__runN.jsonl`
(see [Result layout](./README.md#result-layout)), so runs from different
bindings, harnesses, and models never collide.

## Common flags

These appear on most run-producing subcommands:

- `<profile>` (first positional, required) — the environment profile
  (`transformers` or `mock`). Determines how the sandbox is built, what the
  bindings/tiers are, and which behavior markers the report tracks. Every run/
  read command is `ag <command> <profile> …`.
- `--runner {claude,pi,mock}` — which coding agent drives each run. `claude`
  (default) shells out to the `claude` CLI (your configured Claude model);
  `pi` shells out to the `pi` CLI, which serves the `--model` via Hugging
  Face inference providers; `mock` synthesizes fake transcripts instantly
  (pair with the `mock` profile).
- `--model <name>` — the model id. For `--runner claude` it is a Claude
  alias/id (`opus`, `claude-sonnet-4-6`) passed to `claude --model`. For
  `--runner pi` it is an HF model id (`Qwen/Qwen3-Coder-480B-A35B-Instruct`).
  Either way it becomes the `<model_id>` path component (`/` → `--`), or
  `default` if omitted. Pass the same `--runner`/`--model` to
  `analyze`/`compare`/`explain`/`upload`/`sync` to read those results back.
  - The `pi` runner requires `HF_TOKEN` (it always uses the `huggingface`
    provider) and uses it only for its *own* model calls (via `--api-key`);
    the token is stripped from the agent's task environment, matching the
    Claude runs.
- `-v` / `--verbose` — emit per-tool-call event lines from each run
  (default is run-level `▶` / `■` summaries only).
- `--force-rerun` — re-execute cells whose `.jsonl` already exists
  (default skips them, since each run costs API tokens + wall time).
- `--max-tool-calls N` — kill a run after this many tool calls; the run's
  `meta.json` records `status=budget_tool_calls`. Default 50. Protects
  against pathological agents that loop forever.
- `--no-live` — disable the rich dashboard (auto-disabled when stderr
  isn't a TTY). Use in CI logs.

## Setup / discovery

### `ag tasks`

List the 8 task ids and their categories (`atomic` / `compositional`).

```bash
ag tasks
```

### `ag setup <profile> <ref> [--remove]`

Build (or remove) the per-commit cache for one ref. Idempotent: rerunning
on an already-built cache is a no-op. Each cache contains:

- `configs/<short-sha>/worktree/` — git worktree of `transformers` at
  the resolved SHA,
- `configs/<short-sha>/.venv/` — a `uv venv` with `pip install -e
  worktree` and the pinned runtime deps (`torch`, `librosa`, ...),
- `configs/<short-sha>/plugin/` — a Claude Code plugin dir holding a
  `SKILL.md` rendered from the commit's derived skill manifest (skipped
  silently for commits that predate the skill-derivation module),
- `configs/<short-sha>/.ready` — a sentinel.

```bash
ag setup transformers HEAD                # build
ag setup transformers 9914a3641f --remove # tear down (~2 GB freed)
```

Called implicitly by `ag run` / `ag suite` / `ag diff` if needed,
but you can prebuild caches in parallel before kicking off a long suite.

## Running

### `ag run <profile> <ref> <task_id> <run_index> [tiers...]`

Execute exactly one run, or one run per variant. Cheap to use for ad-hoc
probing of a single (commit, task) cell.

```bash
ag run transformers HEAD classify-sentiment 1                # all 3 variants
ag run transformers HEAD classify-sentiment 1 bare           # just bare
ag run transformers HEAD classify-sentiment 1 bare clone     # two variants
```

Accepts the common flags above. Skips a cell if its `.jsonl` already
exists unless `--force-rerun` is set.

### `ag suite <profile> <ref> [--runs N] [--tasks ...] [--tiers ...]`

Run the full task suite for **one** commit (3 variants × 8 tasks). `<ref>`
can be a SHA, a branch name, or a tag (`main`, `v4.56.0`, …) — what it was
tested as is recorded in `results/<commit>/ref.json` and badged in the
report. `--name "kv-cache rewrite"` adds an experiment title to the same
marker: it becomes the commit's display name everywhere in the report
(scoreboard, axes, chips, drill-down), with the ref/sha demoted to the
detail line and the branch/release badge kept. Re-running with `--name`
updates the title; runs without it keep the existing one. The number of
runs per cell is resolved per task: an explicit `--runs N`
**overrides every** per-task `runs:` in `tasks.yaml`; without `--runs`,
each task uses its own `runs:` (cheap tasks default to 5) or 3 if it has none.

```bash
ag suite transformers HEAD                                   # per-task runs: (or 3)
ag suite transformers HEAD --tiers skill --tasks summarize-text caption-image
ag suite transformers HEAD --runs 5                          # force 5 for ALL tasks
ag suite transformers HEAD --runner pi --model <hf-id> --job # run it on HF Jobs instead
```

**`--job` — run on HF Jobs.** Submits the suite as a detached
[HF Job](https://huggingface.co/docs/huggingface_hub/guides/jobs) instead of
executing locally. The job bootstraps uv + the `pi` CLI + clones of
`transformers` and this repo, mounts the bucket read+write at `/bucket`
(results land in it directly, no upload step), and seeds local `results/`
from the bucket first so completed cells are skipped — resubmitting after an
interruption resumes where it left off. Tune with `--flavor` (default
`t4-medium`, 100 GB ephemeral; `t4-small`'s 50 GB evicts under the HF model
cache), `--timeout` (default `4h`; HF's own default is 30m), `--image`
(default `node:22-bookworm`; any apt-capable image works), `--bucket`.
Requires `--runner pi` + `--model` (the `claude` CLI can't authenticate on
Jobs) and passes your `HF_TOKEN` as the job secret. Branch/tag refs are
validated against the GitHub remote at submit time (`git ls-remote`) with
did-you-mean suggestions, so a typo'd ref fails in ~1s locally instead of
minutes into a paid job. Track with `hf jobs ps` / `hf jobs logs <id>`;
pull results with `ag report transformers --pull`.

Progress shows in the rich dashboard: a panel header, a counters line,
and a table with rows = (task, variant) and one column for the ref.
Log lines (`[3/72] → ...`) scroll above. With `-v`, every tool call is
logged.

### `ag diff <profile> <spec> [--runs N] [--tasks ...] [--tiers ...]`

The **end-to-end** path: takes a ref range (`A..B` or `A..B..C`),
ensures every commit's cache is built, runs the suite for each, and
prints the comparison report on stdout when finished. This is what you
run for an actual before/after measurement — no need to invoke `setup` /
`suite` / `compare` separately.

```bash
ag diff transformers 0ea540efff..59e4754341 > progress.md
ag diff transformers A..B..C --runs 5 --tasks summarize-text caption-image
```

Iteration order is **task-first**: each task completes on all
refs × variants × runs before the next task begins. So if you
ctrl-C halfway, every fully-finished task already has equal sample
sizes across refs and is comparable.

The rich dashboard lights up with rows = (task, variant) and one
column per ref; cells fill in as runs finish, color-coded green/red
row-relative (best/worst across refs) on median time and tool count,
with `⏻` / `!` flags for aborted / failed runs. While it's running,
you can introspect any cell from another terminal with `ag explain`
(see below).

## Inspection

All inspection commands read from `results/<commit>/<harness>/<model_id>/`
and require neither `claude` nor a venv. They're safe to run **while a diff
is in progress**
— they tolerate in-flight `.jsonl` files (last-line partial writes) and
missing `.meta.json` sidecars.

### `ag analyze <profile> <short-sha> [task_id]`

Per-commit markdown report. With a task id, only that task; without one,
every task that has results for the given SHA. Useful as a single-commit
deep dive before/after a `compare`.

```bash
ag analyze transformers 59e4754341
ag analyze transformers 59e4754341 caption-image --model sonnet > caption.md
```

### `ag compare <profile> <refs...>`

Side-by-side comparison across two or more refs already on disk.
Produces the self-describing markdown report (preamble + variant
definitions + commit metadata + metric glossary + headline + per-variant
summaries + per-task tables) that the worked example in the README shows.

```bash
ag compare transformers 0ea540efff 59e4754341 > progress.md
ag compare transformers 9914a3641f..03836b6ec6..8135eabc1c
```

Accepts refs as separate tokens or as a `A..B..C` range (same as `diff`).
Results must already exist; use `ag diff` to build + compare in one
shot.

### `ag explain <profile> <tier> <task> <refs...>`

Focused per-cell timeline for **one** (variant, task) cell across one or
more refs. The drill-down complement to `compare` — when a cell looks
weird in the dashboard, this is what you run to find out *why*.

```bash
ag explain transformers bare summarize-text 0ea540efff..59e4754341 --model sonnet-old
ag explain transformers skill caption-image 59e4754341
```

For each ref it prints, per run on disk:

- a one-line header (✓/✗ match, elapsed, exit code, tool-call count, error count, `tokens in:/out:`),
- the full tool-call timeline with `❗` markers + first-line snippets on errored calls,
- the final answer truncated, with a `[contains '...']` / `[missing '...']` flag against the task's expected substring,

followed by a side-by-side metric diff (approach bucket, errors, median
time, median tools, median tokens in/out, match rate) when ≥2 refs are
given, and the list of `.jsonl` trace paths at the bottom for hand-off to
an LLM (wrap in `BEGIN/END UNTRUSTED TRACE` markers per
[SECURITY.md](./SECURITY.md)).

If the requested `<harness>/<model_id>` namespace has nothing for the cell,
`explain` auto-detects the right namespace (or, if multiple namespaces have
data, lists them so you can pick).

## Trace upload

### `ag upload <user>/<dataset>`

Upload the native agent session files captured under
`traces/<commit>/<harness>/<model_id>/`
(every run captures one — sharing traces is the point of the harness) to a
Hugging Face Hub dataset, where they render in the
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces). Takes
the same `--runner`/`--model` flags to resolve the trace
namespace.

```bash
# run the suite (native sessions are captured automatically) …
ag suite transformers 59e4754341 --runner pi --model <id>
# … then upload them
ag upload me/transformers-agent-traces --runner pi --model <id>          # DRY RUN
ag upload me/transformers-agent-traces --runner pi --model <id> --push   # upload
```

- **Dry-run by default.** Without `--push` it stages the files, writes a
  `traces`-tagged dataset card, and prints the exact `hf upload` command —
  but uploads nothing.
- `--push` runs the upload (requires the `hf` CLI and `hf auth login`).
- Datasets are created **private** unless you pass `--public`. Traces may
  contain prompts, tool output, local paths, and secrets — review before
  publishing.

## Bucket sync

### `ag sync [<namespace>/<bucket>]`

Mirror the local run state with a Hugging Face
[**bucket**](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)
(S3-like Xet object storage) via `hf buckets sync`. Unlike `upload` (which
packages traces for one namespace as a standalone dataset), `sync` mirrors the
**whole** `results/` and `traces/` trees plus a generated
`results/MANIFEST.json` — the record of *which* configs/commits were run
(per-commit git subject/date and the set of harness/model/variant/task/run
cells present). The bucket id defaults to `lysandre/transformers-agentic-use`.

```bash
ag sync                       # DRY RUN: refresh the manifest + print the sync plan
ag sync --push                # create bucket if needed + sync results/ + traces/ up
ag sync --pull                # sync results/ + traces/ back down
ag sync me/other-bucket --push  # target a different bucket
ag sync --push --delete       # prune bucket files no longer present locally
```

- **Dry-run by default.** Without `--push`/`--pull` it (re)writes
  `results/MANIFEST.json`, prints a per-commit summary, and shows the exact
  `hf buckets create` / `hf buckets sync` commands — but transfers nothing.
- `--push` ensures the bucket exists (`hf buckets create --exist-ok`) then
  syncs `results/` and `traces/` up to `hf://buckets/<id>/results` and
  `.../traces`. `--pull` syncs them back down, then refreshes the local
  manifest. `hf buckets sync` only transfers changed files.
- `--delete` adds rsync-style `--delete` to the sync (remove receiver-side
  files absent on the sender). Off by default — sync only adds/updates.
- Buckets are created **private** unless you pass `--public`. Because the
  layout is commit-first, pushing a new commit only adds/refreshes that
  commit's subtree — runs for other commits in the bucket are untouched. Same
  safety caveats as `upload`: review before publishing.

Requires the `hf` CLI (`huggingface_hub` with bucket support) and
`hf auth login`.

## Report

### `ag report <profile> [refs...]`

Generate a **self-contained static HTML report** over the runs under
`results/` — a single `report/index.html`, organized **commit-first** so the
top of the page answers "which commit is doing better?" at a glance:

1. **Scoreboard** — one column per commit (date order), every metric as a row
   (CLI adoption %, match %, errored-calls %, failed-runs %, median time,
   median new/out tokens), best/worst commit highlighted green/red. Commits
   tested as a branch or tag show their ref name with a color-coded badge
   (`branch` = indigo, `release` = green; plain commits are unbadged) — the
   label is captured at run time in `results/<commit>/ref.json`, with a
   `git tag --points-at` fallback for older data.
2. **Cross-commit trend** — the selected metric across commits; one aggregate
   line by default, with a "split lines by" selector for variant or
   harness/model breakdowns.
3. **Per-task heatmap** — task × **commit** grid colored by the selected
   metric; clicking a cell drills into its runs: status/exit flags, tool-call
   timeline pills (CLI highlighted, errors red with snippet tooltips),
   final-answer snippet, trace pointer.
4. **Distributions** — box plots (every run an individual point) for elapsed
   time and new/repeat/out tokens, grouped by commit.
5. **Model vs model** — secondary cut: the metric grouped by
   `harness/model_id`, one bar per commit.

Run records are embedded as JSON and charts render client-side, so the page
stays interactive (filters for commits / models / variants / tasks) while
being a plain static file. All parsing reuses the same code paths as
`analyze`/`explain` — numbers always agree across the three views.

```bash
ag report transformers                     # all commits on disk → report/index.html
ag report transformers 0f0036c888          # restrict to specific refs / ranges
ag report transformers --pull --open       # bucket → local → report → browser
ag report transformers --push              # publish as a private static HF Space
```

- `--pull` — run the bucket pull (`ag sync --pull`) first.
- `--push` — upload `report/` as a **static HF Space** (default id
  `lysandre/transformers-agentic-use-report`, override with `--space`);
  otherwise the upload plan is only printed. Spaces are **private** unless
  `--public`. The `report/` dir is complete Space content (`index.html`,
  `plotly.min.js`, `README.md` with `sdk: static`), so you can equally add
  the Space as a git remote and push it manually.
- `--bucket` — bucket id used by `--pull` and for trace pointers in the
  drill-down (default `lysandre/transformers-agentic-use`).
- `--open` — open the generated report in your browser.
- `plotly.min.js` is fetched once (pinned version) and cached next to the
  report so the published Space is self-contained; if the fetch fails the
  page falls back to the CDN.

## Workflows

Three typical entry points:

```bash
# 1. End-to-end: run + compare in one command (most common).
ag diff transformers 0ea540efff..59e4754341 > progress.md

# 2. Stage by stage, e.g. when you want to inspect intermediate state.
ag setup transformers 0ea540efff
ag setup transformers 59e4754341
ag suite transformers 0ea540efff
ag suite transformers 59e4754341
ag compare transformers 0ea540efff 59e4754341 > progress.md

# 3. Add a third commit to an existing comparison without re-running A vs B.
ag suite transformers 03836b6ec6
ag compare transformers 0ea540efff 03836b6ec6 59e4754341 > progress.md
```

`compare` (and the comparison appended by `diff`) produces one table per
task, columns per SHA, cells showing the approach bucket —
`CLI-clean=2/3 Python-retry=1/3` etc. Clean CLI-ward flips stand out
immediately; regressions are equally visible.
