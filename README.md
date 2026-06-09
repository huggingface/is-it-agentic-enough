<p align="center">
  <img width="100%" alt="Screenshot 2026-04-28 at 16 19 34" src="https://github.com/user-attachments/assets/f8d3962d-cd28-4891-90e1-dc54ad841989" />
</p>

# `ag` — a profile-based agentic-eval harness

`ag` runs a list of **tasks** through a coding agent inside an **environment**,
and scores each run's answer against an **expected response** — then reports how
behavior and correctness move across an axis you choose.

Everything environment-specific lives behind a **profile** (the first CLI argument, `ag <command> <profile> …`):

- **`transformers`** (default) — the original study: build a git worktree of
  `transformers` at a **binding** (a revision), across three assistance
  **tiers** (`bare`/`clone`/`skill`), and track adoption of the CLI vs
  `pipeline()` etc. via behavior **markers**. `ag diff transformers A..B` compares revisions.
- **`mock`** — a fast, fake profile (no install, no real agent) for exercising
  the UI and smoke-testing the whole pipeline end-to-end in seconds. Pair with
  `--runner mock`: `ag suite mock dev --runner mock`.

The original transformers study is the worked example throughout this README;
the harness core (tasks, tiers, runners, markers, matching, reports) is generic.

> ⚠️ **Trusted local use only.** With the `transformers` profile this runs
> `claude` with `--permission-mode bypassPermissions` and executes code from
> whatever `transformers` ref you point it at. See [SECURITY.md](./SECURITY.md)
> before pointing it at anything you didn't write yourself, and before
> sharing the contents of `results/`. (The `mock` profile runs no agent and is
> always safe.)

## Install

```bash
uv venv --python 3.13 .env
uv pip install --python .env/bin/python -e .
```

This installs the `ag` command into `.env/bin/ag`. The harness assumes
the transformers source repo lives at `../transformers` relative to this
directory; override with `AG_TRANSFORMERS_SRC=/path/to/transformers` if
it's elsewhere. Runtime state (`configs/`, `workspaces/`, `results/`)
lands next to cwd; override via `AG_DATA_DIR`.

## Tiers (the `transformers` profile)

A profile defines **tiers** — "how much help the agent gets." The `transformers`
profile ships three:

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

Each `(binding × tier × task)` is run N times (default 3) to smooth model
non-determinism. A different profile declares its own tiers (the `mock` profile
reuses these three so its reports look the same).

## Behavior markers

A profile declares **markers** — independent, possibly-overlapping named
regexes matched against a run's commands / written code / read paths / final
answer. Each is tracked as adoption (`fired-runs / total-runs`) per cell and
across bindings, so the report shows *"how did adoption of behavior X move
across revisions / model growth."* The `transformers` profile ships `cli`
(invoked the `transformers` CLI), `pipeline` (used `pipeline(...)`), `ran-help`,
and `agentic-exemplar` (read a `cli/agentic/*.py` exemplar). Generic profiles
define their own, or none.

## Scoring

Each task may set `expected:` and a `match:` mode — `substring` (default),
`exact`, or `regex` — and the agent's final answer is checked against it
(the `✓match` signal). Markers measure *behavior*; matching measures
*correctness*.

Refs can be **SHAs, branch names, or tags** (`ag suite transformers main`,
`ag suite transformers v4.56.0`) — anything `git rev-parse` resolves against your
transformers checkout (run `git fetch` there first for fresh branches).
What a commit was tested *as* is recorded in `results/<commit>/ref.json`
and shows up color-coded in the report (`branch` / `release` badges).

## Runners: Claude Code or any model via Pi + HF inference providers

By default the harness drives **Claude Code** (the `claude` CLI). With
`--runner pi` it instead drives **[Pi](https://github.com/badlogic/pi-mono)**
(the `pi` CLI), which can serve *any* model through **Hugging Face inference
providers** (and other providers Pi knows). All three variants work for both
runners — `bare`/`clone` rely on `cwd`-based discovery, and the `skill` variant
reuses the *same* `SKILL.md` (both agents implement the
[Agent Skills standard](https://agentskills.io/specification); Claude loads it
via `--plugin-dir`, Pi via `--skill`).

```bash
# Any HF-served model. HF_TOKEN is used only for Pi's own model calls; it is
# stripped from the agent's task environment so model downloads stay anonymous
# and comparable to the Claude runs.
export HF_TOKEN=hf_...
ag diff transformers A..B --runner pi \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct > progress.md
```

### Running suites on HF Jobs

Add `--job` to run the suite on Hugging Face infrastructure instead of your
machine ([HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)):

```bash
ag suite transformers <ref> --runner pi --model Qwen/Qwen3-Coder-480B-A35B-Instruct --job
# → detached job; track with `hf jobs ps` / `hf jobs logs <id>`,
#   then `ag report transformers --pull` to bring the new runs into the dashboard
```

The job bootstraps everything itself (uv, the `pi` CLI, clones of
`transformers` and this repo), mounts the bucket **read+write** at `/bucket`
so results land in it directly, and seeds its local `results/` from the
bucket first so already-completed cells are skipped — interrupted jobs are
resumable by resubmitting. Each run's workspace (a full transformers worktree
for the `clone` variant) is deleted as soon as its trace is captured, so a long
suite doesn't fill the pod's ephemeral disk. `--flavor` (default `t4-medium`,
100 GB — task-model downloads fill the HF cache; `t4-small`'s 50 GB evicts),
`--timeout` (default `4h` — HF's own default is only 30m), `--image`, and `--bucket`
tune the job. Only the `pi` runner works on Jobs (the `claude` CLI needs
interactive auth; `pi` just needs the `HF_TOKEN` secret, which the harness
strips from the agent's task environment as usual).

Pi's native event stream is normalized to Claude Code's schema at write time, so
the analysis/report commands (`analyze`/`compare`/`explain`) work identically
for both runners. Results are laid out by commit first:
`results/<commit>/<harness>/<model_id>/` (see
[Result layout](#result-layout)), so a Pi/HF run never collides with a Claude
run. Pass the same `--runner`/`--model` to
`analyze`/`compare`/`explain`/`upload`/`sync` to read them back. Note: HF
inference providers generally don't prompt-cache, so
the `repeat` (cache-read) token column is ~0 for Pi runs — read `new`≈input and
`out`=output. `ag explain` shows explicit `tokens in:/out:` per run and
median in/out per cell.

## Uploading traces to the Hugging Face Hub

Every run persists the agent's **native** session file (Claude Code / Pi)
under `traces/<commit>/<harness>/<model_id>/`, mirroring the
[result layout](#result-layout) — sharing traces is the whole point, so this
is unconditional. These are natively rendered by the Hub
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces). `ag
upload` packages them into a dataset (with a `traces`-tagged card) and uploads
via the `hf` CLI:

```bash
ag suite transformers <ref> --runner pi --model <id>
ag upload <user>/<dataset> --runner pi --model <id>          # DRY RUN
ag upload <user>/<dataset> --runner pi --model <id> --push   # actually upload
```

Uploads are **dry-run by default** (nothing is pushed without `--push`) and
datasets are created **private** unless you pass `--public`. Review traces
before publishing — they can contain prompts, command output, and local paths.

## Commands

The full CLI reference — every subcommand, every flag, the typical
workflows — lives in [API.md](./API.md). At a glance:

Every run/read command takes the **profile** as its first positional
(`ag <command> <profile> …`) — the profile dictates the environment; the
revision only varies within it.

```bash
ag tasks                                       # list the task ids
ag setup <profile> <ref> [--remove]            # build / tear down per-revision env
ag run <profile> <ref> <task> <run_index> [tiers...]  # one cell, ad-hoc
ag suite <profile> <ref>                       # full suite for one revision
ag diff <profile> <ref1>..<ref2>[..<refN>]     # end-to-end: run + compare
ag analyze <profile> <binding> [task_id]       # per-revision markdown report
ag compare <profile> <refs...>                 # cross-revision diff table
ag explain <profile> <tier> <task> <refs...>   # per-cell tool-call timeline
ag upload <user>/<dataset>                     # push captured traces to the Hub (dry-run by default)
ag sync [<namespace>/<bucket>]                 # mirror results/ + traces/ + manifest with the HF bucket (dry-run)
ag report <profile> [refs...]                  # static HTML report (charts + drill-down); --push → HF Space
```

Add `--runner pi --model <id>` to any run-producing
command (`run`/`suite`/`diff`) to evaluate an HF-served model instead of Claude.

Most-common path: `ag diff transformers A..B > progress.md` builds caches, runs the
suite on each ref, and prints the comparison report in one shot. While
it's running you can run `ag explain transformers <tier> <task> A..B` from
another terminal to drill into any cell that looks weird in the live
dashboard.

## Worked example

The checked-in [`progress.md`](./progress.md) was produced by running the
harness against two commits straddling the agent-first CLI work.

```bash
ag suite transformers 0ea540efff           # "before" — /v1/completions endpoint commit
ag suite transformers 59e4754341           # "after"  — bugfixes after the CLI landed
ag compare transformers 0ea540efff 59e4754341 > progress.md
```

Each `(commit × variant × task)` cell ran 3 times, so a full suite for one
commit is 3 variants × 8 tasks × 3 runs = 72 runs (skipping `skill` for the
earlier commit, which predates the manifest).

### Result layout

Runs are stored commit-first, so everything for one commit lives under one
directory tree:

```
results/<commit>/<harness>/<model_id>/<variant>__<task>__run<N>.jsonl   (+ .meta.json)
```

- **`<commit>`** — the 10-char short SHA the run was executed against.
- **`<harness>`** — the coding agent that drove the run: `claude` or `pi`.
- **`<model_id>`** — the model name with `/` replaced by `--` so it stays a
  single path segment, or `default` when `--model` is omitted. For `claude`
  e.g. `opus`; for `pi` (always HF-served) e.g.
  `Qwen--Qwen3-Coder-480B-A35B-Instruct`.

So a default Claude run lands at
`results/0ea540efff/claude/default/bare__classify-sentiment__run1.jsonl`, and a
Pi/HF run at
`results/0ea540efff/pi/Qwen--Qwen3-Coder-480B-A35B-Instruct/bare__classify-sentiment__run1.jsonl`.
Native session traces mirror this exact tree under `traces/` (captured on
every run).

For every individual run the harness writes two files:

- `<variant>__<task>__run<N>.jsonl` — the raw Claude Code
  stream-json transcript: `system` events (session start, hook fires),
  every `assistant` turn with its tool-call arguments, every `user` turn
  carrying the `tool_result` payload (with `is_error` flags), and the
  final `result` event. This is the full trace of what the agent saw,
  what it ran, and what came back — nothing is summarised away.
- `<variant>__<task>__run<N>.meta.json` — a small sidecar:
  resolved SHA, variant, task id, run index, runner, model,
  status, tool-call count, wall-clock seconds, exit code, token accounting
  (`input`, `output`, `cache_read`, `cache_creation`), the exact `claude`
  command that was executed, and the workspace path.

For the two commits above, that's 6 variants-with-data × 8 tasks × 3 runs =
~144 `.jsonl` + 144 `.meta.json` files under `results/`.

### Syncing with the bucket

`ag sync` mirrors `results/` + `traces/` and a generated
`results/MANIFEST.json` (the record of *which* configs/commits were run —
per-commit git subject/date plus the set of harness/model/variant/task/run
cells present) to a Hugging Face **[bucket](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)**
(S3-like Xet object storage) via `hf buckets sync`. It is **dry-run by default**:

```bash
ag sync                       # DRY RUN: refresh the manifest + print the sync plan
ag sync --push                # create bucket if needed + sync results/ + traces/ + manifest up
ag sync --pull                # sync results/ + traces/ back down from the bucket
ag sync --push --delete       # also prune bucket files that no longer exist locally
```

The bucket defaults to `lysandre/transformers-agentic-use`; pass a different
`<namespace>/<name>` as the first argument. Buckets are created **private**
unless you pass `--public`. `results/` and `traces/` land under matching
prefixes in the bucket (`hf://buckets/<id>/results`, `.../traces`).
`hf buckets sync` only transfers files that changed, and because the layout is
commit-first, syncing different commits (or different machines) never
overwrites unrelated runs — each `--push` just adds/refreshes that commit's
subtree (unless you pass `--delete`).

### Visualizing: `ag report`

`ag report` distills every run under `results/` into a **single static
HTML page** with interactive Plotly charts — cross-commit trends, model-vs-model
comparison, a per-task heatmap with click-through drill-down to individual runs
(tool-call timeline, errors, final answer, trace pointer), and token/duration
distributions. The run records are embedded as JSON and rendered client-side,
so the page keeps its filters (commit / model / variant / task) with no server.

```bash
ag report transformers                     # write report/index.html + print the path
ag report transformers --pull --open       # refresh from the bucket, then open in a browser
ag report transformers --push              # publish as a private static HF Space
```

The `report/` directory (`index.html` + `plotly.min.js` + `README.md` with
`sdk: static`) is complete Space content — `--push` uploads it via the `hf`
CLI, or add the Space as a git remote and push it yourself. Like
`sync`/`upload`, nothing is published without an explicit `--push`.

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
