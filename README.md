<p align="center">
  <img width="100%" alt="agent-eval report" src="https://github.com/user-attachments/assets/f8d3962d-cd28-4891-90e1-dc54ad841989" />
</p>

# `agent-eval` — is your library *agentic enough*?

`agent-eval` measures **how a coding agent actually uses a library** — not just whether it
gets the right answer, but *how it gets there*: reaching for a CLI vs. hand-writing
Python, how many tokens and seconds it burns, and how often it errors. It runs the
same tasks across **library revisions** and **models**, and renders everything as a
single static HTML report you can host as a Hugging Face Space.

Everything environment-specific — including the task suite — lives behind a
**profile**: `transformers` (the reference study) or `mock` (a fast, no-agent
profile for trying the UI). Runs are launched from a YAML matrix as Hugging Face
Jobs, and results are explored in the static report it builds.

> 📝 The *why* — and the findings — are in the [blog post](./blog/measuring-agentic-use.md).

> ⚠️ **Trusted local use only.** The `transformers` profile runs a coding agent with
> bypassed permissions and executes code from whatever revision you point it at, and
> traces can contain prompts/output/paths. See [SECURITY.md](./SECURITY.md) before
> pointing it at code you didn't write or sharing results. (The `mock` profile runs
> no agent and is always safe.)

## Install

```bash
git clone https://github.com/huggingface/is-transformers-agentic-enough
cd is-transformers-agentic-enough
uv venv --python 3.13 .env
uv pip install --python .env/bin/python -e .
```

The `transformers` profile expects a `transformers` checkout at `../transformers`
(override with `AE_TRANSFORMERS_SRC`). Runtime state lands next to the cwd
(override with `AE_DATA_DIR`).

## 1. Define your matrix and launch it

Launching is YAML-only: declare the open models × revisions and launch the matrix
as detached Hugging Face Jobs.

```yaml
# eval.yaml
profile: transformers
tasks: [classify-sentiment, fill-mask, image-classify]   # omit for all tasks
runs: 5
flavor: t4-medium
models:                                   # open models, served on HF inference providers
  - Qwen/Qwen3-Coder-30B-A3B-Instruct
  - google/gemma-4-31B-it
revisions:
  - v5.8.0
  - v5.9.0
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
```

```bash
export HF_TOKEN=hf_...
agent-eval batch eval.yaml                              # dry-run: print the plan only
agent-eval batch eval.yaml --submit --watch            # one job per model × revision
agent-eval batch eval.yaml --submit --watch --per-task # one job per model × revision × task
```

Each revision is run across three **tiers** — how much help the agent gets:
`bare` (nothing) → `clone` (the repo in the working dir) → `skill` (a packaged
Skill). Every run records its transcript, metadata, and native agent session.

Jobs persist each run to a shared Hugging Face **bucket** the moment it finishes,
so a crash never loses completed runs. Re-running the same file automatically
**skips cells already done** and flags partially-done ones (a prior job was likely
killed); `--watch` reports failures with their logs.

## 2. Build and publish the report

Results are explored only here — the report, not the CLI.

```bash
agent-eval report transformers --pull --open                              # refresh + open locally
agent-eval report transformers --pull --push --space your-org/your-report # publish as a static HF Space
```

The report is one self-contained, theme-aware page with three tabs:

- **Overview** — match %, median time, median tokens, error % across your chosen
  axes, plus label-adoption (CLI vs. `pipeline()`, …) and per-run distributions.
- **Coverage** — a task × revision heatmap of `done / expected` runs.
- **Results** — every task (prompt, input image/audio, match rule) and what each
  model answered, with click-through into the failing responses.

Configuration (which models / revisions / tiers / tasks) lives behind the **⚙ gear**.

## More

- **Full CLI reference** — every subcommand and flag: [API.md](./API.md)
- **Security & safe sharing**: [SECURITY.md](./SECURITY.md)
