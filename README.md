<p align="center">
  <img width="100%" alt="ag report" src="https://github.com/user-attachments/assets/f8d3962d-cd28-4891-90e1-dc54ad841989" />
</p>

# `ag` — is your library *agentic enough*?

`ag` measures **how a coding agent actually uses a library** — not just whether it
gets the right answer, but *how it gets there*: reaching for a CLI vs. hand-writing
Python, how many tokens and seconds it burns, and how often it errors. It runs the
same tasks across **library revisions** and **models**, and renders everything as a
single static HTML report you can host as a Hugging Face Space.

Everything environment-specific lives behind a **profile** (the first CLI argument):
`transformers` (the reference study) or `mock` (a fast, no-agent profile for trying
the UI).

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
(override with `AG_TRANSFORMERS_SRC`). Runtime state lands next to the cwd
(override with `AG_DATA_DIR`).

## 1. Run a suite for one revision

```bash
# Claude Code as the agent, all tiers/tasks, 5 runs each, on the v5.9.0 tag
ag suite transformers v5.9.0 --runner claude --runs 5

# An open model served on HF, a few tasks, on a named branch
export HF_TOKEN=hf_...
ag suite transformers 4d15b215f3 --runner pi \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --name "w/ CLI + Skill" --tasks classify-sentiment fill-mask tokenize-count
```

Each revision is run across three **tiers** — how much help the agent gets:
`bare` (nothing) → `clone` (the repo in the working dir) → `skill` (a packaged
Skill). Every run records its transcript, metadata, and native agent session.

## 2. Many models × revisions on HF Jobs

Declare the matrix in a YAML file and launch it as detached jobs:

```yaml
# eval.yaml
profile: transformers
tasks: [classify-sentiment, fill-mask, image-classify]
flavor: t4-medium
models:
  - claude
  - Qwen/Qwen3-Coder-30B-A3B-Instruct
  - google/gemma-4-31B-it
revisions:
  - v5.8.0
  - v5.9.0
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
```

```bash
ag batch eval.yaml --submit --watch                # one job per model × revision
ag batch eval.yaml --submit --watch --per-task     # one job per model × revision × task
```

Jobs persist each run to a shared Hugging Face **bucket** the moment it finishes,
so a crash never loses completed runs. Re-running the same file automatically
**skips cells already done** and flags partially-done ones (a prior job was likely
killed); `--watch` reports failures with their logs.

## 3. Build and publish the report

```bash
ag report transformers --pull --open                              # refresh + open locally
ag report transformers --pull --push --space your-org/your-report # publish as a static HF Space
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
- **Bucket maintenance scripts**: [`scripts/`](./scripts) (migrate / clean / remove a model)
