# transformers agent behavior: 0ea540efff → 59e4754341  [model: sonnet]

## Context

This report was produced by the `is-transformers-agentic-enough` harness.
The harness runs headless Claude Code against a fixed set of tasks, using
a pinned build of the `transformers` library at each commit being compared.
The goal is to measure how an agent's *behaviour* changes across commits —
specifically, whether it uses the `transformers` CLI (the subject of the
9-commit "agent-first CLI" effort) vs. falling back to writing Python.

Each (commit × variant × task) cell is typically run N times (default 3)
to smooth out model non-determinism. The stats below report medians and
totals over those runs.

## Variants

Each run happens under one of three "variants" that differ only in how the
transformers CLI is surfaced to the agent. Installed transformers version
is identical across variants for a given commit; the difference is context.

- **bare** — `pip install transformers` only. Workspace contains only task
  inputs. No pointer to the CLI exists; the agent would need to discover it
  on its own (e.g. by running `transformers --help`).
- **clone** — same install, but the agent's cwd **is** a git worktree of the
  transformers repo at that commit. `AGENTS.md`, `CLAUDE.md`, and
  `src/transformers/cli/agentic/*.py` auto-discover via Claude Code's
  standard cwd scanning.
- **skill** — same install, plus a Claude Code plugin directory containing
  a generated `SKILL.md` is loaded with `--plugin-dir`. SKILL.md explicitly
  tells the agent "for atomic tasks, use the CLI". Skipped for commits
  where the skill can't be derived (the `_skill_derive` module doesn't
  exist yet).

## Commits compared

| Short SHA | Date | Subject |
|---|---|---|
| 0ea540efff | 2026-04-22 | Add /v1/completions endpoint (OpenAI legacy completions API) to `transformers serve` (#44558) |
| 59e4754341 | 2026-04-23 | Bugfixes |

Commits are displayed in the order given on the command line; when the user passed `A..B` this is chronological, but arbitrary ordering is also valid.

## Tasks

Each task is a natural-language prompt handed to the agent. All prompts name a specific Hugging Face model so the agent must actually load and run the model (preventing it from answering purely from world knowledge).

| id | category | expected substring | prompt (one-line preview) |
|---|---|---|---|
| `classify-sentiment` | atomic | `positive` | Using the model distilbert/distilbert-base-uncased-finetuned-sst-2-english, classify the sentiment of this sentence and… |
| `extract-entities` | atomic | `tim cook` | Using the model dslim/bert-base-NER, extract the named entities from this sentence and report them with their types: "A… |
| `transcribe-audio` | atomic | `—` | Using the model openai/whisper-tiny, transcribe the audio file at ./inputs/sample.wav and report the transcript. |
| `caption-image` | atomic | `cat` | Using the model llava-hf/llava-interleave-qwen-0.5b-hf, caption the image at ./inputs/cat.jpg and report the caption. |
| `tokenize-count` | atomic | `10` | Using the tokenizer from HuggingFaceTB/SmolLM2-360M-Instruct, tokenize the sentence "The quick brown fox jumps over the… |
| `summarize-text` | atomic | `—` | Using the model facebook/bart-large-cnn, summarize this article in one or two sentences:  The James Webb Space Telescop… |
| `compose-transcribe-sentiment` | compositional | `—` | Transcribe the audio file at ./inputs/sample.wav using openai/whisper-tiny, then classify the sentiment of the transcri… |
| `compose-caption-translate` | compositional | `—` | Caption the image at ./inputs/cat.jpg using llava-hf/llava-interleave-qwen-0.5b-hf, then translate the caption to Frenc… |

**Category meaning.** `atomic` = one existing CLI command in the post-effort state covers the task; the expected behaviour shift is Python → CLI. `compositional` = no single CLI command fits; the agent must write Python (ideally modelled on the `cli/agentic/*.py` exemplars rather than `pipeline(...)`).

**`expected substring`.** If set, each run's final output is checked for a case-insensitive substring match; this is the `✓match` signal in cells. Tasks without an `expected` field are not checked for correctness.

## Metrics and cell format

Each cell in the per-task tables uses the format:

```
**approach** · ✓match · !failed/total · ⇢first-success · 📖docs · ⏻abort · median-time · new · repeat · out
```

Fields that are zero or not applicable are omitted. This report is framed
around **ease of use** — whether the agent had an easy time using transformers
(fewer retries, less thrashing, earlier success), not just whether it used
the CLI. The CLI/Python split is retained but each is further split by
whether any tool call errored (``-retry``) or not (``-clean``).

### Approach — what path the agent took

Each run is bucketed by examining its tool-call sequence, then split by
whether any tool call in the run returned ``is_error=true``:

- `CLI-clean=k/n` / `CLI-retry=k/n` — ran the `transformers` CLI via Bash;
  retry variant means ≥1 errored tool call (a traceback, non-zero exit, etc.).
- `Python-clean=k/n` / `Python-retry=k/n` — executed Python (via `python -c`,
  `Write`+`python <file>`, or similar) without invoking the CLI; retry
  variant means ≥1 errored tool call.
- `no-tool=k/n` — answered with zero tool calls (from model knowledge).
- `other=k/n` — used tools that fit no bucket above (e.g. WebFetch).

Cells containing any CLI adoption are bolded.

### ✓match — correctness check

`✓k/m` where `m` is the number of runs with an `expected` substring
defined for the task, and `k` is the number of those runs whose final
output contained that substring (case-insensitive). Omitted when the task
has no `expected` field.

### !failed/total — tool-call errors

`!k/n` where `k` is the number of tool calls across all runs in the cell
that were flagged as errors (via `is_error: true` on the tool_result),
and `n` is the total tool calls in the cell. Omitted when `k=0`. A tool
error is not necessarily fatal — the agent often recovers and still
matches `expected`.

### ⇢first-success — how fast the agent got a useful answer

`⇢k` where `k` is the median (across runs in the cell) tool-call index at
which the agent's tool_result content first contained the task's
`expected` substring. Lower is better — a well-equipped agent finds the
answer earlier in its exploration. Omitted for tasks without an
`expected` field, and for runs where the expected substring never
appeared in a tool_result (e.g. the match came only from the final
narrative answer).

### 📖docs — which docs the agent consulted

`📖` followed by any non-zero combination of:

- ``agentic=k/n`` — runs that explicitly Read or Grepped a
  ``src/transformers/cli/agentic/*.py`` exemplar.
- ``help=k/n`` — runs that invoked ``transformers … --help``.
- ``AGENTS.md=k/n`` / ``CLAUDE.md=k/n`` / ``SKILL.md=k/n`` — runs that
  explicitly Read that file. These are usually loaded into the agent's
  context automatically by variant configuration (cwd scanning for the
  clone variant, plugin loader for the skill variant), so these fields
  typically show zero and are omitted — explicit Reads are a rare
  behaviour and worth flagging when they happen.

### ⏻ — runs aborted early

The harness kills a run that exceeds a tool-call budget (default 50,
configurable via `--max-tool-calls`) or a wall-clock timeout. When any
run in a cell was aborted, the cell shows ``⏻<reason>:<count>``:

- ``budget_tool_calls`` — killed because tool call count exceeded the budget.
  The run's JSONL stops mid-sequence; no final answer exists. A pattern of
  frequent abort-by-budget suggests the agent is thrashing (retries,
  exploration without converging).
- ``timeout`` — killed because wall-clock elapsed exceeded the internal
  limit (15 minutes). Rarely triggered unless the agent is genuinely stuck.

Aborted runs still contribute tool-call, token, and time numbers up to
the point of kill; they do not contribute to `✓match` (the `expected`
check needs a final answer).

### median-time — wall-clock seconds

Median across the runs in the cell, rounded to the nearest second.

### new / repeat / out — token accounting

Claude Code reports four token fields per assistant turn:
`input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`. These are summed across turns and then split
into two input-side aggregates for reporting:

- `new` = `input_tokens + cache_creation_input_tokens`.
  The unique prompt content the run introduced (system prompt, any
  SKILL.md, files the agent `Read`, tool-result content). Grows when the
  agent needs more information per run; does NOT grow with turn count
  after the cache is warm.
- `repeat` = `cache_read_input_tokens`.
  Tokens re-read from the prompt cache on turns 2, 3, ... Grows with
  turn count (more tool calls → more API turns → more re-reads of the
  same cached prefix). A high `repeat` without a proportional `new`
  increase typically indicates "the agent took more turns", not "the
  prompt got bigger".
- `out` = `output_tokens`. Tokens the model generated (text + tool-call
  arguments). Larger `out` typically means the agent wrote more text
  or more detailed tool inputs.

Both `new` and `repeat` are input-side; they do not overlap. Values are
formatted with `k` (1000) and `M` (1_000_000) suffixes above 1000.

### Summary tables

The report carries two kinds of summary:

- **Headline summary** — split into *atomic tasks* and *compositional
  tasks*, because they have very different token/time profiles and
  aggregating them together lets compositional tasks dominate the
  headline numbers. Each ref gets one column.
- **Per-variant summary** — one sub-table per variant (`bare`, `clone`,
  `skill`) with refs as columns. Use these to ask "holding the variant
  constant, what changed between commits?" directly.

Ratios (`cli_runs/total_runs`, `errored_calls/total_calls`,
`matched_total/matched_seen`) preserve sample size so the reader can see
the denominators directly. Medians are reported with IQR
(``23s (IQR 19–28)``) when the cell has ≥4 runs; below that, the
min–max range is shown in place of IQR.

Above each summary table, a ``⚠ coverage:`` line is emitted whenever the
refs being compared have mismatched coverage (e.g. ``skill`` data
missing for one commit). Headline ratios compare uneven samples in that
case, so the reader is told before reading the numbers.

### What the report does NOT include

- No judgment about which numbers are "good" or "bad".
- No causal claims about what drove a difference between commits.
- Missing cells (`—`) reflect runs that were never executed for that
  (commit, variant, task), not failures. Partial coverage is common when
  a suite was interrupted or deliberately scoped with `--tasks` / `--variants`.

## Summary — atomic tasks

⚠ coverage: skill/caption-image missing for 0ea540efff; skill/classify-sentiment missing for 0ea540efff; skill/extract-entities missing for 0ea540efff; skill/summarize-text missing for 0ea540efff; skill/tokenize-count missing for 0ea540efff; skill/transcribe-audio missing for 0ea540efff. Headline ratios compare uneven samples.

| Metric | 0ea540efff | 59e4754341 |
|---|---|---|
| runs | 36 | 54 |
| CLI adoption (clean + retry) | 1/36 (0 clean / 1 retry) | 19/54 (19 clean / 0 retry) |
| Python (clean + retry) | 35/36 (27 clean / 8 retry) | 33/54 (32 clean / 1 retry) |
| no-tool / other | 0 / 0 | 0 / 2 |
| runs where final output matched `expected` | 24/24 | 36/36 |
| errored tool calls (is_error=true) | 14/131 | 2/203 |
| runs with any error | 9/36 | 2/54 |
| runs aborted (tool-call budget) | 0 | 0 |
| runs aborted (wall-clock timeout) | 0 | 0 |
| median wall-time per run | 20s (IQR 18–32) | 21s (IQR 18–29) |
| median `new` tokens per run | 18k (IQR 16k–23k) | 16k (IQR 16k–32k) |
| median `repeat` tokens per run | 72k (IQR 51k–118k) | 93k (IQR 76k–128k) |
| median output tokens per run | 31 (IQR 25–108) | 40 (IQR 25–97) |
| total `new` tokens (all runs) | 874k | 1.3M |
| total `repeat` tokens (all runs) | 5.1M | 7.4M |
| total output tokens (all runs) | 3.9k | 5.7k |

## Summary — compositional tasks

⚠ coverage: skill/compose-caption-translate missing for 0ea540efff; skill/compose-transcribe-sentiment missing for 0ea540efff. Headline ratios compare uneven samples.

| Metric | 0ea540efff | 59e4754341 |
|---|---|---|
| runs | 12 | 18 |
| CLI adoption (clean + retry) | 0/12 (0 clean / 0 retry) | 8/18 (8 clean / 0 retry) |
| Python (clean + retry) | 12/12 (8 clean / 4 retry) | 10/18 (8 clean / 2 retry) |
| no-tool / other | 0 / 0 | 0 / 0 |
| runs where final output matched `expected` | — | — |
| errored tool calls (is_error=true) | 12/130 | 6/194 |
| runs with any error | 4/12 | 2/18 |
| runs aborted (tool-call budget) | 0 | 0 |
| runs aborted (wall-clock timeout) | 0 | 0 |
| median wall-time per run | 44s (IQR 28–107) | 46s (IQR 33–73) |
| median `new` tokens per run | 28k (IQR 21k–41k) | 33k (IQR 21k–67k) |
| median `repeat` tokens per run | 242k (IQR 107k–455k) | 232k (IQR 145k–508k) |
| median output tokens per run | 285 (IQR 81–573) | 236 (IQR 135–314) |
| total `new` tokens (all runs) | 448k | 760k |
| total `repeat` tokens (all runs) | 4.1M | 6.9M |
| total output tokens (all runs) | 4.2k | 4.5k |

## Per-variant summary

Holding the variant constant, how did behaviour change between commits? Each sub-table aggregates across all tasks for one variant.

## Variant: bare

| Metric | 0ea540efff | 59e4754341 |
|---|---|---|
| runs | 24 | 24 |
| CLI adoption (clean + retry) | 1/24 (0 clean / 1 retry) | 3/24 (3 clean / 0 retry) |
| Python (clean + retry) | 23/24 (17 clean / 6 retry) | 19/24 (18 clean / 1 retry) |
| no-tool / other | 0 / 0 | 0 / 2 |
| runs where final output matched `expected` | 12/12 | 12/12 |
| errored tool calls (is_error=true) | 18/205 | 6/244 |
| runs with any error | 7/24 | 2/24 |
| runs aborted (tool-call budget) | 0 | 0 |
| runs aborted (wall-clock timeout) | 0 | 0 |
| median wall-time per run | 27s (IQR 19–71) | 25s (IQR 17–75) |
| median `new` tokens per run | 17k (IQR 16k–39k) | 17k (IQR 16k–36k) |
| median `repeat` tokens per run | 121k (IQR 51k–346k) | 86k (IQR 51k–503k) |
| median output tokens per run | 139 (IQR 25–377) | 138 (IQR 25–390) |
| total `new` tokens (all runs) | 770k | 753k |
| total `repeat` tokens (all runs) | 6.9M | 7.9M |
| total output tokens (all runs) | 6.2k | 6.0k |

## Variant: clone

| Metric | 0ea540efff | 59e4754341 |
|---|---|---|
| runs | 24 | 24 |
| CLI adoption (clean + retry) | 0/24 (0 clean / 0 retry) | 0/24 (0 clean / 0 retry) |
| Python (clean + retry) | 24/24 (18 clean / 6 retry) | 24/24 (22 clean / 2 retry) |
| no-tool / other | 0 / 0 | 0 / 0 |
| runs where final output matched `expected` | 12/12 | 12/12 |
| errored tool calls (is_error=true) | 8/56 | 2/99 |
| runs with any error | 6/24 | 2/24 |
| runs aborted (tool-call budget) | 0 | 0 |
| runs aborted (wall-clock timeout) | 0 | 0 |
| median wall-time per run | 23s (IQR 19–33) | 29s (IQR 25–44) |
| median `new` tokens per run | 21k (IQR 20k–28k) | 36k (IQR 23k–45k) |
| median `repeat` tokens per run | 73k (IQR 53k–109k) | 146k (IQR 100k–196k) |
| median output tokens per run | 31 (IQR 22–84) | 89 (IQR 39–229) |
| total `new` tokens (all runs) | 552k | 871k |
| total `repeat` tokens (all runs) | 2.3M | 4.0M |
| total output tokens (all runs) | 1.8k | 3.0k |

## Variant: skill

⚠ coverage: skill/caption-image missing for 0ea540efff; skill/classify-sentiment missing for 0ea540efff; skill/compose-caption-translate missing for 0ea540efff; skill/compose-transcribe-sentiment missing for 0ea540efff; skill/extract-entities missing for 0ea540efff; skill/summarize-text missing for 0ea540efff; skill/tokenize-count missing for 0ea540efff; skill/transcribe-audio missing for 0ea540efff. Headline ratios compare uneven samples.

| Metric | 0ea540efff | 59e4754341 |
|---|---|---|
| runs | 0 | 24 |
| CLI adoption (clean + retry) | — | 24/24 (24 clean / 0 retry) |
| Python (clean + retry) | — | 0/24 (0 clean / 0 retry) |
| no-tool / other | — | 0 / 0 |
| runs where final output matched `expected` | — | 12/12 |
| errored tool calls (is_error=true) | — | 0/54 |
| runs with any error | — | 0/24 |
| runs aborted (tool-call budget) | 0 | 0 |
| runs aborted (wall-clock timeout) | 0 | 0 |
| median wall-time per run | — | 20s (IQR 18–29) |
| median `new` tokens per run | — | 16k (IQR 14k–18k) |
| median `repeat` tokens per run | — | 93k (IQR 76k–122k) |
| median output tokens per run | — | 32 (IQR 25–41) |
| total `new` tokens (all runs) | — | 430k |
| total `repeat` tokens (all runs) | — | 2.4M |
| total output tokens (all runs) | — | 1.2k |

## Per-task results

### caption-image

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-retry=2/3 Python-clean=1/3 · ✓3/3 · !4/13 · ⇢1 · 32s · new:18k · repeat:122k · out:139 | Python-clean=3/3 · ✓3/3 · ⇢2 · 52s · new:31k · repeat:276k · out:409 |
| clone | Python-clean=2/3 Python-retry=1/3 · ✓3/3 · !1/8 · ⇢1 · 37s · new:24k · repeat:122k · out:87 | Python-clean=3/3 · ✓3/3 · ⇢2 · 📖agentic=3/3 · 29s · new:36k · repeat:125k · out:57 |
| skill | — | **CLI-clean=3/3** · ✓3/3 · 25s · new:16k · repeat:93k · out:31 |

### classify-sentiment

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-clean=3/3 · ✓3/3 · ⇢3 · 20s · new:16k · repeat:86k · out:140 | Python-clean=3/3 · ✓3/3 · ⇢1 · 17s · new:16k · repeat:51k · out:25 |
| clone | Python-clean=3/3 · ✓3/3 · ⇢1 · 14s · new:20k · repeat:53k · out:25 | Python-clean=3/3 · ✓3/3 · ⇢5 · 📖agentic=3/3 · 28s · new:36k · repeat:154k · out:60 |
| skill | — | **CLI-clean=3/3** · ✓3/3 · ⇢2 · 18s · new:16k · repeat:93k · out:19 |

### compose-caption-translate

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-retry=2/3 Python-clean=1/3 · !8/77 · 137s · new:49k · repeat:682k · out:842 | **Python-retry=1/3 CLI-clean=2/3** · !5/80 · 📖agentic=1/3 help=1/3 CLAUDE.md=1/3 · 111s · new:75k · repeat:764k · out:535 |
| clone | Python-clean=1/3 Python-retry=2/3 · !4/19 · 73s · new:27k · repeat:276k · out:401 | Python-retry=1/3 Python-clean=2/3 · !1/21 · 📖agentic=3/3 · 52s · new:48k · repeat:261k · out:285 |
| skill | — | **CLI-clean=3/3** · 35s · new:21k · repeat:145k · out:172 |

### compose-transcribe-sentiment

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-clean=3/3 · 40s · new:21k · repeat:214k · out:268 | Python-clean=3/3 · 51s · new:27k · repeat:374k · out:229 |
| clone | Python-clean=3/3 · 26s · new:28k · repeat:102k · out:35 | Python-clean=3/3 · 📖agentic=3/3 · 46s · new:57k · repeat:262k · out:242 |
| skill | — | **CLI-clean=3/3** · 32s · new:18k · repeat:145k · out:100 |

### extract-entities

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-clean=3/3 · ✓3/3 · ⇢1 · 20s · new:16k · repeat:51k · out:25 | Python-clean=3/3 · ✓3/3 · ⇢1 · 19s · new:16k · repeat:51k · out:25 |
| clone | Python-clean=3/3 · ✓3/3 · ⇢1 · 21s · new:563 · repeat:73k · out:1 | Python-clean=3/3 · ✓3/3 · ⇢1 · 📖agentic=3/3 · 29s · new:15k · repeat:137k · out:43 |
| skill | — | **CLI-clean=3/3** · ✓3/3 · ⇢2 · 18s · new:16k · repeat:93k · out:41 |

### summarize-text

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | **CLI-retry=1/3 Python-retry=2/3** · !6/69 · 📖agentic=1/3 · 112s · new:66k · repeat:908k · out:496 | **other=2/3 CLI-clean=1/3** · !1/50 · 📖SKILL.md=1/3 · 66s · new:50k · repeat:388k · out:435 |
| clone | Python-retry=3/3 · !3/6 · 31s · new:23k · repeat:109k · out:28 | Python-clean=3/3 · 📖agentic=3/3 · 31s · new:44k · repeat:170k · out:40 |
| skill | — | **CLI-clean=3/3** · 20s · new:17k · repeat:76k · out:17 |

### tokenize-count

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-clean=3/3 · ✓3/3 · ⇢1 · 16s · new:16k · repeat:51k · out:25 | Python-clean=3/3 · ✓3/3 · ⇢1 · 16s · new:16k · repeat:51k · out:25 |
| clone | Python-clean=3/3 · ✓3/3 · ⇢1 · 15s · new:14k · repeat:41k · out:17 | Python-clean=3/3 · ✓3/3 · ⇢3 · 19s · new:23k · repeat:85k · out:87 |
| skill | — | **CLI-clean=3/3** · ✓3/3 · ⇢2 · 18s · new:14k · repeat:76k · out:25 |

### transcribe-audio

| Variant | 0ea540efff | 59e4754341 |
|---|---|---|
| bare | Python-clean=3/3 · 19s · new:17k · repeat:69k · out:69 | Python-clean=3/3 · 18s · new:17k · repeat:69k · out:87 |
| clone | Python-clean=3/3 · 19s · new:21k · repeat:72k · out:34 | Python-retry=1/3 Python-clean=2/3 · !1/10 · 📖agentic=3/3 · 26s · new:29k · repeat:106k · out:27 |
| skill | — | **CLI-clean=3/3** · 20s · new:15k · repeat:76k · out:33 |

