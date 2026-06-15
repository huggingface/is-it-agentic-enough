# Is it agentic enough? Engineering an agent-dedicated CI for transformers

> This is a human-made, agent-focused blogpost.

While software has always aimed to optimize its execution in terms of computational cost and time, we're now rarely
interacting with the software directly, instead passing through an additional agentic layer doing most of the guessing
and algorithmic scaffolding without us looking at it. 

More than that, depending on the task, the agentic layer can now completely bypass the software we once used in 
favor of rewriting it from scratch. It is not sufficient anymore for the software to be performant and exact: it
needs to be agent-optimized if we want it to be leveraged by agents.

Most benchmarks focus on "can the agent get the right answer?". Here, we'll introduce why we think a tool-specific
harness which instead focuses on "how does the agent get to the answer?" is particularly relevant; and provide a
simple implementation of one such harness for the open-source ecosystem.

## How do you optimize software for agents?

I'm a strong believer in the following two software principles:
- If it isn't tested, then it doesn't work
- If it isn't documented, then it doesn't exist

This remains the same within the realm of agentic-optimized tooling, and, for once, the two are directly tied to
each other.

If you want your tool to exist for an agent, then it needs to be discoverable. The API needs to be clear and the docs 
need to be extensive. They need to be structured in a way that the agent has rapid access to the useful files and
examples. The more bundled, monolithic they are, the more tokens you'll burn through trying to get the information.

If you want your tool to work for an agent, then you need to test it for agentic-use. This is going to be the focus 
of this blogpost.

## Testing software for agentic-use

<maybe a quick mention of why we're using transformers here?>

Two agents can both produce the correct label for a sentiment-classification task,
but one:

- writes a 40-line Python script, imports `transformers`, debugs a shape error,
  re-runs twice, and finally prints the answer; 

while the other

- types `transformers classify --model … --text "…"` and is done in one call.

According to the model used, it's likely that the two amount to the same result. Still,
you'll see different **cost, latency, token usage, and failures**.

If your evaluation only checks the final string, you're blind to these as well as 
whether a change you shipped to the library (a CLI improvement, better error messages, a
Skill) actually helped agents.

This is what we're trying to do with this harness: we want to evaluate how much work
an agent had to do a given task, and what changes can be done to the library to help
them out.

### Are all agents equal?

[CLAUDE please develop this according to the notes below]

- Start defining that there are two types of agents:
    - the big ones, which will get the good result no matter what
    - the small ones, which might not get to the result even with a lot of time

Concretely, the harness scores each run on:

- **match %** — did the final answer contain the expected result (per-task,
  case-insensitive substring / regex / exact, all explicit in the report);
- **median time** and **median tokens** (new vs. cached vs. generated);
- **runs with error %** — including a guard that flags runs which produced *nothing*
  (0 output tokens, no tool calls, no answer) so silent failures don't masquerade
  as "0";
- **label adoption** — profile-defined behavior markers like `cli` ("ran the
  `transformers` command-line tool instead of writing Python"), `pipeline`,
  `ran-help`, etc., each with a one-line description shown in the UI.


Screenshot of what are the expected results we'd like to see across models and revisions


Because it captures the **native agent trace** of every run, you can also just…
read what happened, and the traces are shareable through the Hub's
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces).

[/END CLAUDE]

### What this changes for closed (API) models

At the time of writing of this blogpost, closed models are increasingly good at coding tasks, especially
well documented ones. The metric we're interested here is therefore not the match %: the model will get it
right, very likely.

But how long did it take? Did it follow an API path we deprecated, because it followed obsolete documentation?
Did it run into an error we had not foreseen?

For a hosted model you pay per token and per second, and or any unnecessary work the agent does:
writing a script, hitting an error, reading a stack trace; every extra turn sends the growing 
context to the model to continue the experiment.

For closed models it becomes interesting to select a single model as the axis, and to check different revisions
of the tools to see the eventual increase/reduction in model load. We used this tool on transformers to estimate
whether the cost of adding a specific CLI and Skill to the project ended up helping out the agents in their work:

---

### What this changes for open models

Open models are widely different to closed models as you have a very granular level of control
over the model: size, configuration, provider, training, etc.

Open models are where the effect is most striking, because smaller open models are
the ones that struggle most with cold-starting a complex API from memory.

A `[[7B–30B]]` open model asked to "use `transformers` to do X" on a `bare`
checkout often:

- guesses an API that changed three releases ago,
- burns its budget on retries, and
- sometimes never converges — producing an empty or wrong answer.

This tool therefore offers a deeper view in something different: token count and elapsed time are
important yes, but it allows to see deeper exactly *which* models struggle with your task or tool.

My intuition is that the smaller model, the harder it will be for it to handle tool calls and 
tasks in general. I set up the tool to run across an array of model sizes to test just that:



> A note on fair comparison: naively averaging across tasks is misleading when
> coverage is uneven (a model that only finished the quick tasks looks fast). The
> report has a **"shared tasks only"** toggle (across models and/or revisions) so
> you compare like-for-like, and a **Coverage** heatmap so you can see exactly which
> task × revision × model cells actually ran.


## Specific markers?

Giving the agent a first-class CLI collapses that loop. When the `cli` marker fires,
a task that took *N* tool calls and a few thousand tokens of trial-and-error becomes
a single command:

- **Lower cost.** Fewer turns ⇒ fewer generated tokens and far fewer *repeated*
  prompt tokens. In our runs, switching from the `bare` tier to the CLI-aware tier
  cut median new tokens from `[[X]]` to `[[Y]]` on the `[[task]]` task. ([[fill in]])
- **More predictable latency.** One deterministic command instead of an open-ended
  generate-debug-retry loop. Median time dropped from `[[X]]s` to `[[Y]]s`. ([[fill in]])
- **More *accurate* usage of the library.** The CLI encodes the right model-loading,
  device, and dtype defaults, so the agent stops reinventing (and mis-implementing)
  them. Fewer `runs with error %`: `[[X]]% → [[Y]]%`. ([[fill in]])
- **Auditability.** The exact command is in the trace. For teams that need to know
  *what* an agent ran against production data, "it typed `transformers classify …`"
  is a much better answer than "it executed some generated Python."

---

## How to use the app

The harness is one CLI, `ag`. A **profile** defines the environment and the
comparison axis; `transformers` is the reference profile (it builds a git worktree
of `transformers` at each revision and derives the Skill), and there's a `mock`
profile for fast UI testing.

### 1. Install

```bash
git clone https://github.com/huggingface/is-transformers-agentic-enough
cd is-transformers-agentic-enough
uv venv --python 3.13 .env
uv pip install --python .env/bin/python -e .
```

### 2. Run a suite for one revision

```bash
# Claude as the agent, all tiers/tasks, 5 runs each, on the v5.9.0 tag
ag suite transformers v5.9.0 --runner claude --runs 5

# An open model served on HF, only the fast tasks, on a named branch
ag suite transformers 4d15b215f3 --runner pi \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --name "w/ CLI + Skill" --tasks classify-sentiment fill-mask tokenize-count
```

Every run records its transcript, metadata (status / tokens / time), and the native
agent session.

### 3. Compare many models × revisions on HF Jobs

Declare the matrix in a YAML file and launch it as detached HF Jobs:

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
  - v5.10.2
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
```

```bash
# Skip cells already in the bucket, launch the rest, and watch them finish
ag batch eval.yaml --submit --watch --skip-complete
```

Jobs persist each run to a shared Hugging Face **bucket** the moment it completes,
so a crash or eviction never loses the runs that already succeeded, and re-running
the same file resumes where you left off.

### 4. Build and publish the report

```bash
# Pull the latest runs from the bucket and open the report locally
ag report transformers --pull --open

# Publish it as a static HF Space
ag report transformers --pull --push --space your-org/agentic-eval-report
```

The report is a single, self-contained, theme-aware page with three tabs:

- **Overview** — match %, median time, median tokens, error % across your chosen
  X-axis (revision / model / tier) and series, plus the configurable
  label-adoption chart and per-run distributions. Includes the fair-comparison
  (shared-tasks) toggle.
- **Coverage** — a task × revision heatmap (toggle models) so you can see what ran.
- **Results** — every task (its exact prompt, the input image/audio, the match rule)
  and, per model, what it answered — including a click-through into the *failing*
  responses so you can tell a genuine miss from a too-strict matcher.

---

## Closing

"Can the model answer?" is table stakes. The question that decides cost and
reliability in production is **"how does the model get the answer, and does our
library make that path short and safe?"** This harness makes that measurable — and
turns "we added a CLI" from a vibe into a number you can put on a graph.

If you maintain a library that agents touch, the takeaway is simple: **the agent
surface is part of your API**, and it's worth benchmarking like one.

*The harness, the task suite, and all the traces behind these numbers are open —
`[[link to repo / Space / dataset]]`.*
