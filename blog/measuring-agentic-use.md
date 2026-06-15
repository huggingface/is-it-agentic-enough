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

### Which agents should you look at?

Not all agents are equal, and their difference changes what you should look at when running them.

*Frontier*

At one end, you have the big, frontier, often closed models. On reasonably common tasks,
these should get the right answer, eventually. For them, __"match %"__ saturates near
100% and stops telling you much about your tool; a more interesting benchmark is the effort
it took the agent to get there: how many turns, tokens and seconds it took, and whether they walked a clean
path or used deprecated APIs.

*Open*

At the other end are the open models. Their size varies greatly, and so do their abilities.
Metrics such as __"match %"__ are way more interesting than for their closed counterparts,
as you can see how model sizes/capabilities affect the different models on your specific
tool.

On top of being insightful for you as a library maintainer, giving you insights on how to update
your repository for agents to interact better with it, it's also useful for you as a user:
if designing agentic workflows or choosing a model for them, you now have a harness to directly
evaluate each of them on a set of tools directly related to the task you want to evaluate.

The harness  scores every run on several axes, so that you can ask the question that's actually
interesting for each class of model:

- **match %** — did the final answer contain the expected result (per-task,
  case-insensitive substring / regex / exact, all explicit in the report);
- **median time** and **median tokens** (new vs. cached vs. generated);
- **runs with error %** — including a guard that flags runs which produced *nothing*
  (0 output tokens, no tool calls, no answer) so silent failures don't masquerade
  as "0";
- **label adoption** — profile-defined behavior markers; see below for an explanation 
  of what this is.


Screenshot of what are the expected results we'd like to see across models and revisions


Because it captures the **native agent trace** of every run, you can also just…
read what happened, and the traces are shareable through the Hub's
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces).

Those two classes call for two different experiments.

### Closed models: hold the model, vary the tool

Since a frontier model will usually get to the correct result, what you're really 
measuring is the effort it took to do so. Did it take ten turns or one? Did it follow 
an API path you deprecated because it trusted obsolete documentation? Did it hit an 
error you hadn't foreseen? 

The natural experiment is to **fix one strong model and vary the tool's
revisions**, watching whether the load it puts on the agent goes up or down. We used
the harness on `transformers` exactly this way, to check whether adding a dedicated
CLI and Skill actually lightened the agents' work:

[screenshot: one model, metrics across tool revisions]

---

### Open models: hold the tool, vary the model

Open models give you a knob the closed APIs don't: granular control over size,
configuration, provider, training, among others. They're also where a good tool surface 
matters most: a small model asked to "use `transformers` to do X" on a `bare` checkout can
guess an API that changed some releases ago, may do unnecessary tool calls, and can
get the wrong answer.

So here the experiment is the opposite of the above: **hold the tool and sweep the model**, to
see *which* models actually cope with your task — not just by token count and time,
but down to which ones can't reliably drive the tool calls at all. My intuition is
that the smaller the model, the harder both tool use and the task get; I ran the
harness across a range of model sizes to test exactly that:

[screenshot: many models of varying size, one tool revision]

> A note on fair comparison: naively averaging across tasks is misleading when
> coverage is uneven (a model that only finished the quick tasks looks fast). The
> report has a **"shared tasks only"** toggle (across models and/or revisions) so
> you compare like-for-like, and a **Coverage** heatmap so you can see exactly which
> task × revision × model cells actually ran.


## Tweaking the tool

### Specific markers

- what are specific markers? -> example of the CLI tool, or pipeline ref
- 

### Analyzing results: a not-so-straightforward understanding

[CLAUDE]

Introduce the entire reason this was made:
- had an intuition for how to simplify transformers usage
    - implement CLI
    - implement skill
- wanted a scientific backing of what I was pushing forward

Analyzing results:
- mention the CLI commit
  - seems to help bigger models (Kimi, etc)
  - is hurting smaller models, intuition: they probably do things from memory

[END CLAUDE]

---

## Trying it yourself

The harness is one CLI, `ag` — install it, run a suite, fan it out across models × revisions on HF Jobs, and publish the report as a Hugging Face Space. The full, kept-current setup and usage instructions live in the [README](../README.md).

## Closing

"Can the model answer?" is table stakes. The question that decides cost and
reliability in production is **"how does the model get the answer, and does our
library make that path short and safe?"** This harness makes that measurable — and
turns "we added a CLI" from a vibe into a number you can put on a graph.

If you maintain a library that agents touch, the takeaway is simple: **the agent
surface is part of your API**, and it's worth benchmarking like one.

*The harness, the task suite, and all the traces behind these numbers are open —
`[[link to repo / Space / dataset]]`.*
