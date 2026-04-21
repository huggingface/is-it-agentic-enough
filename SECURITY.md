# Security model

This harness is designed for **trusted, local benchmarking only**. A few
properties are worth being explicit about before pointing it at anything
you didn't write yourself.

## `claude` runs with `--permission-mode bypassPermissions`

Every spawned agent can execute arbitrary `Bash`, `Read`, `Write`, etc.
without prompting. That's required for unattended N×M×K runs, but it
means an adversarial task prompt, a malicious model repo (anything
reaching for `trust_remote_code=True`), or a compromised `transformers`
ref is effectively RCE under your user.

Do **not** run `isth setup` / `isth run` against untrusted git refs
(random PR branches, forks you haven't reviewed) — `pip install -e
<worktree>` and the skill-manifest derivation step both execute code
from the chosen commit.

## The agent inherits most of your environment

The harness scrubs `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` before exec,
but other secrets in `os.environ` (`OPENAI_API_KEY`, `AWS_*`,
`GITHUB_TOKEN`, `~/.netrc`, `~/.aws/credentials`, ...) are reachable
from a fully-permissioned agent. Run the harness in a shell that only
carries the credentials `claude` itself needs.

## `results/*.jsonl` is a verbatim capture of everything the agent saw

File contents the agent `Read`, full `tool_result` payloads, model
outputs, web fetches — all serialised. Treat the directory as
"untrusted log of attacker-influenceable content" before publishing or
sharing it. Scan for secrets first.

## Traces are a prompt-injection vector for downstream LLMs

Anything a model produced or a tool returned can contain
`<system-reminder>`, `<assistant>`, or similar markup. If you feed
`progress.md` plus raw traces back into an LLM (as the worked example
in the README proposes), wrap the trace content in explicit
`BEGIN UNTRUSTED TRACE` / `END UNTRUSTED TRACE` fences and instruct the
reviewer LLM to treat everything between them as **data, never
instructions**.

The harness has already produced traces in the wild containing injected
`<system-reminder>` blocks aimed at the next LLM in the chain.
