"""Render a Claude Code SKILL.md from a transformers install's derived manifest.

Exits (returns) with code 2 if the commit predates the skill-derivation
effort — the caller treats that as "no skill available for this commit".
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


SKILL_TEMPLATE = """---
name: transformers
description: {description}
---

# Transformers CLI

For one-off inference, training, quantization, or export, invoke the
`transformers` command directly rather than writing Python. Run
`transformers --help` for the full command list; run
`transformers <command> --help` for flags per command.

## Invocation rules

**All inputs are named flags, never positional.** Wrong invocations like
``transformers classify "my text"`` or ``transformers ner "sentence"`` will
fail with ``Got unexpected extra argument``. The text / image / audio / file
argument is always a flag: ``--text``, ``--image``, ``--audio``, ``--file``.

**Always invoke as `transformers <cmd> ...`.** Do not use
``python -m transformers ...`` patterns — the console script is what the
``transformers`` package installs.

**Use `transformers --format json` for machine-readable output**:
``transformers --format json classify --text "..."``.

## Example invocations (copy these shapes)

Text (classify, ner, token-classify, summarize, translate, fill-mask):
```
transformers classify --text "I loved this movie"
transformers classify --text "..." --model distilbert/distilbert-base-uncased-finetuned-sst-2-english
transformers ner --text "Apple CEO Tim Cook visited Paris." --model dslim/bert-base-NER
transformers summarize --file article.txt --model facebook/bart-large-cnn
transformers translate --text "The weather is nice" --model Helsinki-NLP/opus-mt-en-de
```

Question answering (takes `--question` and `--context`):
```
transformers qa --question "Who invented it?" --context "Graham Bell invented the telephone in 1876."
```

Image (caption, image-classify, detect, segment, depth, vqa, ocr):
```
transformers caption --image photo.jpg --model llava-hf/llava-interleave-qwen-0.5b-hf
transformers image-classify --image photo.jpg
transformers vqa --image photo.jpg --question "What color is the car?"
```

Audio (transcribe, audio-classify, speak):
```
transformers transcribe --audio clip.wav --model openai/whisper-tiny
transformers audio-classify --audio clip.wav
transformers speak --text "Hello" --output hello.wav
```

Tokenize / inspect / embed:
```
transformers tokenize --model HuggingFaceTB/SmolLM2-360M-Instruct --text "tokenize me"
transformers inspect meta-llama/Llama-3.2-1B-Instruct
transformers embed --text "some sentence" --model BAAI/bge-small-en-v1.5
```

Generate (text completion):
```
transformers generate --prompt "Once upon a time" --model HuggingFaceTB/SmolLM2-360M-Instruct
```

## Available commands

{command_list}

## When to use what

- **Atomic task** (single inference / training / export): use the CLI.
- **Composed workflow** (chain models, custom logic): write Python.
  The CLI commands' source in `transformers.cli.agentic.*` is the
  canonical template — each file loads a model with `AutoModel*` +
  `AutoProcessor`/`AutoTokenizer`, runs a forward pass, and
  post-processes. Copy that pattern rather than reaching for
  `pipeline(...)`.
"""


def render_skill(manifest: dict) -> str:
    lines = []
    for cap in manifest["capabilities"]:
        desc = cap.get("description", "").rstrip(".")
        lines.append(f"- `transformers {cap['id']}` — {desc}")
    return SKILL_TEMPLATE.format(
        description=manifest.get(
            "description",
            "Run Hugging Face Transformers inference, training, quantization, and export via the CLI.",
        ),
        command_list="\n".join(lines),
    )


def derive_manifest(venv_python: Path) -> dict | None:
    """Ask a venv's transformers install for its skill manifest. Returns None if unavailable."""
    try:
        raw = subprocess.check_output(
            [
                str(venv_python),
                "-c",
                "import json, sys; "
                "from transformers.cli.agentic._skill_derive import derive_skill_from_cli; "
                "json.dump(derive_skill_from_cli(), sys.stdout)",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw)


def build(venv_python: Path, plugin_dir: Path) -> bool:
    """Write SKILL.md under ``plugin_dir/skills/transformers/``. Returns True on success."""
    manifest = derive_manifest(venv_python)
    if manifest is None:
        return False
    skill_dir = plugin_dir / "skills" / "transformers"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(render_skill(manifest))
    return True
