#!/usr/bin/env python
"""Remove a model's runs from the bucket, across every revision.

Bucket files live at ``{results,traces}/<revision>/<harness>/<model_id>/…``; a
"model entry" is one ``<harness>/<model_id>`` namespace (e.g.
``pi/Qwen--Qwen2.5-7B-Instruct:together``). This deletes every file under that
namespace in both ``results/`` and ``traces/``, leaving other models and the
per-revision ``ref.json`` markers untouched.

Dry-run by default — it only lists what it *would* delete. Pass ``--apply`` to
actually delete. Requires ``HF_TOKEN`` (or ``hf auth login``) with write access.

Usage:
    python scripts/remove_model.py --list                       # show every model entry + file count
    python scripts/remove_model.py pi/Qwen--Qwen3-Coder-Next     # dry-run (namespace as shown by --list)
    python scripts/remove_model.py Qwen/Qwen3-Coder-Next         # also accepts the original model id
    python scripts/remove_model.py pi/Qwen--Qwen3-Coder-Next --apply
"""

from __future__ import annotations

import argparse
from collections import Counter

DEFAULT_BUCKET = "lysandre/transformers-agentic-use"


def _ns_parts(path: str):
    """``(namespace, model_id)`` for a model-scoped file, else ``(None, None)``.

    Bundled layout: ``{results,traces}/<revision>/<harness>/<model_id>.jsonl``
    (4 segments, model is the file). Also handles the legacy per-run layout
    (``…/<model_id>/<file>``, 5+ segments). ``ref.json`` markers are skipped."""
    parts = path.split("/")
    if not parts or parts[0] not in ("results", "traces"):
        return None, None
    if len(parts) == 4 and parts[3].endswith(".jsonl"):
        harness, model_id = parts[2], parts[3][: -len(".jsonl")]
        return f"{harness}/{model_id}", model_id
    if len(parts) >= 5:  # legacy per-run files
        harness, model_id = parts[2], parts[3]
        return f"{harness}/{model_id}", model_id
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="model entry to remove: '<harness>/<model_id>' or a model id")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"bucket id (default: {DEFAULT_BUCKET})")
    ap.add_argument("--list", action="store_true", help="list every model entry with its file count, then exit")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--chunk", type=int, default=200, help="files per delete batch (default: 200)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    files = [it for it in api.list_bucket_tree(args.bucket, recursive=True)
             if getattr(it, "type", None) == "file"]

    if args.list or not args.model:
        counts: Counter[str] = Counter()
        for it in files:
            ns, _ = _ns_parts(it.path)
            if ns:
                counts[ns] += 1
        print(f"bucket {args.bucket}: {len(counts)} model entr{'y' if len(counts) == 1 else 'ies'}")
        for ns, n in sorted(counts.items()):
            print(f"  {n:>5}  {ns}")
        if not args.model:
            print("\nPass one of the above to remove it (dry-run); add --apply to delete.")
        return 0

    target = args.model
    sanitized = target.replace("/", "--")  # accept the original "org/name" model id
    stale = []
    for it in files:
        ns, model_id = _ns_parts(it.path)
        if ns and (target == ns or target == model_id or sanitized == model_id):
            stale.append(it.path)

    if not stale:
        print(f"no files matched `{target}` in {args.bucket}. Run with --list to see the entries.")
        return 1

    stale.sort()
    print(f"{len(stale)} file(s) under `{target}`:")
    for p in stale:
        print(f"  - {p}")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to delete the above.")
        return 0

    for i in range(0, len(stale), args.chunk):
        batch = stale[i:i + args.chunk]
        api.batch_bucket_files(args.bucket, delete=batch)
        print(f"deleted {i + len(batch)}/{len(stale)}")
    print(f"✓ removed `{target}` ({len(stale)} files) from {args.bucket}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
