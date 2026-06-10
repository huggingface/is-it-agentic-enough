#!/usr/bin/env python
"""One-time migration: repack the OLD per-run file layout into the new bundled
format (one JSONL per model per revision).

    OLD:  results/<rev>/<harness>/<model>/<tier>__<task>__runN.jsonl  (+ .meta.json)
          traces/<rev>/<harness>/<model>/<tier>__<task>__runN.jsonl
    NEW:  results/<rev>/<harness>/<model>.jsonl   # one line per run (meta+events)
          traces/<rev>/<harness>/<model>.jsonl    # one line per run (native session)

Two modes:

* **Local** (default): ``python scripts/migrate_bucket.py <dir>`` repacks the
  ``results/`` and ``traces/`` under ``<dir>`` in place.

* **Remote, end-to-end** (``--bucket <id>``): pull the bucket into a scratch
  dir, repack it, and push the bundles back with ``--delete`` (so the old loose
  files are removed) — leaving the bucket clean in one command::

      python scripts/migrate_bucket.py --bucket lysandre/transformers-agentic-use          # dry-run
      python scripts/migrate_bucket.py --bucket lysandre/transformers-agentic-use --apply  # do it

Dry-run by default in both modes. Requires the ``hf`` CLI (+ ``hf auth login``)
for the remote mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _runs_in(model_dir: Path):
    """Yield ``(tier, task, run, jsonl_path)`` for each old per-run file."""
    for jsonl in sorted(model_dir.glob("*__*__run*.jsonl")):
        if jsonl.name.endswith(".meta.json"):
            continue
        try:
            tier, task, runtok = jsonl.stem.split("__")
        except ValueError:
            continue
        yield tier, task, int(runtok.replace("run", "") or 0), jsonl


def _read_events(jsonl: Path) -> list:
    events = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return events


def _repack_tree(tree: Path, *, is_results: bool, apply: bool) -> tuple[int, int]:
    """Repack every ``<rev>/<harness>/<model>/`` dir into a sibling ``<model>.jsonl``.
    Returns ``(cells, runs)``."""
    cells = runs = 0
    # old per-run files are 4 levels deep: <rev>/<harness>/<model>/<file>.jsonl
    model_dirs = sorted({p.parent for p in tree.glob("*/*/*/*.jsonl")})
    for model_dir in model_dirs:
        rows = []
        for tier, task, run, jsonl in _runs_in(model_dir):
            if is_results:
                meta_path = jsonl.with_suffix(".meta.json")
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                rows.append({"tier": tier, "task": task, "run": run,
                             "meta": meta, "events": _read_events(jsonl)})
            else:
                rows.append({"tier": tier, "task": task, "run": run, "raw": jsonl.read_text()})
        if not rows:
            continue
        rows.sort(key=lambda r: (r["tier"], r["task"], r["run"]))
        # NB: not `.with_suffix` — model ids contain dots (e.g. "Qwen2.5-7B"),
        # which `with_suffix` would mistake for an extension and truncate.
        bundle = model_dir.parent / (model_dir.name + ".jsonl")
        cells += 1
        runs += len(rows)
        rel = bundle.relative_to(tree.parent)
        print(f"  {rel}  ({len(rows)} runs)")
        if apply:
            bundle.write_text("".join(json.dumps(r) + "\n" for r in rows))
            shutil.rmtree(model_dir)  # drop the old per-run files
    return cells, runs


def _repack_root(root: Path, *, apply: bool) -> int:
    """Repack ``results/`` + ``traces/`` under ``root``. Returns total cells."""
    total_cells = total_runs = 0
    for name, is_results in (("results", True), ("traces", False)):
        tree = root / name
        if not tree.exists():
            continue
        print(f"{name}/:")
        c, r = _repack_tree(tree, is_results=is_results, apply=apply)
        total_cells += c
        total_runs += r
    if total_cells:
        print(f"\n{total_cells} cell file(s), {total_runs} run(s).")
    return total_cells


def _run(cmd: list[str]) -> None:
    print("  ▶ " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"command failed (exit {rc}): {' '.join(cmd)}")


def _uri(bucket: str, prefix: str) -> str:
    base = bucket[len("hf://buckets/"):] if bucket.startswith("hf://buckets/") else bucket
    return f"hf://buckets/{base}/{prefix}"


def _migrate_remote(bucket: str, work: Path, apply: bool) -> int:
    if shutil.which("hf") is None:
        raise SystemExit("the `hf` CLI is required for --bucket mode "
                         "(install: `curl -LsSf https://hf.co/cli/install.sh | bash`; then `hf auth login`).")
    trees = ["results", "traces"]
    print(f"== migrate bucket {bucket} (scratch: {work}) ==\n")

    print("1) pull current bucket contents:")
    for p in trees:
        _run(["hf", "buckets", "sync", _uri(bucket, p), str(work / p)])

    print("\n2) repack to the bundled format:")
    cells = _repack_root(work, apply=apply)
    if cells == 0:
        print("nothing to migrate (bucket already bundled?). Done.")
        return 0

    print("\n3) push bundles back (--delete drops the old loose files):")
    push = []
    for p in trees:
        push.append(["hf", "buckets", "sync", str(work / p), _uri(bucket, p), "--delete"])
    if not apply:
        for cmd in push:
            print("  " + " ".join(cmd))
        print("\nDRY RUN — re-run with --apply to repack and push. Nothing was changed on the bucket.")
        return 0
    for cmd in push:
        _run(cmd)
    print(f"\n✓ bucket {bucket} migrated to the bundled format.")
    print("  Refresh your local copy with:  ag report transformers --pull")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=".", help="local mode: dir holding results/ and traces/ (default: .)")
    ap.add_argument("--bucket", help="remote mode: bucket id to migrate end-to-end (pull → repack → push --delete)")
    ap.add_argument("--work", default="_bucketmig", help="scratch dir for --bucket mode (default: _bucketmig)")
    ap.add_argument("--apply", action="store_true", help="actually write/push (default: dry-run)")
    args = ap.parse_args()

    if args.bucket:
        return _migrate_remote(args.bucket, Path(args.work), args.apply)

    if _repack_root(Path(args.dir), apply=args.apply) == 0:
        print("nothing to migrate (no old per-run files found — already bundled?).")
        return 0
    if not args.apply:
        print("DRY RUN — re-run with --apply to write the bundles and remove the old files.")
    else:
        print("✓ repacked. For a bucket, push with `hf buckets sync <dir>/<tree> hf://buckets/<id>/<tree> --delete`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
