#!/usr/bin/env python
"""Delete bucket files that were pushed before a cutoff date (default: today, UTC).

Each file in an HF bucket carries an ``uploaded_at`` timestamp — when that
version was pushed. This prunes everything pushed strictly *before* the cutoff,
so a fresh re-push from today survives while stale earlier pushes are removed.

Dry-run by default — it only lists what it *would* delete. Pass ``--apply`` to
actually delete. Requires ``HF_TOKEN`` (or ``hf auth login``) with write access.

Usage:
    python scripts/clean_bucket.py                       # dry-run, cutoff = today UTC
    python scripts/clean_bucket.py --apply               # delete pre-today pushes
    python scripts/clean_bucket.py --before 2026-06-01   # custom cutoff (UTC date)
    python scripts/clean_bucket.py --prefix results/     # restrict to a subtree
    python scripts/clean_bucket.py --use-mtime           # key off mtime, not uploaded_at
"""

from __future__ import annotations

import argparse
import datetime as dt

DEFAULT_BUCKET = "lysandre/transformers-agentic-use"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"bucket id (default: {DEFAULT_BUCKET})")
    ap.add_argument("--before", metavar="YYYY-MM-DD",
                    help="delete files pushed strictly before this UTC date (default: today)")
    ap.add_argument("--prefix", default=None, help="only consider files under this path prefix")
    ap.add_argument("--use-mtime", action="store_true",
                    help="compare against `mtime` instead of `uploaded_at`")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--chunk", type=int, default=200, help="files per delete batch (default: 200)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    if args.before:
        cutoff = dt.datetime.fromisoformat(args.before).replace(tzinfo=dt.timezone.utc)
    else:
        today = dt.datetime.now(dt.timezone.utc).date()
        cutoff = dt.datetime(today.year, today.month, today.day, tzinfo=dt.timezone.utc)
    field = "mtime" if args.use_mtime else "uploaded_at"

    api = HfApi()
    stale, kept = [], 0
    for it in api.list_bucket_tree(args.bucket, prefix=args.prefix, recursive=True):
        if getattr(it, "type", None) != "file":
            continue
        ts = getattr(it, field, None) or getattr(it, "mtime", None)
        if ts is not None and ts < cutoff:
            stale.append((it.path, ts))
        else:
            kept += 1

    stale.sort()
    print(f"bucket {args.bucket}: {len(stale)} file(s) pushed before "
          f"{cutoff.date()} (by {field}), {kept} kept")
    for path, ts in stale:
        print(f"  - {path}  [{ts:%Y-%m-%d %H:%M}]")

    if not stale:
        print("nothing to delete.")
        return 0
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to delete the above.")
        return 0

    paths = [p for p, _ in stale]
    for i in range(0, len(paths), args.chunk):
        batch = paths[i:i + args.chunk]
        api.batch_bucket_files(args.bucket, delete=batch)
        print(f"deleted {i + len(batch)}/{len(paths)}")
    print(f"✓ deleted {len(paths)} file(s) from {args.bucket}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
