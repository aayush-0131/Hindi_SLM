#!/usr/bin/env python3
"""Interleave final parquet shard filenames so sources mix in the training stream.

Round 1's shards sorted as all 100 `fw2_*` then all 26 `sg_*`, which cost a
+42% train-loss spike once per epoch (ROUND2_HANDOFF.md sec 9.1). This renames
them so sources alternate. Bytes are untouched -- no re-upload, no
re-tokenization -- and the val shard is left name-identical so val bpb stays
comparable across the Round 1 / Round 2 window boundary.

Idempotent: safe to re-run, and re-run it after adding shards with
`add_fineweb2.py`.

    # look first, change nothing
    python3 run_interleave.py --corpus-dir /path/to/final/stable --dry-run

    # do it
    python3 run_interleave.py --corpus-dir /path/to/final/stable

Run this BEFORE uploading the corpus -- renaming locally costs nothing, whereas
renaming after transfer means another pass over the server.
"""

import argparse
import os
import sys

from data_pipeline.shard.interleave_shards import DEFAULT_VAL_GLOB, apply


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="directory of final parquet shards (e.g. final/stable)")
    ap.add_argument("--val-glob", default=DEFAULT_VAL_GLOB,
                    help=f"glob for the pinned validation shard(s), left "
                         f"untouched (default: {DEFAULT_VAL_GLOB})")
    ap.add_argument("--manifest", default=None,
                    help="where to write the old->new mapping (default: "
                         "interleave_manifest.json inside --corpus-dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the mixing report; touch nothing")
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir):
        print(f"no such directory: {args.corpus_dir}", file=sys.stderr)
        return 1

    manifest = args.manifest
    if manifest is None and not args.dry_run:
        manifest = os.path.join(args.corpus_dir, "interleave_manifest.json")

    try:
        apply(args.corpus_dir, val_glob=args.val_glob,
              manifest_path=manifest, dry_run=args.dry_run)
    except (AssertionError, ValueError) as exc:
        print(f"\nREFUSING TO PROCEED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
