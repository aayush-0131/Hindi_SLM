#!/usr/bin/env python3
"""Check the training corpus for eval-set contamination (handoff sec 9.6).

Never done. Without it, any "we beat Goldfish-Hindi" claim is attackable --
Hindi benchmarks are largely translations of English sets that have been on the
public web for years, and three of our seven sources are web crawls.

    # quick estimate: first 12 shards
    python3 run_decontaminate.py --corpus-dir <final/stable> --limit-shards 12

    # full pass, parallel over shards
    python3 run_decontaminate.py --corpus-dir <final/stable> \\
        --workers 32 --json-out decontamination.json

The number that matters is CONTAMINATED EVAL ITEMS, not contaminated corpus
documents. Feed `contaminated_items` from the JSON into an exclusion list when
scoring, and report the excluded count alongside the score.

Exit 1 if any task exceeds --max-item-contamination, so this can gate the
comparison rather than just informing it.
"""

import argparse
import json
import os
import sys

from eval_harness.decontaminate import (
    DEFAULT_N, EVAL_SOURCES, build_eval_index, scan_corpus,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("-n", "--ngram", type=int, default=DEFAULT_N,
                    help=f"n-gram size in words (default {DEFAULT_N}, the "
                         f"GPT-3 convention)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="scan only the first N shards (quick estimate)")
    ap.add_argument("--max-item-contamination", type=float, default=None,
                    help="exit 1 if any task's contaminated-item fraction "
                         "exceeds this (e.g. 0.02)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    corpus_glob = os.path.join(args.corpus_dir, "*.parquet")
    if not os.path.isdir(args.corpus_dir):
        print(f"no such directory: {args.corpus_dir}", file=sys.stderr)
        return 1

    print(f"building {args.ngram}-gram index from the eval sets")
    index, meta, failures = build_eval_index(n=args.ngram)
    if failures:
        print(f"\n{len(failures)} eval source(s) FAILED TO LOAD -- these are "
              f"NOT covered by this report:", file=sys.stderr)
        for label, repo, err in failures:
            print(f"  {label} ({repo}): {err}", file=sys.stderr)
        print("  Repo ids in eval_harness/decontaminate.py:EVAL_SOURCES are "
              "unverified (see its docstring). Fix them before trusting a "
              "clean result -- a task missing from this report reads as "
              "'clean' when it is really 'unchecked'.", file=sys.stderr)
    if not index:
        print("no eval n-grams built; nothing to check.", file=sys.stderr)
        return 1
    print(f"  {len(index):,} distinct n-grams across "
          f"{len(meta)} task(s)\n")

    print(f"scanning {args.corpus_dir}"
          f"{f' (first {args.limit_shards} shards)' if args.limit_shards else ''}")
    total_docs, hit_docs, hit_items = scan_corpus(
        corpus_glob, index, n=args.ngram, workers=args.workers,
        limit_shards=args.limit_shards)

    print(f"\n{'=' * 68}\nCONTAMINATED EVAL ITEMS  (the number that matters)\n"
          f"{'=' * 68}")
    print(f"  {'task':16s} {'items':>8s} {'hit':>6s} {'rate':>8s} {'short':>7s}")
    worst = 0.0
    summary = {}
    for label, info in meta.items():
        hits = hit_items.get(label, [])
        rate = len(hits) / info["items"] if info["items"] else 0.0
        worst = max(worst, rate)
        summary[label] = {
            "repo": info["repo"], "items": info["items"],
            "contaminated": len(hits), "rate": rate,
            "items_too_short_for_ngram": info["items_too_short_for_ngram"],
            "contaminated_items": hits,
        }
        print(f"  {label:16s} {info['items']:8,} {len(hits):6,} {rate:7.2%} "
              f"{info['items_too_short_for_ngram']:7,}")

    print(f"\n{'=' * 68}\nCORPUS DOCUMENTS CONTAINING EVAL TEXT  (context only)\n"
          f"{'=' * 68}")
    print(f"  scanned {total_docs:,} documents")
    if hit_docs:
        for source, count in hit_docs.most_common():
            print(f"    {source:28s} {count:8,} doc(s)")
    else:
        print("    none")

    print(f"\nInterpretation: contaminated corpus documents are largely "
          f"harmless\n(a few in millions changes no weights). Contaminated "
          f"EVAL ITEMS are not --\nexclude them when scoring and report the "
          f"excluded count.")
    if args.limit_shards:
        print(f"\nNOTE: partial scan ({args.limit_shards} shards). Item "
              f"contamination can only RISE with a full scan, never fall.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"ngram": args.ngram, "docs_scanned": total_docs,
                       "partial_scan_shards": args.limit_shards,
                       "load_failures": failures,
                       "doc_hits_by_source": dict(hit_docs),
                       "tasks": summary}, fh, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")

    if args.max_item_contamination is not None and worst > args.max_item_contamination:
        print(f"\nFAIL: worst item contamination {worst:.2%} exceeds "
              f"{args.max_item_contamination:.2%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
