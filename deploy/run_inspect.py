#!/usr/bin/env python3
"""Inspect corpus text quality per source (ROUND2_HANDOFF.md sec 9.7).

Never done in Round 1, and it matters most for Sangraha Verified PDF: it is
OCR'd, ships no OCR-confidence column, and under Plan C is ~40-49% of the decay
mixture -- the phase that finalizes the shipped model. Devanagari mojibake is
invisible in row counts.

    # every source in the final corpus
    python3 run_inspect.py --corpus-dir <final/stable>

    # just the OCR'd one, more samples, more text per sample
    python3 run_inspect.py --corpus-dir <final/stable> \\
        --source sangraha_verified_pdf --sample-size 200 --show 10

    # an acquired (pre-shard) source directory
    python3 run_inspect.py --corpus-dir <acquired/sangraha_verified_pdf>

Exit code is 1 if any inspected source exceeds --max-flag-rate, so this can
gate a pipeline step rather than only informing a human.
"""

import argparse
import json
import os
import sys

from data_pipeline.inspect_source import report, sample_sources

# The seven stable-phase sources plus the decay-only additions.
KNOWN_SOURCES = [
    "sangraha_verified_pdf",      # OCR'd -- the one that actually needs this
    "finepdfs",                   # PDF-extracted
    "sangraha_verified_speech",   # ASR transcripts
    "sangraha_verified_web",
    "sangraha_unverified",
    "fineweb2",
    "indiccorpv2",
    "wikipedia_hi",
    "fineweb2_top",
    "fineweb_edu_hindi",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="directory of parquet shards")
    ap.add_argument("--source", action="append",
                    help="restrict to this source value (repeatable). Default: "
                         "every known source present in the shards")
    ap.add_argument("--sample-size", type=int, default=100,
                    help="documents to sample per source (default 100, per "
                         "handoff sec 9.7)")
    ap.add_argument("--show", type=int, default=5,
                    help="samples to print per source")
    ap.add_argument("--snippet-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; change it to read a different 100")
    ap.add_argument("--max-flag-rate", type=float, default=None,
                    help="exit 1 if any source's flagged-document rate exceeds "
                         "this (e.g. 0.10). Default: report only")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir):
        print(f"no such directory: {args.corpus_dir}", file=sys.stderr)
        return 1
    shard_glob = os.path.join(args.corpus_dir, "*.parquet")

    sources = args.source
    if not sources:
        # Discover which sources are actually present, rather than probing all
        # ten and printing eight "not found" blocks.
        import pyarrow.parquet as pq
        import glob as _glob
        present = set()
        paths = sorted(_glob.glob(shard_glob))
        if not paths:
            print(f"no parquet files in {args.corpus_dir}", file=sys.stderr)
            return 1
        for path in paths:
            names = set(pq.ParquetFile(path).schema_arrow.names)
            if "source" not in names:
                present.add(None)     # no source column: inspect as one blob
                break
            present.update(pq.read_table(path, columns=["source"])
                           .column("source").to_pylist())
        if present == {None}:
            sources = [None]
        else:
            ordered = [s for s in KNOWN_SOURCES if s in present]
            sources = ordered + sorted(present - set(ordered))

    # ONE pass over the shards for every source. A pass per source meant
    # re-reading ~15 GB seven times.
    print(f"sampling {len(sources)} source(s) in one pass over "
          f"{args.corpus_dir} ...", flush=True)
    reservoirs, seen = sample_sources(
        shard_glob, sources, sample_size=args.sample_size, seed=args.seed)

    results = []
    for source in sources:
        label = source if source is not None else os.path.basename(
            args.corpus_dir.rstrip(os.sep)) + " (no source column)"
        res = report(reservoirs[source], seen[source], label,
                     show=args.show, snippet_chars=args.snippet_chars)
        if res:
            results.append(res)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"  {'source':30s} {'sampled':>8s} {'flagged':>8s} {'rate':>7s}")
    worst = 0.0
    for r in results:
        worst = max(worst, r["flag_rate"])
        print(f"  {r['source'][:30]:30s} {r['sampled']:8d} {r['flagged']:8d} "
              f"{r['flag_rate']:6.1%}")

    if args.max_flag_rate is not None and worst > args.max_flag_rate:
        print(f"\nFAIL: worst flag rate {worst:.1%} exceeds the "
              f"{args.max_flag_rate:.1%} threshold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
