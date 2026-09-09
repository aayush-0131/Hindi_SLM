#!/usr/bin/env python3
"""Write final decay-phase shards from a plan already produced by
`run_decay_corpus.py build` -- the step that command's own output says is
"not wired up in this command yet."

Usage:
    python3 write_decay_shards.py --variant a --artifacts <dir>
    python3 run_interleave.py --corpus-dir <dir>/final/decay_a   # MANDATORY after

Requires (produced by earlier `run_decay_corpus.py` stages, in this order):
    <artifacts>/decay_<variant>_plan.json      (from `build`)
    <artifacts>/decay_tokens_per_doc.json       (from `measure`)
    <artifacts>/acquired/<source>/              (from `acquire`, per component)
"""

import argparse
import json
import os

from data_pipeline.mixture.decay_planner import COMPONENT_SOURCES, decay_targets
from data_pipeline.shard.shard_writer import (
    count_survivors, doc_limits_from_token_plan, write_phase_shards,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["a", "b"])
    ap.add_argument("--artifacts", default="data_pipeline_artifacts")
    ap.add_argument("--shard-size", type=int, default=50_000)
    args = ap.parse_args()

    plan_path = os.path.join(args.artifacts, f"decay_{args.variant}_plan.json")
    with open(plan_path) as fh:
        plan = json.load(fh)

    tpd_path = os.path.join(args.artifacts, "decay_tokens_per_doc.json")
    with open(tpd_path) as fh:
        tokens_per_doc = json.load(fh)

    targets = decay_targets(args.variant)

    print(f"Decay {args.variant.upper()}: counting real survivor docs per "
          f"component (post-dedup/filter availability, not raw row count)")
    survivor_counts = {}
    for component in targets:
        source = COMPONENT_SOURCES[component]
        source_dir = os.path.join(args.artifacts, "acquired", source)
        survivor_counts[component] = count_survivors(source_dir)
        print(f"  {component:34s} {survivor_counts[component]:>12,} "
              f"survivor docs  ({source_dir})")

    limits = doc_limits_from_token_plan(
        plan["allocation"], tokens_per_doc, survivor_counts)

    print("\nDocument limits derived from the token plan:")
    source_dirs_and_limits = []
    for component, doc_limit in sorted(limits.items(), key=lambda kv: -kv[1]):
        source = COMPONENT_SOURCES[component]
        source_dir = os.path.join(args.artifacts, "acquired", source)
        print(f"  {component:34s} {doc_limit:>12,} docs  ({source})")
        source_dirs_and_limits.append((source_dir, doc_limit))

    out_dir = os.path.join(args.artifacts, "final", f"decay_{args.variant}")
    print(f"\nwriting shards -> {out_dir}")
    write_phase_shards(source_dirs_and_limits, out_dir, shard_size=args.shard_size)

    print(f"\nNOW MANDATORY (not optional): "
          f"python3 run_interleave.py --corpus-dir {out_dir}")
    print("Without it, shards land as contiguous per-source blocks and "
          "nanochat streams them in sorted-filename order -- recreating the "
          "same +42% train-loss spike per epoch that Round 1 hit.")


if __name__ == "__main__":
    main()
