#!/usr/bin/env python3
"""
Streams FineWeb-2 hin_Deva straight into final training shards -- no GPU
classifier, no MinHash, no mixture planning.

WHY THIS EXISTS (and what it deliberately gives up):

Requirement 4.1 mandates the `finepdfs_edu_classifier_hin_Deva` quality
classifier on FineWeb-2 at a 20-25% keep rate. That is not affordable: the
Hindi split holds 22,095,985 documents, and measured classifier throughput
puts a full pass at 157-393 GPU-hours against a 40-hour total allocation. A
partial pass is worse than useless -- 50,000 documents kept ~11,000, about
16M tokens, 0.5% of the corpus.

Requirement 4.3's MinHash dedup is also skipped, for a similar reason: ~128
hashes over ~600 shingles per document is ~380 billion hash operations at 5M
documents, roughly 8 hours in Python. It is also largely redundant here --
FineWeb-2 is already deduplicated by its own pipeline, and measured
cross-source duplicate rate on this corpus was 0.07%.

WHAT WE DO INSTEAD: filter on the per-document metadata FineWeb-2 already
ships, which costs nothing:
  - `language_score`   -- confidence the document really is Hindi; drops
                          misdetected pages that the label alone would admit.
  - `minhash_cluster_size` -- FineWeb-2's OWN near-duplicate cluster size. A
                          large value means the document is boilerplate or
                          mirrored across many pages. This is the dedup signal,
                          precomputed upstream at full-corpus scale, which is
                          strictly better coverage than anything we could build
                          in-memory here.

This is a genuine reduction in quality control versus the spec, not an
equivalent substitution, and it is recorded as such. It is also the only way
the corpus exists at all within the allocation.

Output: parquet shards named fw2_NNNNN.parquet, matching shard_writer's final
schema (text, source, doc_id) so nanochat's dataloader reads them directly.
The distinct prefix means these can be written alongside run_pipeline.py's
shard_NNNNN.parquet files without name collisions -- so this can run
CONCURRENTLY with the main pipeline.
"""

import argparse
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

FINAL_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
    ("doc_id", pa.string()),
])

# Defaults chosen to be permissive: the goal here is volume with the worst
# documents removed, not a tight quality bar we cannot validate.
DEFAULT_MIN_LANGUAGE_SCORE = 0.80
DEFAULT_MAX_CLUSTER_SIZE = 20
DEFAULT_MIN_CHARS = 200


def stream_and_write(out_dir, max_docs, shard_size, min_language_score,
                     max_cluster_size, min_chars, progress_every,
                     start_shard_idx, shard_prefix="fw2", skip_docs=0):
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-2", "hin_Deva",
                      split="train", streaming=True)  # train only (Requirement 4.4)

    buf = []
    shard_idx = start_shard_idx
    kept = scanned = 0
    rej_lang = rej_dupe = rej_short = 0
    t0 = time.time()

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        path = os.path.join(out_dir, f"{shard_prefix}_{shard_idx:05d}.parquet")
        pq.write_table(pa.Table.from_pylist(buf, schema=FINAL_SCHEMA), path)
        shard_idx += 1
        buf = []

    for row in ds:
        scanned += 1
        # SKIP the prefix of the stream that a previous run already consumed.
        # `load_dataset(streaming=True)` iterates shards in a fixed order, so
        # the first N documents are reproducible -- which is what makes this a
        # valid way to reach FRESH documents. Without it, a second run re-adds
        # the same documents the corpus already contains: Round 1 consumed
        # 5,937,266 rows, so a fresh run from 0 would duplicate ~5.9M docs and
        # add far fewer new unique tokens than the --max-docs figure suggests.
        if scanned <= skip_docs:
            if progress_every and scanned % (progress_every * 5) == 0:
                print(f"[fineweb2] skipping {scanned:,}/{skip_docs:,} "
                      f"already-consumed rows", flush=True)
            continue
        text = row.get("text") or ""

        if len(text) < min_chars:
            rej_short += 1
        else:
            # Absent metadata must not silently reject everything, so a missing
            # field is treated as "no opinion" and passes.
            lang = row.get("language_score")
            cluster = row.get("minhash_cluster_size")
            if lang is not None and lang < min_language_score:
                rej_lang += 1
            elif cluster is not None and cluster > max_cluster_size:
                rej_dupe += 1
            else:
                buf.append({
                    "text": text,
                    "source": "fineweb2",
                    "doc_id": str(row.get("id", f"fw2-{scanned}")),
                })
                kept += 1
                if len(buf) >= shard_size:
                    flush()

        if kept >= max_docs:
            break
        if scanned % progress_every == 0:
            rate = scanned / max(time.time() - t0, 1e-9)
            considered = max(scanned - skip_docs, 1)
            print(f"[fineweb2] scanned {scanned:,} kept {kept:,} "
                  f"({100 * kept / considered:.1f}%) | rejected: lang={rej_lang:,} "
                  f"dupe={rej_dupe:,} short={rej_short:,} | {rate:,.0f} rows/s",
                  flush=True)

    flush()
    dt = time.time() - t0
    print(f"\n[fineweb2] DONE in {dt / 60:.1f} min")
    print(f"  scanned:  {scanned:,}" +
          (f" ({skip_docs:,} skipped as already-consumed)" if skip_docs else ""))
    # Keep rate over rows CONSIDERED, not total scanned -- otherwise a large
    # --skip-docs makes the rate unreadable against Round 1's 84.2% baseline.
    considered = max(scanned - skip_docs, 1)
    print(f"  kept:     {kept:,} ({100 * kept / considered:.1f}% of "
          f"{considered:,} considered)")
    print(f"  rejected: language_score<{min_language_score}: {rej_lang:,} | "
          f"cluster_size>{max_cluster_size}: {rej_dupe:,} | "
          f"<{min_chars} chars: {rej_short:,}")
    print(f"  shards:   {shard_prefix}_{start_shard_idx:05d}.."
          f"{shard_prefix}_{shard_idx - 1:05d} in {out_dir}")
    return kept, shard_idx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="where to write fw2_*.parquet shards")
    ap.add_argument("--max-docs", type=int, default=5_000_000,
                    help="stop after keeping this many documents "
                         "(5M ~= 3.1B tokens at the measured 624 tokens/doc)")
    ap.add_argument("--shard-size", type=int, default=50_000,
                    help="documents per shard; also the RAM flush interval")
    ap.add_argument("--min-language-score", type=float,
                    default=DEFAULT_MIN_LANGUAGE_SCORE)
    ap.add_argument("--max-cluster-size", type=int,
                    default=DEFAULT_MAX_CLUSTER_SIZE)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--progress-every", type=int, default=100_000)
    ap.add_argument("--allow-no-skip", action="store_true",
                    help="permit appending with --skip-docs 0, accepting "
                         "duplicate documents")
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="skip this many rows of the stream before keeping "
                         "anything. REQUIRED when appending to a corpus an "
                         "earlier run already fed: the stream order is fixed, "
                         "so starting from 0 re-adds documents you already "
                         "have. Round 1 consumed 5,937,266 rows -> use 6000000.")
    ap.add_argument("--start-shard-idx", type=int, default=0,
                    help="first shard number; raise it if shards with this "
                         "prefix already exist in --out-dir and you are "
                         "adding more")
    ap.add_argument("--shard-prefix", default="fw2",
                    help="filename prefix for written shards (default: fw2). "
                         "Must sort BEFORE the val shard -- see the reminder "
                         "printed after a successful run.")
    args = ap.parse_args()

    listing = os.listdir(args.out_dir) if os.path.isdir(args.out_dir) else []
    # Overwrite guard: only shards sharing THIS prefix can actually be
    # clobbered, since the filename is prefix + index.
    existing = [f for f in listing
                if f.startswith(f"{args.shard_prefix}_")
                and f.endswith(".parquet")]
    if existing and args.start_shard_idx == 0:
        print(f"REFUSING TO RUN: {len(existing)} "
              f"{args.shard_prefix}_*.parquet shards already exist in "
              f"{args.out_dir} and --start-shard-idx is 0, which would "
              f"overwrite them. Pass --start-shard-idx {len(existing)} to "
              f"append, or delete them first.", file=sys.stderr)
        return 1

    # Ordering guard, separate from the overwrite guard above. nanochat streams
    # shards in SORTED filename order, so appending a block of shards under one
    # prefix puts every new document in one contiguous run of the stream --
    # which is precisely the source-ordering problem that cost Round 1 a +42%
    # train-loss spike once per epoch (ROUND2_HANDOFF.md sec 9.1). Adding data
    # is therefore only half the job; the corpus has to be re-interleaved
    # afterwards. Warn loudly here rather than let that be discovered from a
    # loss curve 5 hours into a GPU window.
    other_shards = [f for f in listing
                    if f.endswith(".parquet")
                    and not f.startswith(f"{args.shard_prefix}_")]
    val_shards = sorted(f for f in listing if f.startswith("zz_"))
    if val_shards:
        # A new shard sorting at/after the val shard would silently BECOME the
        # validation set. Refuse rather than warn: this one is unrecoverable.
        sample_name = f"{args.shard_prefix}_{args.start_shard_idx:05d}.parquet"
        if sample_name >= val_shards[0]:
            print(f"REFUSING TO RUN: shards named {sample_name!r} sort at or "
                  f"after the validation shard {val_shards[0]!r}. nanochat "
                  f"takes the LAST sorted shard as its val split, so these "
                  f"would silently replace the validation set. Choose a "
                  f"--shard-prefix that sorts earlier.", file=sys.stderr)
            return 1

    if existing and args.skip_docs == 0:
        print(f"REFUSING TO RUN: {len(existing)} shard(s) with prefix "
              f"{args.shard_prefix!r} already exist, but --skip-docs is 0. "
              f"FineWeb-2 streams in a FIXED order, so this would re-add the "
              f"same documents those shards already contain -- inflating the "
              f"corpus with duplicates while appearing to grow it. Round 1 "
              f"consumed 5,937,266 rows; pass --skip-docs 6000000 (or 0 "
              f"explicitly via --allow-no-skip if you really want overlap).",
              file=sys.stderr)
        if not args.allow_no_skip:
            return 1

    print(f"Streaming FineWeb-2 hin_Deva (train) -> {args.out_dir}")
    print(f"  target: {args.max_docs:,} kept docs "
          f"(~{args.max_docs * 624 / 1e9:.1f}B tokens at 624 tok/doc)")
    print(f"  filters: language_score>={args.min_language_score}, "
          f"minhash_cluster_size<={args.max_cluster_size}, "
          f">={args.min_chars} chars\n")
    stream_and_write(
        args.out_dir, args.max_docs, args.shard_size, args.min_language_score,
        args.max_cluster_size, args.min_chars, args.progress_every,
        args.start_shard_idx, args.shard_prefix, args.skip_docs,
    )

    if other_shards:
        print(f"\n{'=' * 68}\nNEXT STEP REQUIRED -- the corpus is now "
              f"source-ordered again\n{'=' * 68}")
        print(f"{len(other_shards)} shard(s) under other prefixes were already "
              f"in {args.out_dir}, so the\nshards just written form one "
              f"contiguous block in nanochat's sorted\nstream. That is the "
              f"Round 1 ordering problem (+42% train-loss spike\nonce per "
              f"epoch). Re-interleave before training:\n")
        print(f"    python3 run_interleave.py --corpus-dir {args.out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
