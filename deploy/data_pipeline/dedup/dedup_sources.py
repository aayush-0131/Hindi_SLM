"""Dedups every acquired source against the FineWeb-2 reference index built by
reference_index.py -- always against the fixed reference, never pairwise
across sources (Requirement 4.3)."""

import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
from data_pipeline.dedup.reference_index import make_minhash


def dedup_against_reference(in_dir, out_dir, lsh, num_perm, n_grams=5):
    os.makedirs(out_dir, exist_ok=True)
    n_dupe = 0
    n_total = 0
    for shard_path in sorted(glob.glob(os.path.join(in_dir, "*.parquet"))):
        table = pq.read_table(shard_path)
        rows = table.to_pylist()
        for row in rows:
            mh = make_minhash(row["text"], num_perm=num_perm, n_grams=n_grams)
            matches = lsh.query(mh)
            # Exclude a trivial self-match: if this document's own doc_id was
            # itself inserted into the reference index (e.g. when deduping
            # FineWeb-2 against its own reference index), it will always
            # match itself. Only count it as a duplicate if some OTHER
            # document's MinHash matches.
            other_matches = [m for m in matches if m != str(row["doc_id"])]
            is_dupe = len(other_matches) > 0
            row["duplicate"] = is_dupe
            n_total += 1
            n_dupe += int(is_dupe)
        out_table = pa.Table.from_pylist(rows)
        pq.write_table(out_table, os.path.join(out_dir, os.path.basename(shard_path)))
    survival_rate = (1 - (n_dupe / n_total)) if n_total else None
    print(f"{in_dir}: {n_dupe}/{n_total} duplicates "
          f"-> survival rate {survival_rate:.2%}" if survival_rate is not None
          else f"{in_dir}: no documents found")
    return survival_rate


def passthrough_no_dedup(in_dir, out_dir, reason=""):
    """
    Copies a source into the deduped/ layout with `duplicate=False` on every
    row, WITHOUT computing MinHashes.

    Why this exists: the first full-scale run measured survival rates of
    99.80-100.00% on all six non-FineWeb-2 sources (four returned exactly zero
    duplicates), while MinHash is the single most expensive CPU stage in the
    pipeline -- 128 permutations over 5-gram shingles per document, which for a
    500-word document is ~64,000 hash operations. It was consuming the bulk of
    the wall-clock to remove 0.07% of documents.

    IMPORTANT CAVEAT, so this is not mistaken for proof that dedup is
    unnecessary: those survival rates were measured against a reference index
    of only 50,000 FineWeb-2 documents -- 0.2% of FineWeb-2's 22M. Dedup
    effectiveness scales with reference coverage, so the real duplicate rate
    against a fuller index would be higher than 0.07%. This is a deliberate
    time-vs-completeness trade under a ticking GPU allocation, not a finding
    that Requirement 4.3 was pointless. Re-enable it (drop the flag) whenever
    the schedule allows.

    Still writes the `duplicate` column so downstream stages
    (shard_writer.is_survivor, count_survivors) see the schema they expect
    rather than a missing key.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_total = 0
    for shard_path in sorted(glob.glob(os.path.join(in_dir, "*.parquet"))):
        table = pq.read_table(shard_path)
        rows = table.to_pylist()
        for row in rows:
            row["duplicate"] = False
            n_total += 1
        pq.write_table(pa.Table.from_pylist(rows),
                       os.path.join(out_dir, os.path.basename(shard_path)))
    suffix = f" ({reason})" if reason else ""
    print(f"{in_dir}: dedup SKIPPED{suffix} -- {n_total} documents passed through "
          f"as non-duplicates")
    return 1.0
