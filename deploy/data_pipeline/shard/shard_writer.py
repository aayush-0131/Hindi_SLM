"""
Writes final parquet shards per mixture phase, matching nanochat's dataloader
contract exactly: a `text` column, no other required columns. The LAST shard
written to each phase directory is implicitly treated as that phase's
validation split by nanochat's own convention -- this pipeline does not create
a separately-named val file.
"""

import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq

FINAL_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
    ("doc_id", pa.string()),
])


def is_survivor(row):
    """A row survives if it wasn't flagged a duplicate and wasn't filtered out
    by the classifier. `row.get("keep")` is None (not False) for sources that
    were never classified -- only compare against `is False` explicitly, not
    truthiness, so that's correctly treated as "no opinion, keep it"."""
    if row.get("duplicate"):
        return False
    if row.get("keep") is False:
        return False
    return True


def count_survivors(parquet_dir):
    """Counts rows in an already deduped/filtered source directory that pass
    is_survivor() -- the real post-dedup, post-filter availability, as
    opposed to a raw row count (Requirement 4.5's measured availability)."""
    total = 0
    for shard_path in glob.glob(os.path.join(parquet_dir, "*.parquet")):
        table = pq.read_table(shard_path)
        for row in table.to_pylist():
            if is_survivor(row):
                total += 1
    return total


def write_phase_shards(source_dirs_and_limits, out_dir, shard_size=50_000):
    """
    source_dirs_and_limits: list of (deduped_parquet_dir, max_docs_from_this_source)
    Reads rows from each (already deduped/filtered) source dir, drops
    duplicates/filtered-out rows, narrows to the final schema, and writes
    sequential shards to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    buf = []
    shard_idx = 0

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=FINAL_SCHEMA)
        pq.write_table(table, os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet"))
        shard_idx += 1
        buf = []

    for source_dir, max_docs in source_dirs_and_limits:
        n = 0
        for shard_path in sorted(glob.glob(os.path.join(source_dir, "*.parquet"))):
            table = pq.read_table(shard_path)
            for row in table.to_pylist():
                if not is_survivor(row):
                    continue
                buf.append({"text": row["text"], "source": row["source"], "doc_id": row["doc_id"]})
                n += 1
                if len(buf) >= shard_size:
                    flush()
                if max_docs is not None and n >= max_docs:
                    break
            if max_docs is not None and n >= max_docs:
                break
    flush()
    print(f"Wrote {shard_idx} shard(s) to {out_dir} "
          f"(nanochat treats the last shard in this directory as val)")


def doc_limits_from_token_plan(token_plan, tokens_per_doc, survivor_doc_counts):
    """
    Converts mixture_planner's per-source TOKEN allocation into the per-source
    DOCUMENT limits write_phase_shards() actually consumes.

    These are different units and were previously conflated: the token plan was
    passed straight through as a document cap, so a source allocated e.g. 470M
    tokens was silently asked for 470M *documents* (and, being capped by what
    existed, just contributed everything it had). Planning in tokens and
    writing in documents both require this explicit conversion.

    Clamped to the source's real survivor count so a rounding-up never asks for
    documents that don't exist.
    """
    limits = {}
    for source, tokens in token_plan.items():
        per_doc = tokens_per_doc.get(source)
        available = survivor_doc_counts.get(source, 0)
        if not per_doc or tokens <= 0:
            limits[source] = 0
            continue
        limits[source] = min(int(round(tokens / per_doc)), available)
    return limits
