"""Shared CorpusDoc schema and atomic parquet writing helpers."""

import os
import shutil
import uuid
import pyarrow as pa
import pyarrow.parquet as pq

CORPUS_DOC_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("text", pa.string()),
    ("source", pa.string()),
    ("subsource", pa.string()),
    ("quality_score", pa.float64()),
])


def normalize_records(records, source, subsource=None):
    """records: iterable of dicts with at least {'doc_id', 'text'}
    (+ optional 'quality_score' / 'subsource')."""
    for r in records:
        yield {
            "doc_id": str(r["doc_id"]),
            "text": r["text"],
            "source": source,
            "subsource": r.get("subsource", subsource),
            "quality_score": r.get("quality_score"),
        }


def append_corpus_docs(records, out_dir, prefix, shard_size=50_000,
                       schema=CORPUS_DOC_SCHEMA, start_idx=0):
    """Append parquet shards to `out_dir` WITHOUT removing what is already there.

    `write_corpus_docs` is all-or-nothing by design: it stages to a temp dir and
    `shutil.rmtree`s the destination before renaming in. That is right for a
    stage that must not be half-applied -- but it makes two things impossible:

      1. crash safety on a long acquisition. A 5-hour Sangraha scan that dies
         at hour 5 loses everything, because nothing is durable until the end.
      2. topping up an existing acquisition. A second scan into the same
         directory DELETES the first scan's output (ROUND2_HANDOFF.md sec 7.12
         records this biting once already).

    So this variant writes shards as it goes, under a caller-supplied `prefix`
    that must be unique per run, and never touches existing files. Downstream
    consumers glob `*.parquet`, so mixed prefixes are read transparently.

    Returns the number of documents written.
    """
    os.makedirs(out_dir, exist_ok=True)
    buf = []
    shard_idx = start_idx
    n_written = 0

    def flush():
        nonlocal buf, shard_idx, n_written
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=schema)
        # Write to a temp name then rename, so a reader globbing *.parquet can
        # never observe a partially-written shard.
        final = os.path.join(out_dir, f"{prefix}_{shard_idx:05d}.parquet")
        tmp = final + f".tmp-{uuid.uuid4().hex[:6]}"
        pq.write_table(table, tmp)
        os.rename(tmp, final)
        n_written += len(buf)
        shard_idx += 1
        buf = []

    for rec in records:
        buf.append(rec)
        if len(buf) >= shard_size:
            flush()
    flush()
    return n_written


def write_corpus_docs(records, out_dir, shard_size=50_000, schema=CORPUS_DOC_SCHEMA):
    """
    Writes `records` (dicts matching `schema`) to parquet shards under a fresh
    temp directory, then atomically renames it into `out_dir` on success -- so a
    failed/partial write is never mistaken for a complete stage output.
    """
    tmp_dir = out_dir.rstrip("/") + f".tmp-{uuid.uuid4().hex[:8]}"
    os.makedirs(tmp_dir, exist_ok=True)
    buf = []
    shard_idx = 0
    n_written = 0

    def flush():
        nonlocal buf, shard_idx, n_written
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=schema)
        pq.write_table(table, os.path.join(tmp_dir, f"shard_{shard_idx:05d}.parquet"))
        n_written += len(buf)
        shard_idx += 1
        buf = []

    for rec in records:
        buf.append(rec)
        if len(buf) >= shard_size:
            flush()
    flush()

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp_dir, out_dir)
    return n_written
