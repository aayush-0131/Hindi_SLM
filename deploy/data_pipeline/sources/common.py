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
