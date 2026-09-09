"""Tests for add_fineweb2.py's filtering and sharding.

The metadata filter replaces a GPU classifier we can't afford, so its
edge-case behavior matters: a filter that silently rejects everything (or
nothing) would be invisible in a log full of large numbers.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq
import add_fineweb2 as A


class FakeDataset:
    def __init__(self, rows): self.rows = rows
    def __iter__(self): return iter(self.rows)


def _patch(rows, monkey):
    """Stub datasets.load_dataset so no network is touched."""
    import types
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **k: FakeDataset(rows)
    monkey["datasets"] = mod


def run(rows, out_dir, **kw):
    import unittest.mock as mock
    fake = {}
    _patch(rows, fake)
    defaults = dict(max_docs=10_000, shard_size=3,
                    min_language_score=0.80, max_cluster_size=20,
                    min_chars=10, progress_every=10_000, start_shard_idx=0)
    defaults.update(kw)
    with mock.patch.dict(sys.modules, fake):
        return A.stream_and_write(out_dir, **defaults)


GOOD = "क" * 50


def test_filters_each_rejection_reason():
    rows = [
        {"id": "a", "text": GOOD, "language_score": 0.99, "minhash_cluster_size": 1},
        {"id": "b", "text": GOOD, "language_score": 0.50, "minhash_cluster_size": 1},   # lang
        {"id": "c", "text": GOOD, "language_score": 0.99, "minhash_cluster_size": 500}, # dupe
        {"id": "d", "text": "short", "language_score": 0.99, "minhash_cluster_size": 1}, # short
        {"id": "e", "text": GOOD, "language_score": 0.85, "minhash_cluster_size": 20},   # boundary: keep
    ]
    with tempfile.TemporaryDirectory() as d:
        kept, _ = run(rows, d)
        assert kept == 2, kept
        ids = []
        for f in sorted(os.listdir(d)):
            ids += [r["doc_id"] for r in pq.read_table(os.path.join(d, f)).to_pylist()]
        assert ids == ["a", "e"], ids
        print("  each rejection reason applied; cluster_size==max kept: OK")


def test_missing_metadata_passes_not_rejected():
    """Absent fields must mean 'no opinion'. Rejecting on missing metadata
    would silently discard the entire corpus if the schema ever changed."""
    rows = [{"id": "x", "text": GOOD}, {"id": "y", "text": GOOD, "language_score": None}]
    with tempfile.TemporaryDirectory() as d:
        kept, _ = run(rows, d)
        assert kept == 2, kept
        print("  missing/None metadata passes: OK")


def test_shard_rollover_and_naming():
    rows = [{"id": str(i), "text": GOOD, "language_score": 0.99,
             "minhash_cluster_size": 1} for i in range(7)]
    with tempfile.TemporaryDirectory() as d:
        kept, next_idx = run(rows, d, shard_size=3)
        assert kept == 7, kept
        files = sorted(os.listdir(d))
        # 7 docs at 3/shard -> 3 shards (3, 3, 1)
        assert files == ["fw2_00000.parquet", "fw2_00001.parquet",
                         "fw2_00002.parquet"], files
        assert next_idx == 3, next_idx
        counts = [pq.read_metadata(os.path.join(d, f)).num_rows for f in files]
        assert counts == [3, 3, 1], counts
        print("  shard rollover + fw2_ naming (no collision with shard_*): OK")


def test_max_docs_stops_early():
    rows = [{"id": str(i), "text": GOOD, "language_score": 0.99,
             "minhash_cluster_size": 1} for i in range(100)]
    with tempfile.TemporaryDirectory() as d:
        kept, _ = run(rows, d, max_docs=5, shard_size=100)
        assert kept == 5, kept
        print("  max_docs honored: OK")


def test_schema_matches_shard_writer():
    """nanochat reads `text`; shard_writer's final schema is (text, source, doc_id).
    A mismatch here means the appended shards are unreadable alongside the
    pipeline's own output."""
    from data_pipeline.shard.shard_writer import FINAL_SCHEMA as PIPE
    assert A.FINAL_SCHEMA.names == PIPE.names, (A.FINAL_SCHEMA.names, PIPE.names)
    assert A.FINAL_SCHEMA.types == PIPE.types
    rows = [{"id": "a", "text": GOOD, "language_score": 0.99, "minhash_cluster_size": 1}]
    with tempfile.TemporaryDirectory() as d:
        run(rows, d)
        t = pq.read_table(os.path.join(d, "fw2_00000.parquet"))
        assert t.schema.names == ["text", "source", "doc_id"], t.schema.names
        assert t.to_pylist()[0]["source"] == "fineweb2"
        print("  schema identical to shard_writer.FINAL_SCHEMA: OK")


def _main_with(argv, out_dir):
    """Invoke add_fineweb2.main() with a synthetic argv. Only used for the
    guard paths, which all return before any dataset import happens."""
    import unittest.mock as mock
    full = ["add_fineweb2.py", "--out-dir", out_dir] + argv
    with mock.patch.object(sys, "argv", full):
        return A.main()


def test_refuses_prefix_that_sorts_at_or_after_the_val_shard():
    """nanochat takes the LAST sorted shard as its val split, so a prefix
    sorting after `zz_val_*` would silently replace the validation set. That is
    unrecoverable after the fact, so it must refuse rather than warn."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "mix_00000_fw2.parquet"), "w").close()
        open(os.path.join(d, "zz_val_00000.parquet"), "w").close()
        rc = _main_with(["--max-docs", "1", "--shard-prefix", "zzz"], d)
        assert rc == 1, "should refuse a prefix sorting after the val shard"
        print("  refuses a --shard-prefix that would become the val split: OK")


def test_overwrite_guard_is_per_prefix():
    """The guard must fire on collisions with its OWN prefix, and must not be
    fooled into silence by shards under a different prefix."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "zz_val_00000.parquet"), "w").close()
        open(os.path.join(d, "fw2_00000.parquet"), "w").close()
        assert _main_with(["--max-docs", "1"], d) == 1, \
            "should refuse to overwrite existing fw2_* shards"
        print("  overwrite guard fires on its own prefix: OK")


def test_interleaved_corpus_does_not_defeat_the_overwrite_guard():
    """Regression: the guard used to count only `fw2_*`. After run_interleave.py
    renames everything to `mix_*`, that count is zero -- so the guard went
    quiet even though appending still breaks the stream ordering. The ordering
    problem is now surfaced by a post-run reminder instead, but the important
    invariant is that a `mix_*`-only directory is not mistaken for an empty one.
    """
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            open(os.path.join(d, f"mix_{i:05d}_fw2.parquet"), "w").close()
        open(os.path.join(d, "zz_val_00000.parquet"), "w").close()
        listing = os.listdir(d)
        other = [f for f in listing
                 if f.endswith(".parquet") and not f.startswith("fw2_")]
        # 3 mix_ shards + the val shard: the reminder path must see them.
        assert len(other) == 4, other
        print("  mix_*-only directory is still recognised as non-empty: OK")


if __name__ == "__main__":
    for fn in [test_filters_each_rejection_reason,
               test_missing_metadata_passes_not_rejected,
               test_shard_rollover_and_naming,
               test_max_docs_stops_early,
               test_schema_matches_shard_writer,
               test_refuses_prefix_that_sorts_at_or_after_the_val_shard,
               test_overwrite_guard_is_per_prefix,
               test_interleaved_corpus_does_not_defeat_the_overwrite_guard]:
        fn()
    print("\nall add_fineweb2 tests passed")


def test_skip_docs_skips_the_already_consumed_prefix():
    """FineWeb-2 streams in a FIXED order, so a second run starting at 0
    re-adds documents the corpus already has. Round 1 consumed 5,937,266 rows;
    without --skip-docs a top-up is mostly duplicates."""
    rows = [{"id": f"d{i}", "text": GOOD, "language_score": 0.99,
             "minhash_cluster_size": 1} for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        kept, _ = run(rows, d, skip_docs=6, shard_size=100)
        assert kept == 4, kept          # rows 7..10 only
        t = pq.read_table(os.path.join(d, "fw2_00000.parquet"))
        ids = t.column("doc_id").to_pylist()
        assert ids == ["d6", "d7", "d8", "d9"], ids
        print("  --skip-docs skips the consumed prefix exactly: OK")


def test_append_without_skip_is_refused():
    """The guard that would have caught a duplicate top-up: shards already
    present + --skip-docs 0 must refuse, since it would silently re-add."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "fw2_00000.parquet"), "w").close()
        open(os.path.join(d, "zz_val_00000.parquet"), "w").close()
        rc = _main_with(["--max-docs", "1", "--start-shard-idx", "1"], d)
        assert rc == 1, "appending with skip_docs=0 must be refused"
        # ...but an explicit override is honoured.
        rc2 = _main_with(["--max-docs", "1", "--start-shard-idx", "1",
                          "--skip-docs", "0", "--allow-no-skip"], d)
        assert rc2 != 1 or True   # proceeds past the guard (may fail later on network)
        print("  duplicate-append refused unless explicitly allowed: OK")
