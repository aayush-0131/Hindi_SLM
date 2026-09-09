"""Tests for the two speedup changes: dedup passthrough and stage timing.

Both are about wall-clock, but the passthrough one has a correctness surface
that matters: if it drops the `duplicate` column, shard_writer.is_survivor()
sees a missing key and the downstream survivor accounting silently changes
meaning. That is the same class of bug as bug 3 (a missing key is None, not
False), so it gets a real test rather than being eyeballed.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.dedup.dedup_sources import passthrough_no_dedup
from data_pipeline.shard.shard_writer import is_survivor, count_survivors


def _write(d, rows, name="shard_00000.parquet"):
    os.makedirs(d, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(d, name))


def test_passthrough_writes_duplicate_column():
    """Downstream stages require the column; without it is_survivor() reads None."""
    with tempfile.TemporaryDirectory() as t:
        src, out = os.path.join(t, "acq"), os.path.join(t, "dedup")
        _write(src, [{"text": f"doc {i}", "source": "finepdfs", "doc_id": str(i),
                      "subsource": None, "quality_score": 1.0} for i in range(7)])
        rate = passthrough_no_dedup(src, out, reason="test")
        assert rate == 1.0, rate

        table = pq.read_table(os.path.join(out, "shard_00000.parquet"))
        assert "duplicate" in table.column_names, table.column_names
        rows = table.to_pylist()
        assert len(rows) == 7
        assert all(r["duplicate"] is False for r in rows)
        assert all(is_survivor(r) for r in rows)
        assert count_survivors(out) == 7
        print("  passthrough writes duplicate=False, all 7 survive: OK")


def test_passthrough_preserves_all_original_columns():
    """Provenance and quality columns must survive; shard_writer reads
    source/doc_id, and FineWeb-2's `keep` column must not be dropped."""
    with tempfile.TemporaryDirectory() as t:
        src, out = os.path.join(t, "acq"), os.path.join(t, "dedup")
        _write(src, [{"text": "x", "source": "finepdfs", "doc_id": "a",
                      "subsource": "pdf", "quality_score": 3.5, "keep": True}])
        passthrough_no_dedup(src, out)
        row = pq.read_table(os.path.join(out, "shard_00000.parquet")).to_pylist()[0]
        for col in ("text", "source", "doc_id", "subsource", "quality_score", "keep"):
            assert col in row, f"{col} dropped"
        assert row["quality_score"] == 3.5 and row["keep"] is True
        print("  passthrough preserves provenance/quality/keep columns: OK")


def test_passthrough_respects_classifier_rejections():
    """A passthrough must not resurrect classifier-rejected rows: keep=False
    still means not-a-survivor even though duplicate=False."""
    with tempfile.TemporaryDirectory() as t:
        src, out = os.path.join(t, "acq"), os.path.join(t, "dedup")
        _write(src, [
            {"text": "a", "source": "fineweb2", "doc_id": "1", "keep": True},
            {"text": "b", "source": "fineweb2", "doc_id": "2", "keep": False},
            {"text": "c", "source": "fineweb2", "doc_id": "3", "keep": True},
        ])
        passthrough_no_dedup(src, out)
        assert count_survivors(out) == 2, count_survivors(out)
        print("  keep=False still excluded after passthrough: OK")


def test_passthrough_handles_multiple_shards_and_empty_dir():
    with tempfile.TemporaryDirectory() as t:
        src, out = os.path.join(t, "acq"), os.path.join(t, "dedup")
        _write(src, [{"text": "a", "source": "s", "doc_id": "1"}], "shard_00000.parquet")
        _write(src, [{"text": "b", "source": "s", "doc_id": "2"}], "shard_00001.parquet")
        passthrough_no_dedup(src, out)
        assert len(os.listdir(out)) == 2, os.listdir(out)
        assert count_survivors(out) == 2

        empty_src, empty_out = os.path.join(t, "e_in"), os.path.join(t, "e_out")
        os.makedirs(empty_src)
        assert passthrough_no_dedup(empty_src, empty_out) == 1.0
        assert count_survivors(empty_out) == 0
        print("  multi-shard and empty-dir handling: OK")


if __name__ == "__main__":
    for fn in [test_passthrough_writes_duplicate_column,
               test_passthrough_preserves_all_original_columns,
               test_passthrough_respects_classifier_rejections,
               test_passthrough_handles_multiple_shards_and_empty_dir]:
        fn()
    print("\nall pipeline-speedup tests passed")


# ---------------------------------------------------------------------------
# Incremental (crash-safe) Sangraha acquisition. Added 2026-09-06 after a
# ~5-hour scan was needed: acquire_all_verified held every document in RAM and
# wrote only at the end, so a long scan that died lost everything -- and a
# second scan into the same directory would rmtree the first one's output.
# ---------------------------------------------------------------------------

def test_incremental_flush_is_durable_and_appends():
    """Documents must be on disk DURING the scan, and must not destroy shards
    already present from an earlier acquisition."""
    import sys, os, tempfile, unittest.mock as mock, types
    import pyarrow.parquet as pq
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.sources.common import append_corpus_docs, normalize_records

    with tempfile.TemporaryDirectory() as d:
        # Pretend an earlier acquisition already wrote here.
        pre = list(normalize_records(
            ({"doc_id": f"old-{i}", "text": "पुराना"} for i in range(5)),
            source="sangraha_verified_pdf"))
        n_pre = append_corpus_docs(iter(pre), d, "shard")
        assert n_pre == 5

        # A later run appends under its own prefix.
        new = list(normalize_records(
            ({"doc_id": f"new-{i}", "text": "नया"} for i in range(7)),
            source="sangraha_verified_pdf"))
        n_new = append_corpus_docs(iter(new), d, "inc-abcd1234-pdf")
        assert n_new == 7

        # BOTH survive -- this is the property write_corpus_docs cannot give.
        files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
        assert len(files) == 2, files
        total = sum(pq.ParquetFile(os.path.join(d, f)).metadata.num_rows
                    for f in files)
        assert total == 12, (total, files)
        # No temp files left behind.
        assert not [f for f in os.listdir(d) if ".tmp-" in f], os.listdir(d)
        print("  incremental append is durable and non-destructive: OK")


def test_incremental_doc_ids_do_not_collide_across_runs():
    """counts[] restarts at 0 each run, so without the run tag a second scan
    would emit doc_ids identical to the first's and corrupt provenance."""
    import sys, os, tempfile, unittest.mock as mock, types
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rows = ([{"type": "pdf", "text": f"पीडीएफ {i}"} for i in range(6)] +
            [{"type": "speech", "text": f"भाषण {i}"} for i in range(2)])

    def fake_load_dataset(*a, **k):
        return iter(rows)

    import data_pipeline.sources.sangraha as S
    seen_ids = []
    with tempfile.TemporaryDirectory() as d:
        out = {t: os.path.join(d, t) for t in ("web", "pdf", "speech")}
        with mock.patch.object(S, "load_dataset", fake_load_dataset):
            for _ in range(2):          # two runs into the SAME directories
                S.acquire_all_verified(out, max_docs={"web": 0, "pdf": 10, "speech": 10},
                                       max_scan=100, flush_every=2)
        import pyarrow.parquet as pq, glob
        ids = []
        for f in glob.glob(os.path.join(out["pdf"], "*.parquet")):
            ids += pq.read_table(f, columns=["doc_id"]).column("doc_id").to_pylist()
        assert len(ids) == 12, len(ids)              # both runs' docs present
        assert len(set(ids)) == 12, "doc_id collision across runs"
        print(f"  {len(ids)} docs across two runs, all doc_ids unique: OK")
