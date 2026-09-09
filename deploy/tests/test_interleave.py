"""Tests for data_pipeline/shard/interleave_shards.py.

The dangerous properties of this module are (a) it renames the real corpus in
place, and (b) nanochat silently treats whichever shard sorts LAST as the
validation split. So the tests below pin: no data loss, the val shard stays
put and stays last, the rename survives an overlapping old/new name set, and
the whole thing is idempotent.
"""

import json
import os
import shutil
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.shard.interleave_shards import (  # noqa: E402
    apply, mixing_report, plan, survey, _interleave_order,
)

SCHEMA = pa.schema([("text", pa.string()),
                    ("source", pa.string()),
                    ("doc_id", pa.string())])


def _write(path, source, n, text="x"):
    """Write a shard of n rows for `source`. `text` length controls file size,
    which is what the byte-share report reads."""
    rows = [{"text": text, "source": source, "doc_id": f"{source}-{i}"}
            for i in range(n)]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def _round1_layout(d):
    """Reproduce Round 1's actual block structure at 1/10 scale: a big block of
    fineweb2, then internally-ordered sangraha sub-blocks, then the val shard."""
    for i in range(10):
        _write(os.path.join(d, f"fw2_{i:05d}.parquet"), "fineweb2", 50)
    for i in range(2):
        _write(os.path.join(d, f"sg_{i:05d}.parquet"),
               "sangraha_verified_web", 50)
    for i in range(2, 4):
        _write(os.path.join(d, f"sg_{i:05d}.parquet"),
               "sangraha_verified_pdf", 50, text="x" * 40)
    for i in range(4, 6):
        _write(os.path.join(d, f"sg_{i:05d}.parquet"),
               "sangraha_unverified", 50, text="x" * 20)
    _write(os.path.join(d, "zz_val_00000.parquet"), "fineweb2", 10)


def _names(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".parquet"))


def _content_fingerprint(d):
    """Multiset of (source, doc_id) across every shard -- invariant under
    renaming, so it catches any actual data loss or duplication."""
    seen = []
    for name in _names(d):
        t = pq.read_table(os.path.join(d, name), columns=["source", "doc_id"])
        seen.extend(zip(t.column("source").to_pylist(),
                        t.column("doc_id").to_pylist()))
    return sorted(seen)


def test_no_data_loss_and_val_stays_last():
    d = tempfile.mkdtemp()
    try:
        _round1_layout(d)
        before_fp = _content_fingerprint(d)
        before_names = _names(d)
        assert before_names[-1] == "zz_val_00000.parquet"

        result = apply(d, manifest_path=os.path.join(d, "m.json"),
                       verbose=False)

        after_names = _names(d)
        assert len(after_names) == len(before_names), after_names
        # Val shard untouched, by name and by position.
        assert "zz_val_00000.parquet" in after_names
        assert after_names[-1] == "zz_val_00000.parquet", after_names[-1]
        # Every train shard renamed into the mix_ scheme.
        assert all(n.startswith("mix_") for n in after_names[:-1]), after_names
        # No document gained, lost, or duplicated.
        assert _content_fingerprint(d) == before_fp
        # Manifest records the mapping and is valid JSON.
        m = json.load(open(os.path.join(d, "m.json")))
        assert len(m["mapping"]) == len(before_names) - 1
        assert result["total_train_shards"] == len(before_names) - 1
        print("  no data loss; val shard name+position preserved: OK")
    finally:
        shutil.rmtree(d)


def test_idempotent():
    d = tempfile.mkdtemp()
    try:
        _round1_layout(d)
        apply(d, verbose=False)
        first = _names(d)
        second_result = apply(d, verbose=False)
        assert second_result["already_interleaved"] is True
        assert second_result["mapping"] == {}
        assert _names(d) == first
        print("  second run is a no-op: OK")
    finally:
        shutil.rmtree(d)


def test_rename_survives_overlapping_old_and_new_names():
    """The old and new name sets intersect (a corpus already using mix_* names
    but in the wrong order). A naive one-pass rename would clobber a shard that
    had not been moved yet; the two-phase staged rename must not."""
    d = tempfile.mkdtemp()
    try:
        # Deliberately adversarial: correct naming scheme, wrong grouping --
        # all fineweb2 first, so almost every shard has to move.
        for i in range(6):
            _write(os.path.join(d, f"mix_{i:05d}_fw2.parquet"), "fineweb2", 50)
        for i in range(6, 9):
            _write(os.path.join(d, f"mix_{i:05d}_sgp.parquet"),
                   "sangraha_verified_pdf", 50)
        _write(os.path.join(d, "zz_val_00000.parquet"), "fineweb2", 10)
        before_fp = _content_fingerprint(d)

        apply(d, verbose=False)

        assert _content_fingerprint(d) == before_fp
        names = _names(d)
        assert len(names) == 10
        assert names[-1] == "zz_val_00000.parquet"
        # The fineweb2 block must actually have been broken up.
        doms = [n.split("_")[2].split(".")[0] for n in names[:-1]]
        longest = 1
        run = 1
        for a, b in zip(doms, doms[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        assert longest < 6, doms
        print("  overlapping old/new name sets renamed safely: OK")
    finally:
        shutil.rmtree(d)


def test_refuses_without_a_val_shard():
    d = tempfile.mkdtemp()
    try:
        _write(os.path.join(d, "fw2_00000.parquet"), "fineweb2", 10)
        try:
            plan(d)
        except ValueError as exc:
            assert "val glob" in str(exc), exc
            print("  refuses when no val shard matches: OK")
        else:
            raise AssertionError("should have refused with no val shard")
    finally:
        shutil.rmtree(d)


def test_refuses_unknown_source():
    d = tempfile.mkdtemp()
    try:
        _write(os.path.join(d, "fw2_00000.parquet"), "some_new_corpus", 10)
        _write(os.path.join(d, "zz_val_00000.parquet"), "fineweb2", 10)
        try:
            plan(d)
        except ValueError as exc:
            assert "unrecognised source" in str(exc), exc
            print("  refuses an unmapped source rather than inventing a tag: OK")
        else:
            raise AssertionError("should have refused on unknown source")
    finally:
        shutil.rmtree(d)


def test_dry_run_touches_nothing():
    d = tempfile.mkdtemp()
    try:
        _round1_layout(d)
        before = _names(d)
        result = apply(d, dry_run=True, verbose=False)
        assert _names(d) == before
        assert result["mapping"], "dry run should still compute a plan"
        assert not os.path.exists(os.path.join(d, "interleave_manifest.json"))
        print("  dry run leaves the directory untouched: OK")
    finally:
        shutil.rmtree(d)


def test_interleave_beats_source_ordering_on_run_length():
    """The whole point of the module: the longest single-source run must drop
    substantially versus the source-ordered input."""
    d = tempfile.mkdtemp()
    try:
        _round1_layout(d)
        train, _ = survey(d)
        source_ordered = sorted(train, key=lambda s: s["name"])
        before = mixing_report(source_ordered, window=4)
        after = mixing_report(_interleave_order(train), window=4)
        assert before["longest_same_source_run"] == 10, before
        assert after["longest_same_source_run"] <= 3, after
        print(f"  longest run {before['longest_same_source_run']} -> "
              f"{after['longest_same_source_run']}: OK")
    finally:
        shutil.rmtree(d)


def test_ordering_is_deterministic():
    d = tempfile.mkdtemp()
    try:
        _round1_layout(d)
        train, _ = survey(d)
        a = [s["name"] for s in _interleave_order(train)]
        b = [s["name"] for s in _interleave_order(train)]
        assert a == b
        print("  ordering is deterministic across calls: OK")
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    for fn in [test_no_data_loss_and_val_stays_last, test_idempotent,
               test_rename_survives_overlapping_old_and_new_names,
               test_refuses_without_a_val_shard, test_refuses_unknown_source,
               test_dry_run_touches_nothing,
               test_interleave_beats_source_ordering_on_run_length,
               test_ordering_is_deterministic]:
        print(fn.__name__)
        fn()
    print("\nall interleave tests passed")
