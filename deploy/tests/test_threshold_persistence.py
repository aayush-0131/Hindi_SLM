"""Threshold persistence + --reuse-fineweb2 support.

design.md's Batch/Job Contract for finepdfs_edu_classifier.py already
specified that calibrate_threshold's result is written to a JSON artifact "so
apply_filter can be rerun deterministically against the same threshold without
recalibrating" -- but it was never implemented. Consequence: every rerun scored
the calibration sample on the GPU again, on top of the full scoring pass, so a
50K-document FineWeb-2 pool put the classifier over the same corpus twice.
That GPU time is GPU the pretraining gates need.

torch is imported at classifier module scope and isn't installed locally, so
the two pure-IO functions are loaded directly from source rather than by
importing the package.
"""
import sys, os, json, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SRC = os.path.join(os.path.dirname(HERE), "data_pipeline", "classify",
                   "finepdfs_edu_classifier.py")


def _load_io_functions():
    """Exec only the persistence helpers, avoiding the module-level torch import."""
    text = open(SRC).read()
    marker = "THRESHOLD_FILENAME ="
    assert marker in text, "persistence block missing from classifier"
    tail = text[text.index(marker):]
    mod = types.ModuleType("clf_io")
    mod.os = os
    exec(compile(tail, SRC, "exec"), mod.__dict__)
    return mod


def test_roundtrip():
    clf = _load_io_functions()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "classified", "fineweb2")
        path = clf.save_threshold(out, 2.71828, 50_000, (0.20, 0.25))
        assert os.path.exists(path)
        data = json.load(open(path))
        assert abs(data["threshold"] - 2.71828) < 1e-9
        assert data["n_scored"] == 50_000
        assert data["target_keep_rate"] == [0.20, 0.25]
        assert abs(clf.load_threshold(out) - 2.71828) < 1e-9
        print("  save/load roundtrip: OK")


def test_missing_returns_none_not_raise():
    """--reuse-fineweb2 gates on this returning None; a raise would abort the run."""
    clf = _load_io_functions()
    with tempfile.TemporaryDirectory() as d:
        assert clf.load_threshold(d) is None
        assert clf.load_threshold(os.path.join(d, "does", "not", "exist")) is None
        print("  missing calibration -> None: OK")


def test_zero_threshold_is_not_confused_with_missing():
    """calibrate_threshold returns 0.0 for an empty score list. 0.0 is a real
    value; `if threshold:` would treat it as absent and silently recalibrate,
    so the reuse check must test `is not None`."""
    clf = _load_io_functions()
    with tempfile.TemporaryDirectory() as d:
        clf.save_threshold(d, 0.0, 0, (0.20, 0.25))
        loaded = clf.load_threshold(d)
        assert loaded == 0.0
        assert loaded is not None
        print("  threshold 0.0 distinguished from missing: OK")


def test_reuse_gate_uses_is_not_none():
    """Pins the run_pipeline.py guard against regressing to a truthiness check."""
    text = open(os.path.join(os.path.dirname(HERE), "run_pipeline.py")).read()
    assert "classifier.load_threshold(classified_dir) is not None" in text, \
        "reuse guard must compare against None explicitly, not use truthiness"
    print("  run_pipeline reuse guard compares to None: OK")


if __name__ == "__main__":
    for fn in [test_roundtrip, test_missing_returns_none_not_raise,
               test_zero_threshold_is_not_confused_with_missing,
               test_reuse_gate_uses_is_not_none]:
        fn()
    print("\nall threshold-persistence tests passed")
