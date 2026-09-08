"""Task 15: unit tests for Gate 2's decision logic (Requirement 5.4).

Flagged in design.md's Testing Strategy as not written. Written now because
run_gate2.py has never actually executed, and these two functions ARE the gate:
_is_non_random decides pass/fail, and a wrong verdict either blocks pretraining
or waves through a broken harness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_harness.gate_report import (
    _primary_metric, _is_non_random, METRIC_PRIORITY, RANDOM_MARGIN,
)


def test_primary_metric_priority_order():
    # acc_norm wins over everything else present
    assert _primary_metric({"f1": 0.9, "acc": 0.8, "acc_norm": 0.7}) == ("acc_norm", 0.7)
    assert _primary_metric({"f1": 0.9, "acc": 0.8}) == ("acc", 0.8)
    assert _primary_metric({"f1": 0.9, "em": 0.6}) == ("em", 0.6)
    assert _primary_metric({"f1": 0.9}) == ("f1", 0.9)
    print("  metric priority:        OK")


def test_primary_metric_fallback_skips_stderr():
    # lighteval emits <metric>_stderr alongside each metric; a stderr value must
    # never be mistaken for the score itself.
    name, val = _primary_metric({"acc_norm_stderr": 0.01, "custom_score": 0.42})
    assert (name, val) == ("custom_score", 0.42), (name, val)
    print("  stderr not picked:      OK")


def test_primary_metric_empty_and_stderr_only():
    assert _primary_metric({}) == (None, None)
    assert _primary_metric({"acc_stderr": 0.02}) == (None, None)
    print("  empty/stderr-only:      OK")


def test_is_non_random_boundaries():
    # 4-choice task: chance 0.25, threshold 0.25 * 1.15 = 0.2875
    assert _is_non_random(0.35, 4) is True
    assert _is_non_random(0.2875, 4) is False      # exactly AT threshold -> not above
    assert _is_non_random(0.28, 4) is False
    assert _is_non_random(0.25, 4) is False        # exactly chance
    # 2-choice task: chance 0.5, threshold 0.575
    assert _is_non_random(0.60, 2) is True
    assert _is_non_random(0.55, 2) is False
    print(f"  thresholds (margin {RANDOM_MARGIN}): OK")


def test_is_non_random_handles_missing_score():
    # A task that ran but yielded no parseable metric must fail closed, not
    # raise a TypeError mid-gate.
    assert _is_non_random(None, 4) is False
    assert _is_non_random(None, 2) is False
    print("  None fails closed:      OK")


def test_boolq_binary_task_is_the_fragile_case():
    """Documents a real Gate 2 risk rather than asserting it away.

    community_boolq_hindi has num_choices=2, so its pass threshold is 0.575.
    Goldfish-Hindi is a 125M model; near-chance accuracy on a binary
    reading-comprehension task is the EXPECTED outcome for a model that small,
    and small models on BoolQ are well known to sit near the majority-class
    rate. Requirement 5.5's intent is 'the harness works', not 'the baseline is
    good' -- but _is_non_random cannot tell those apart, so a legitimately
    weak-but-correct result fails the whole gate.

    Concretely: 0.55 on BoolQ-hi fails, while the same 0.55 on any 4-choice
    task passes comfortably. That asymmetry is what makes BoolQ the task most
    likely to block Gate 2 spuriously.
    """
    assert _is_non_random(0.55, 2) is False   # would FAIL the gate
    assert _is_non_random(0.55, 4) is True    # identical score, passes
    print("  BoolQ fragility:        OK (documented, see docstring)")


if __name__ == "__main__":
    for fn in [test_primary_metric_priority_order,
               test_primary_metric_fallback_skips_stderr,
               test_primary_metric_empty_and_stderr_only,
               test_is_non_random_boundaries,
               test_is_non_random_handles_missing_score,
               test_boolq_binary_task_is_the_fragile_case]:
        fn()
    print("\nall gate2 logic tests passed")
