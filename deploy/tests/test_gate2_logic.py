"""Task 15: unit tests for Gate 2's decision logic (Requirement 5.4).

Flagged in design.md's Testing Strategy as not written. Written now because
run_gate2.py has never actually executed, and these two functions ARE the gate:
_is_non_random decides pass/fail, and a wrong verdict either blocks pretraining
or waves through a broken harness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_harness import gate_report
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


# ---------------------------------------------------------------------------
# Round 2 fixes: the two things that would have made Gate 2's first-ever real
# run fail for reasons unrelated to the harness being correct.
# ---------------------------------------------------------------------------

def test_binary_tasks_are_curated_out_of_the_gate_not_failed_by_it():
    """BoolQ has num_choices=2, so _is_non_random demands >0.575. A 125M
    Goldfish near chance on binary Hindi reading comprehension is EXPECTED, and
    under the old code that single task failed the entire gate -- blocking
    Requirement 8.2's decay selection, which is what Gate 2 exists to enable.

    It must now land in the measured-but-not-gating bucket instead.
    """
    from eval_harness import config as C
    assert C.GATE_ON_BINARY_TASKS is False, \
        "binary tasks must not gate by default"
    assert C.MIN_DISCRIMINATIVE_CHOICES >= 3, C.MIN_DISCRIMINATIVE_CHOICES

    binary = [(s, n) for s, n in C.CANDIDATE_TASKS if n < 3]
    assert binary, "expected at least one binary candidate (BoolQ)"
    for spec, n in binary:
        chance = 1.0 / n
        required = chance * gate_report.RANDOM_MARGIN
        # The precise reason it is not gateable: the bar sits above what a
        # size-matched baseline plausibly reaches on a binary task.
        assert required > 0.5, (spec, required)
        assert n < C.MIN_DISCRIMINATIVE_CHOICES
    print(f"  {len(binary)} binary task(s) excluded from gating, still measured: OK")


def test_multi_choice_tasks_still_gate():
    """The fix must not weaken the gate for tasks where near-chance IS a real
    signal -- 4-choice tasks have a 0.25 chance baseline and stay strict."""
    from eval_harness import config as C
    gating = [(s, n) for s, n in C.CANDIDATE_TASKS
              if n >= C.MIN_DISCRIMINATIVE_CHOICES]
    assert len(gating) >= 3, gating
    for spec, n in gating:
        assert not gate_report._is_non_random(1.0 / n, n), spec
        assert gate_report._is_non_random(1.0 / n * 1.5, n), spec
    print(f"  {len(gating)} multi-choice task(s) still gate strictly: OK")


def test_sample_limit_flag_is_detected_not_assumed():
    """Previously hardcoded to `--max-samples`, an unverified guess. A wrong
    guess makes lighteval reject the argument and every task 'fails', which
    would read as a harness failure. Now resolved against --help."""
    import unittest.mock as mock
    from eval_harness import lighteval_runner as R

    for help_text, expected in [
        ("  --max-samples INT   cap samples", "--max-samples"),
        ("  --limit INT         limit examples", "--limit"),
        ("  --max_samples INT", "--max_samples"),
    ]:
        R._DETECTED_FLAG = None
        R._DETECTION_FAILED = False
        fake = mock.Mock(stdout=help_text, stderr="")
        with mock.patch.object(R.subprocess, "run", return_value=fake):
            assert R.detect_sample_limit_flag(force=True) == expected, help_text

    # No recognised flag -> None, so run_task omits it rather than passing an
    # argument lighteval would reject.
    R._DETECTED_FLAG = None
    R._DETECTION_FAILED = False
    fake = mock.Mock(stdout="  --output-dir PATH", stderr="")
    with mock.patch.object(R.subprocess, "run", return_value=fake):
        assert R.detect_sample_limit_flag(force=True) is None
    print("  sample-limit flag detected from --help, degrades safely: OK")


def test_limit_flag_word_boundary():
    """`--limit` must not be matched inside `--limit-tokens`, which would send
    a flag that takes a different argument entirely."""
    import unittest.mock as mock
    from eval_harness import lighteval_runner as R
    R._DETECTED_FLAG = None
    R._DETECTION_FAILED = False
    fake = mock.Mock(stdout="  --limit-tokens INT   something else", stderr="")
    with mock.patch.object(R.subprocess, "run", return_value=fake):
        assert R.detect_sample_limit_flag(force=True) is None
    print("  --limit does not false-match --limit-tokens: OK")
