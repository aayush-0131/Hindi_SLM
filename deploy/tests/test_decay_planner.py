"""Tests for data_pipeline/mixture/decay_planner.py.

The properties that matter: Requirement 3.3's Decay-A exclusion of
fineweb-edu-hindi must be enforced (it is the only thing the A/B comparison in
Requirement 8.2 measures), upsampling must stay bounded, and a mixture that
cannot meet Requirement 3.7's ratios must SAY SO rather than silently ship.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.mixture.decay_planner import (  # noqa: E402
    DEFAULT_MAX_UPSAMPLE, decay_targets, plan_decay_mixture,
)

BUDGET = 1_000_000_000
ABUNDANT = {
    "sangraha_verified_pdf_textbooks": 10_000_000_000,
    "hindi_wikipedia": 10_000_000_000,
    "sangraha_verified_speech_nptel": 10_000_000_000,
    "fineweb2_top2pct": 10_000_000_000,
    "fineweb_edu_hindi": 10_000_000_000,
}


def test_hits_nominal_ratios_when_supply_is_abundant():
    for variant in ("a", "b"):
        avail = {k: v for k, v in ABUNDANT.items()
                 if k in decay_targets(variant)}
        plan = plan_decay_mixture(variant, avail, BUDGET)
        for component, nominal in plan["targets"].items():
            got = plan["realized_shares"][component]
            assert abs(got - nominal) < 0.01, (variant, component, got, nominal)
        assert all(f == 1.0 for f in plan["upsample"].values()), plan["upsample"]
        assert abs(plan["realized_total"] - BUDGET) < BUDGET * 0.001
    print("  abundant supply reproduces Requirement 3.7's ratios exactly: OK")


def test_decay_a_refuses_fineweb_edu_hindi():
    """Requirement 3.3: decay-only source, and Decay A is native-only
    (Requirement 7.3). Leaking it into A would make the A/B comparison
    meaningless."""
    assert "fineweb_edu_hindi" not in decay_targets("a")
    avail = dict(ABUNDANT)
    try:
        plan_decay_mixture("a", avail, BUDGET)
    except AssertionError as exc:
        assert "fineweb_edu_hindi" in str(exc), exc
        print("  Decay A refuses fineweb-edu-hindi even if supplied: OK")
    else:
        raise AssertionError("Decay A should refuse fineweb_edu_hindi supply")


def test_decay_b_includes_fineweb_edu_hindi_at_15pct():
    plan = plan_decay_mixture("b", ABUNDANT, BUDGET)
    assert abs(plan["realized_shares"]["fineweb_edu_hindi"] - 0.15) < 0.01
    print("  Decay B carries fineweb-edu-hindi at 15%: OK")


def test_upsampling_is_bounded():
    """A tiny source must not be repeated without limit -- the decay phase
    shapes the shipped model, so heavy repetition risks memorisation."""
    avail = dict(ABUNDANT)
    avail["sangraha_verified_speech_nptel"] = 1_000_000  # 1M vs a 180M share
    plan = plan_decay_mixture("a", {k: v for k, v in avail.items()
                                    if k in decay_targets("a")},
                              BUDGET, max_upsample=2.0)
    factor = plan["upsample"]["sangraha_verified_speech_nptel"]
    assert factor <= 2.0 + 1e-9, factor
    allocated = plan["allocation"]["sangraha_verified_speech_nptel"]
    assert allocated <= 2_000_000, allocated
    print(f"  tiny source capped at {factor:.2f}x, not backfilled unbounded: OK")


def test_shortfall_is_reported_not_hidden():
    """Every component starved -> the plan must report a total shortfall."""
    starved = {k: 1_000_000 for k in decay_targets("a")}
    plan = plan_decay_mixture("a", starved, BUDGET)
    assert plan["realized_total"] < BUDGET
    assert any("TOTAL:" in d for d in plan["deviations"]), plan["deviations"]
    print("  total shortfall surfaced as a deviation: OK")


def test_rounding_does_not_produce_a_spurious_shortfall():
    """_water_fill truncates per component, so a satisfiable plan can land a
    few tokens light. That must not be reported as a real shortfall."""
    plan = plan_decay_mixture("a", {k: v for k, v in ABUNDANT.items()
                                    if k in decay_targets("a")},
                              3452 * 524_288)
    assert not any("TOTAL:" in d for d in plan["deviations"]), plan["deviations"]
    print("  integer-rounding remainder not flagged as shortfall: OK")


def test_zero_availability_is_flagged_loudly():
    avail = {k: v for k, v in ABUNDANT.items() if k in decay_targets("a")}
    avail["hindi_wikipedia"] = 0
    plan = plan_decay_mixture("a", avail, BUDGET)
    assert plan["allocation"]["hindi_wikipedia"] == 0
    assert any("hindi_wikipedia" in d and "0.0%" in d
               for d in plan["deviations"]), plan["deviations"]
    print("  a never-acquired component is reported at 0%, not skipped: OK")


def test_subslice_substitutions_are_recorded():
    """Requirement 3.7 names sub-slices (textbook/eGyanKosh, NPTEL, top-2%)
    that are not identifiable in the released data. Each substitution must
    appear in the plan rather than being silently assumed."""
    plan = plan_decay_mixture("a", {k: v for k, v in ABUNDANT.items()
                                    if k in decay_targets("a")}, BUDGET)
    joined = " ".join(plan["deviations"])
    for phrase in ("eGyanKosh", "NPTEL", "top 2%"):
        assert phrase in joined, (phrase, plan["deviations"])
    print("  all three sub-slice substitutions recorded: OK")


def test_rejects_bad_inputs():
    try:
        decay_targets("c")
    except ValueError:
        pass
    else:
        raise AssertionError("variant 'c' should be rejected")
    try:
        plan_decay_mixture("a", ABUNDANT, BUDGET, max_upsample=0.5)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("max_upsample < 1.0 should be rejected")
    print("  invalid variant and max_upsample rejected: OK")


def test_default_upsample_is_conservative():
    """Documented intent: far below the stable phase's ~4-epoch ceiling,
    because these sets are orders of magnitude smaller."""
    assert DEFAULT_MAX_UPSAMPLE <= 2.0, DEFAULT_MAX_UPSAMPLE
    print(f"  default max_upsample is {DEFAULT_MAX_UPSAMPLE}x: OK")


if __name__ == "__main__":
    for fn in [test_hits_nominal_ratios_when_supply_is_abundant,
               test_decay_a_refuses_fineweb_edu_hindi,
               test_decay_b_includes_fineweb_edu_hindi_at_15pct,
               test_upsampling_is_bounded,
               test_shortfall_is_reported_not_hidden,
               test_rounding_does_not_produce_a_spurious_shortfall,
               test_zero_availability_is_flagged_loudly,
               test_subslice_substitutions_are_recorded,
               test_rejects_bad_inputs,
               test_default_upsample_is_conservative]:
        print(fn.__name__)
        fn()
    print("\nall decay planner tests passed")
