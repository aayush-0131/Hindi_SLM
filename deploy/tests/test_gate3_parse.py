"""Parser tests for run_gate3.py against base_train.py's REAL print formats.

Line formats copied verbatim from nanochat/scripts/base_train.py:
  print0(f"GPU: {name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
  print0(f"Parameter counts:") / print0(f"{key:24s}: {value:,}")
  print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
  print0(f"step {step:05d}/{num_iterations:05d} ({pct:.2f}%) | loss: {l:.6f} | "
         f"lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {t:,} | "
         f"bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {m:.2f}m{eta}")
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_gate3 as g

FAKE = """GPU: NVIDIA H200 | Peak FLOPS (BF16): 9.89e+14
COMPUTE_DTYPE: torch.bfloat16 (bf16 default)
✓ Using Flash Attention 3: efficient, new and awesome.
Vocab size: 32,768
Parameter counts:
embedding               : 37,748,736
transformer_matrices    : 382,205,952
lm_head                 : 37,748,736
total                   : 419,954,688
Estimated FLOPs per token: 2.519728e+09
Using user-provided number of iterations: 60
step 00001/00060 (1.67%) | loss: 10.412345 | lrm: 0.03 | dt: 21050.10ms | tok/sec: 24,907 | bf16_mfu: 6.31 | epoch: 0 pq: 0 rg: 0 | total time: 0.35m
step 00019/00060 (31.67%) | loss: 8.112345 | lrm: 0.48 | dt: 4200.00ms | tok/sec: 124,830 | bf16_mfu: 31.80 | epoch: 0 pq: 1 rg: 0 | total time: 1.80m
step 00025/00060 (41.67%) | loss: 7.523456 | lrm: 1.00 | dt: 3550.20ms | tok/sec: 147,679 | bf16_mfu: 38.42 | epoch: 0 pq: 3 rg: 1 | total time: 2.15m
step 00040/00060 (66.67%) | loss: 7.001234 | lrm: 1.00 | dt: 3480.00ms | tok/sec: 150,657 | bf16_mfu: 39.18 | epoch: 0 pq: 5 rg: 2 | total time: 3.05m
step 00060/00060 (100.00%) | loss: 6.712345 | lrm: 0.05 | dt: 3510.00ms | tok/sec: 149,369 | bf16_mfu: 38.85 | epoch: 0 pq: 8 rg: 0 | total time: 4.20m
"""


def parse(text):
    steps, params, flops, peak = [], {}, None, None
    in_block = False
    for line in text.splitlines(keepends=True):
        if m := g.STEP_RE.search(line):
            steps.append({"step": int(m.group(1)), "dt_ms": float(m.group(2)),
                          "mfu": float(m.group(3))}); continue
        if m := g.FLOPS_RE.search(line):
            flops = float(m.group(1)); in_block = False; continue
        if m := g.PEAK_RE.search(line):
            peak = float(m.group(1)); continue
        if line.strip().startswith("Parameter counts:"):
            in_block = True; continue
        if in_block:
            if m := g.PARAM_RE.match(line.strip()):
                params[m.group(1).strip()] = int(m.group(2).replace(",", ""))
            else:
                in_block = False
    return steps, params, flops, peak


def test_parsing():
    steps, params, flops, peak = parse(FAKE)
    assert len(steps) == 5, steps
    assert steps[0]["step"] == 1 and steps[0]["mfu"] == 6.31
    assert steps[2]["dt_ms"] == 3550.20 and steps[2]["mfu"] == 38.42
    assert params["total"] == 419_954_688, params
    assert params["transformer_matrices"] == 382_205_952
    assert "Estimated FLOPs per token" not in params, "flops line leaked into param block"
    assert flops == 2.519728e9
    assert peak == 9.89e14
    print("  parsing:              OK")


def test_pass_discards_warmup():
    steps, params, flops, peak = parse(FAKE)
    r = g.evaluate(steps, params, flops, peak)
    # steps 1 and 19 are <= WARMUP_STEPS_DISCARDED(20) and must be excluded;
    # including step 1's 6.31% MFU would drag the mean under the 35% bar.
    assert r["steps_measured"] == 3, r
    assert r["steps_discarded_as_warmup"] == 2, r
    assert abs(r["measured_mfu_pct"] - (38.42 + 39.18 + 38.85) / 3) < 0.01, r
    assert r["passed"] is True, r
    import training.config as C
    # Derived from config, and from TOTAL_STEP_EQUIVALENTS rather than
    # TOTAL_STEPS -- two decay branches are paid for in wall-clock but counted
    # once in a single trajectory. This assertion used to read `is True`, which
    # was only correct while TOTAL_STEPS was the stale 28,000 (Bug 14).
    expected_fits = (r["mean_sec_per_step"] * C.TOTAL_STEP_EQUIVALENTS
                     <= C.SPEC_GPU_HOUR_CEILING * 3600)
    assert r["fits_gpu_hour_ceiling"] is expected_fits, (
        r, C.TOTAL_STEP_EQUIVALENTS)
    assert r["gpu_hour_ceiling"] == C.SPEC_GPU_HOUR_CEILING
    print(f"  warmup discard:       OK (MFU {r['measured_mfu_pct']}%, "
          f"{r['projected_total_hours']}h projected over "
          f"{C.TOTAL_STEP_EQUIVALENTS:,} step-equivalents, "
          f"fits={r['fits_gpu_hour_ceiling']})")


def test_requirement_1_4_recompute():
    steps, params, flops, peak = parse(FAKE)
    r = g.evaluate(steps, params, flops, peak)
    rec = r["requirement_1_4_recompute"]
    assert rec["realized_params"] == 419_954_688
    assert rec["delta_pct"] == 5.0, rec          # +5% over the spec's 400M
    # Sourced from config rather than the literal 28,000 this used to assert.
    # TOTAL_STEPS was corrected 28,000 -> 30,500 (the live run's value) and
    # this test was the thing that caught the mismatch.
    import training.config as C
    expected_d = C.TOTAL_STEPS * C.TOTAL_BATCH_TOKENS
    assert rec["D_tokens_at_current_step_budget"] == expected_d, (
        rec["D_tokens_at_current_step_budget"], expected_d)
    assert abs(rec["C_6ND"] - 6 * 419_954_688 * expected_d) < 1e6
    assert 30 < rec["tokens_per_param"] < 40, rec
    print(f"  Req 1.4 recompute:    OK (N={rec['realized_params']:,}, "
          f"C={rec['C_6ND']:.2e}, {rec['tokens_per_param']} tok/param)")


def test_low_mfu_fails():
    bad = FAKE.replace("bf16_mfu: 38.42", "bf16_mfu: 24.10") \
              .replace("bf16_mfu: 39.18", "bf16_mfu: 23.90") \
              .replace("bf16_mfu: 38.85", "bf16_mfu: 24.00")
    steps, params, flops, peak = parse(bad)
    r = g.evaluate(steps, params, flops, peak)
    assert r["passed"] is False, r
    assert "below the 35.0% bar" in r["reason"], r["reason"]
    print("  sub-threshold fails:  OK")


def test_no_usable_steps_fails_closed():
    # A crash right after startup must FAIL, not pass on an empty average.
    steps, params, flops, peak = parse(FAKE.split("step 00001")[0])
    r = g.evaluate(steps, params, flops, peak)
    assert r["passed"] is False, r
    assert "did not get far enough" in r["reason"], r["reason"]
    print("  fails closed on crash: OK")


def test_red_flag_on_slow_steps():
    slow = FAKE.replace("dt: 3550.20ms", "dt: 5200.00ms") \
               .replace("dt: 3480.00ms", "dt: 5300.00ms") \
               .replace("dt: 3510.00ms", "dt: 5250.00ms")
    steps, params, flops, peak = parse(slow)
    r = g.evaluate(steps, params, flops, peak)
    assert "red_flag" in r, r
    assert r["fits_gpu_hour_ceiling"] is False, r
    print(f"  red flag on slow step: OK ({r['projected_total_hours']}h projected)")


if __name__ == "__main__":
    for fn in [test_parsing, test_pass_discards_warmup, test_requirement_1_4_recompute,
               test_low_mfu_fails, test_no_usable_steps_fails_closed,
               test_red_flag_on_slow_steps]:
        fn()
    print("\nall gate3 parser tests passed")
