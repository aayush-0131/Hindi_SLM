#!/usr/bin/env python3
"""
Gate 3 -- training throughput validation (Requirement 6).

Runs a short nanochat shakedown at the real vocab size and depth, reads the MFU
nanochat already computes per step, and applies Requirement 6.3's >=35% bar.

Two things this does beyond just reading a number off the screen:

  1. Requirement 6.2 describes computing MFU by hand as
     total_training_flops / total_training_time / 989e12. base_train.py already
     does exactly this every step and prints it as `bf16_mfu`, using
     get_peak_flops(device_name) as the divisor. We parse that rather than
     recomputing it, so the gate measures the same quantity the training run
     will report for the next 28 hours.

  2. Requirement 1.4 requires the compute estimate be recomputed and documented
     whenever the realized parameter count differs from the spec's ~400M. It
     does differ here -- nanochat has no GQA and a 4x ReLU^2 MLP rather than
     SwiGLU-3072 (see training/config.py ARCH_DEVIATIONS) -- so we read the
     realized count off the shakedown and redo C=6ND against the real N.

Early steps are discarded: torch.compile and LR warmup make the first steps
unrepresentatively slow, and averaging them in would fail a run that is
actually fine.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from training import config as cfg

STEP_RE = re.compile(r"step\s+(\d+)/\s*\d+.*?dt:\s*([\d.]+)ms.*?bf16_mfu:\s*([\d.]+)")
FLOPS_RE = re.compile(r"Estimated FLOPs per token:\s*([\d.eE+-]+)")
PARAM_RE = re.compile(r"^(\w[\w ]*?)\s*:\s*([\d,]+)\s*$")
PEAK_RE = re.compile(r"Peak FLOPS \(BF16\):\s*([\d.eE+-]+)")

WARMUP_STEPS_DISCARDED = 20


def run_shakedown(nanochat_dir, extra_args):
    cmd = [sys.executable, "-m", "scripts.base_train"] + cfg.gate3_args() + extra_args
    print("+ " + " ".join(cmd) + f"\n  (cwd={nanochat_dir})\n", flush=True)

    proc = subprocess.Popen(
        cmd, cwd=nanochat_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=os.environ.copy(),
    )
    steps, params, flops_per_token, peak_flops = [], {}, None, None
    in_param_block = False

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        if m := STEP_RE.search(line):
            steps.append({"step": int(m.group(1)),
                          "dt_ms": float(m.group(2)),
                          "mfu": float(m.group(3))})
            continue
        if m := FLOPS_RE.search(line):
            flops_per_token = float(m.group(1))
            in_param_block = False
            continue
        if m := PEAK_RE.search(line):
            peak_flops = float(m.group(1))
            continue
        if line.strip().startswith("Parameter counts:"):
            in_param_block = True
            continue
        if in_param_block:
            if m := PARAM_RE.match(line.strip()):
                params[m.group(1).strip()] = int(m.group(2).replace(",", ""))
            else:
                in_param_block = False

    rc = proc.wait()
    return rc, steps, params, flops_per_token, peak_flops


def evaluate(steps, params, flops_per_token, peak_flops):
    """Apply Requirement 6.3's bar and Requirement 1.4's recompute."""
    usable = [s for s in steps if s["step"] > WARMUP_STEPS_DISCARDED]
    if not usable:
        return {
            "passed": False,
            "reason": f"No steps past step {WARMUP_STEPS_DISCARDED} were logged "
                      f"({len(steps)} step lines seen). The shakedown did not get far "
                      f"enough to measure steady-state throughput.",
        }

    mean_mfu = sum(s["mfu"] for s in usable) / len(usable)
    mean_dt = sum(s["dt_ms"] for s in usable) / len(usable) / 1000.0
    passed = mean_mfu >= cfg.MFU_PASS_THRESHOLD

    result = {
        "passed": passed,
        "measured_mfu_pct": round(mean_mfu, 2),
        "threshold_pct": cfg.MFU_PASS_THRESHOLD,
        "mean_sec_per_step": round(mean_dt, 3),
        "sec_per_step_budget": cfg.SEC_PER_STEP_BUDGET,
        "steps_measured": len(usable),
        "steps_discarded_as_warmup": len(steps) - len(usable),
        "gpu_peak_flops": peak_flops,
        "flops_per_token": flops_per_token,
        "param_counts": params,
    }
    if not passed:
        result["reason"] = (
            f"Measured MFU {mean_mfu:.2f}% is below the {cfg.MFU_PASS_THRESHOLD}% bar "
            f"(Requirement 6.3/6.4). Do NOT begin the pretraining commitment. Check, "
            f"in order: whether torch.compile actually succeeded, whether Flash "
            f"Attention 3 loaded (the SDPA fallback is much slower), the gradient "
            f"accumulation count implied by --device-batch-size, and whether another "
            f"tenant is on this GPU index."
        )

    # --- wall-clock feasibility, which the MFU bar alone does not tell you
    stable_h = cfg.STABLE_STEPS * mean_dt / 3600
    total_h = cfg.TOTAL_STEPS * mean_dt / 3600
    result["projected_stable_hours"] = round(stable_h, 2)
    result["projected_total_hours"] = round(total_h, 2)
    result["fits_28h_budget"] = total_h <= 28.0
    if mean_dt > cfg.SEC_PER_STEP_RED_FLAG:
        result["red_flag"] = (
            f"{mean_dt:.2f}s/step exceeds the spec's {cfg.SEC_PER_STEP_RED_FLAG}s "
            f"sub-30%-MFU red flag."
        )

    # --- Requirement 1.4: recompute C = 6ND against the REALIZED parameter count
    realized_n = params.get("total")
    if realized_n:
        d_total = cfg.TOTAL_STEPS * cfg.TOTAL_BATCH_TOKENS
        result["requirement_1_4_recompute"] = {
            "spec_param_target": cfg.SPEC_PARAM_TARGET,
            "realized_params": realized_n,
            "delta_pct": round(100 * (realized_n - cfg.SPEC_PARAM_TARGET)
                               / cfg.SPEC_PARAM_TARGET, 1),
            "D_tokens_at_current_step_budget": d_total,
            "C_6ND": 6 * realized_n * d_total,
            "tokens_per_param": round(d_total / realized_n, 1),
            "note": "The spec's recomputed target was C~=3.53e19 at N=400M. If C_6ND "
                    "exceeds that materially, the honest options are to cut "
                    "TOTAL_STEPS or to accept a higher realized compute -- but the "
                    "binding constraint is GPU-hours (projected_total_hours), not "
                    "C, so prefer the wall-clock number for scheduling.",
        }
    return result


def main():
    ap = argparse.ArgumentParser(description="Gate 3: MFU shakedown (Requirement 6)")
    ap.add_argument("--nanochat-dir", default="../nanochat",
                    help="path to the nanochat repo (relative paths are safest on "
                         "this host -- $HOME resolves differently between the "
                         "team02 and usr1-iairo identities)")
    ap.add_argument("--report", default="gate3_report.json")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra flags passed through to base_train")
    args = ap.parse_args()

    if not os.path.isdir(args.nanochat_dir):
        print(f"GATE 3 FAILED: nanochat dir not found: {args.nanochat_dir}",
              file=sys.stderr)
        return 1

    if not args.skip_preflight:
        from training.preflight import main as preflight_main, PreflightError
        try:
            preflight_main()
        except PreflightError as e:
            print(f"\nGATE 3 FAILED (preflight): {e}", file=sys.stderr)
            return 1
        print("\n" + "=" * 72 + "\nStarting shakedown\n" + "=" * 72)

    rc, steps, params, flops, peak = run_shakedown(args.nanochat_dir, args.extra)
    result = evaluate(steps, params, flops, peak)
    result["base_train_exit_code"] = rc
    if rc != 0 and result.get("passed"):
        result["passed"] = False
        result["reason"] = f"base_train exited non-zero ({rc}) despite acceptable MFU."

    with open(args.report, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print(f"GATE 3 {'PASSED' if result['passed'] else 'FAILED'}")
    print("=" * 72)
    if "measured_mfu_pct" in result:
        print(f"  MFU:               {result['measured_mfu_pct']}% "
              f"(bar: {result['threshold_pct']}%)")
        print(f"  step time:         {result['mean_sec_per_step']}s "
              f"(budget: {result['sec_per_step_budget']}s)")
        print(f"  projected stable:  {result['projected_stable_hours']}h "
              f"(Requirement 7.1 allows 22.4h)")
        print(f"  projected total:   {result['projected_total_hours']}h "
              f"(ceiling 28h) -> {'FITS' if result['fits_28h_budget'] else 'OVER BUDGET'}")
    if rec := result.get("requirement_1_4_recompute"):
        print(f"  realized params:   {rec['realized_params']:,} "
              f"({rec['delta_pct']:+}% vs the spec's ~400M)")
        print(f"  tokens/param:      {rec['tokens_per_param']}")
    for key in ("red_flag", "reason"):
        if key in result:
            print(f"  {key}: {result[key]}")
    print(f"\n  report written to {args.report}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
