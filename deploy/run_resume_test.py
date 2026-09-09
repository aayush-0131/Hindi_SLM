#!/usr/bin/env python3
"""
Validates Requirement 7.7's checkpoint-and-continue mechanism BEFORE the
multi-day gap between GPU windows, not after it.

Requirement 7.7 (checkpoint at window close, continue from that checkpoint when
the next window opens) is load-bearing for the entire second GPU window, and it
has never been executed. If nanochat's checkpoint does not round-trip Muon's
optimizer state, the step counter, or the dataloader position, that is
discoverable in ~15 minutes now and catastrophic to discover at the start of
window 2.

Verified against nanochat source, so this test knows what "working" means:
  - save_checkpoint() writes model_{step}.pt, optim_{step}_rank{r}.pt, meta_{step}.json
  - load_checkpoint(..., load_optimizer=True) restores model AND optimizer
  - meta_data carries "dataloader_state_dict", and base_train.py feeds it back
    into tokenizing_distributed_data_loader_with_state_bos_bestfit(
        resume_state_dict=...), so data position resumes too rather than
    restarting the corpus from shard 0

Three failure modes this distinguishes:
  1. Resume crashes            -> the flag/artifact contract is broken
  2. Resume runs but loss spikes -> optimizer state was NOT restored (the
     dangerous silent case: it looks like it worked)
  3. Resume runs but data restarts -> dataloader state was NOT restored, so
     window 2 would re-train window 1's tokens
"""

import argparse
import json
import os
import re
import subprocess
import sys

from training import config as cfg

STEP_RE = re.compile(
    r"step\s+(\d+)/\s*\d+.*?loss:\s*([\d.]+).*?dt:\s*([\d.]+)ms.*?"
    r"bf16_mfu:\s*([\d.]+).*?epoch:\s*(\d+)\s+pq:\s*(\d+)\s+rg:\s*(\d+)"
)

# Loss must not jump by more than this fraction across the resume boundary.
# A cold optimizer typically produces a jump far larger than this, so the
# threshold does not need to be tight to be diagnostic.
LOSS_SPIKE_TOLERANCE = 0.15


def run(nanochat_dir, extra_args, label):
    # -u is load-bearing: base_train's stdout is a PIPE here, so Python
    # block-buffers it. If the child dies without flushing (a C-level abort,
    # os._exit, or a CUDA abort), the buffered tail -- including the
    # traceback -- is lost, and the failure looks like "exited 1, no output".
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"] + extra_args
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}\n+ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=nanochat_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=os.environ.copy())
    steps = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if m := STEP_RE.search(line):
            steps.append({
                "step": int(m.group(1)), "loss": float(m.group(2)),
                "dt_ms": float(m.group(3)), "mfu": float(m.group(4)),
                "epoch": int(m.group(5)), "pq_idx": int(m.group(6)),
                "rg_idx": int(m.group(7)),
            })
    return proc.wait(), steps


def phase_args(total_steps, save_every, resume_from=None):
    """Same architecture flags as the real run -- a resume test on a different
    model shape would not exercise the same checkpoint contract."""
    args = [
        f"--depth={cfg.DEPTH}", f"--aspect-ratio={cfg.ASPECT_RATIO}",
        f"--head-dim={cfg.HEAD_DIM}", f"--window-pattern={cfg.WINDOW_PATTERN}",
        f"--max-seq-len={cfg.MAX_SEQ_LEN}",
        f"--total-batch-size={cfg.TOTAL_BATCH_TOKENS}",
        f"--device-batch-size={cfg.DEVICE_BATCH_SIZE}",
        f"--num-iterations={total_steps}",
        f"--warmdown-ratio={cfg.WARMDOWN_RATIO_STABLE}",
        f"--warmup-steps={cfg.WARMUP_STEPS}",
        f"--save-every={save_every}",
        "--eval-every=-1", "--sample-every=-1", "--core-metric-every=-1",
        "--model-tag=resume_test",
    ]
    if resume_from is not None:
        args.append(f"--resume-from-step={resume_from}")
    return args


def check_artifacts(base_dir, step):
    ckpt = os.path.join(base_dir, "base_checkpoints", "resume_test")
    expected = [f"model_{step:06d}.pt", f"optim_{step:06d}_rank0.pt"]
    found, missing = [], []
    for name in expected:
        (found if os.path.exists(os.path.join(ckpt, name)) else missing).append(name)
    meta = [f for f in os.listdir(ckpt) if f.startswith("meta_")] \
        if os.path.isdir(ckpt) else []
    return {"checkpoint_dir": ckpt, "found": found, "missing": missing,
            "meta_files": meta}


def _find_nanochat(explicit=None):
    """Locate the nanochat repo without assuming the caller's cwd.

    The default used to be the bare relative path "../nanochat", which is only
    correct when invoked from inside deploy/. Run one directory up -- the
    natural place, since that is where hindi_env.sh lives -- and it resolved to
    a sibling of workspaces/ and failed with "nanochat dir not found".
    Checked in order: explicit flag, $NANOCHAT_DIR, the parent of this file's
    directory, then $TEAM.
    """
    import os
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("NANOCHAT_DIR"):
        candidates.append(os.environ["NANOCHAT_DIR"])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), "nanochat"))
    if os.environ.get("TEAM"):
        candidates.append(os.path.join(os.environ["TEAM"], "nanochat"))
    candidates.append("../nanochat")
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "nanochat")):
            return os.path.abspath(c)
    return None


def main():
    ap = argparse.ArgumentParser(description="Requirement 7.7 resume smoke test")
    ap.add_argument("--nanochat-dir", default=None)
    ap.add_argument("--save-at", type=int, default=20,
                    help="checkpoint interval; the test resumes from this step")
    ap.add_argument("--total-steps", type=int, default=40,
                    help="run 1 stops here; run 2 resumes from --save-at to here")
    ap.add_argument("--report", default="resume_test_report.json")
    args = ap.parse_args()

    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        print("FAILED: NANOCHAT_BASE_DIR is not set -- checkpoints would go to "
              "~/.cache and this test would not exercise the persistent disk.",
              file=sys.stderr)
        return 1

    result = {"save_at": args.save_at, "total_steps": args.total_steps}

    args.nanochat_dir = _find_nanochat(args.nanochat_dir)
    if not args.nanochat_dir:
        print("RESUME TEST FAILED: could not locate the nanochat repo. Pass "
              "--nanochat-dir explicitly.", file=sys.stderr)
        return 1
    print(f"nanochat repo: {args.nanochat_dir}")

    rc1, steps1 = run(args.nanochat_dir,
                      phase_args(args.total_steps, args.save_at),
                      f"RUN 1: train 0 -> {args.total_steps}, "
                      f"checkpointing every {args.save_at}")
    result["run1_exit_code"] = rc1
    result["run1_steps_logged"] = len(steps1)
    if rc1 != 0:
        result["passed"] = False
        result["reason"] = f"Run 1 exited {rc1}; cannot test resume."
        return finish(result, args.report)

    result["artifacts"] = check_artifacts(base_dir, args.save_at)
    if result["artifacts"]["missing"]:
        result["passed"] = False
        result["reason"] = (
            f"Checkpoint artifacts missing at step {args.save_at}: "
            f"{result['artifacts']['missing']}. Without an optim_*.pt shard, a "
            f"resume restores weights but NOT the Muon optimizer state -- the "
            f"silent-loss-spike failure mode."
        )
        return finish(result, args.report)

    at_ckpt = next((s for s in steps1 if s["step"] == args.save_at), None)
    result["loss_at_checkpoint"] = at_ckpt["loss"] if at_ckpt else None
    result["dataloader_at_checkpoint"] = (
        {k: at_ckpt[k] for k in ("epoch", "pq_idx", "rg_idx")} if at_ckpt else None
    )

    rc2, steps2 = run(args.nanochat_dir,
                      phase_args(args.total_steps, args.save_at,
                                 resume_from=args.save_at),
                      f"RUN 2: resume from step {args.save_at}")
    result["run2_exit_code"] = rc2
    result["run2_steps_logged"] = len(steps2)
    if rc2 != 0:
        result["passed"] = False
        result["reason"] = (
            f"Resume exited {rc2}. Requirement 7.7 is NOT satisfiable as "
            f"configured -- fix this before spending window-1 GPU hours on a run "
            f"you cannot continue in window 2."
        )
        return finish(result, args.report)

    after = [s for s in steps2 if s["step"] > args.save_at]
    if not after:
        result["passed"] = False
        result["reason"] = "Resume logged no steps past the checkpoint step."
        return finish(result, args.report)

    first = after[0]
    result["loss_after_resume"] = first["loss"]
    result["dataloader_after_resume"] = {
        k: first[k] for k in ("epoch", "pq_idx", "rg_idx")
    }

    problems = []

    # (2) optimizer state actually restored?
    if at_ckpt:
        jump = (first["loss"] - at_ckpt["loss"]) / at_ckpt["loss"]
        result["loss_jump_fraction"] = round(jump, 4)
        if jump > LOSS_SPIKE_TOLERANCE:
            problems.append(
                f"Loss jumped {jump:+.1%} across the resume boundary "
                f"({at_ckpt['loss']:.4f} -> {first['loss']:.4f}), over the "
                f"{LOSS_SPIKE_TOLERANCE:.0%} tolerance. This is the signature of "
                f"optimizer state NOT being restored. Resuming a 22h run this way "
                f"would silently discard optimization progress."
            )

    # (3) dataloader position actually restored?
    if at_ckpt:
        before_pos = (at_ckpt["epoch"], at_ckpt["pq_idx"], at_ckpt["rg_idx"])
        after_pos = (first["epoch"], first["pq_idx"], first["rg_idx"])
        result["dataloader_advanced"] = after_pos >= before_pos
        if after_pos < before_pos:
            problems.append(
                f"Dataloader position went backwards on resume: {before_pos} -> "
                f"{after_pos}. Window 2 would re-train window 1's tokens instead "
                f"of continuing through the corpus."
            )

    result["passed"] = not problems
    if problems:
        result["reason"] = " | ".join(problems)
    return finish(result, args.report)


def finish(result, report_path):
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\n" + "=" * 72)
    print(f"RESUME TEST {'PASSED' if result.get('passed') else 'FAILED'}")
    print("=" * 72)
    for key in ("loss_at_checkpoint", "loss_after_resume", "loss_jump_fraction",
                "dataloader_at_checkpoint", "dataloader_after_resume",
                "dataloader_advanced"):
        if key in result and result[key] is not None:
            print(f"  {key}: {result[key]}")
    if "reason" in result:
        print(f"\n  {result['reason']}")
    if result.get("passed"):
        print("\n  Requirement 7.7's mechanism is verified: weights, optimizer "
              "state, and data position all survive a stop/resume cycle.")
    print(f"\n  report written to {report_path}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
