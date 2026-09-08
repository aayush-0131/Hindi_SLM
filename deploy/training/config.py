"""
Derived training configuration for Requirements 1, 6, 7 -- the first code for
these requirements (no design.md pass exists for them yet).

Every value here was checked against nanochat's ACTUAL scripts/base_train.py
argparse surface and nanochat/gpt.py, not inferred from the build spec. Where
the spec and nanochat disagree, the deviation is recorded in ARCH_DEVIATIONS
rather than silently papered over -- Requirement 1.4 requires realized-vs-spec
differences to be documented.
"""

# ---------------------------------------------------------------- architecture
# nanochat computes: base_dim = depth * aspect_ratio, then rounds model_dim UP
# to a multiple of head_dim, then num_heads = model_dim // head_dim.
#   24 * 48 = 1152; 1152 is already a multiple of 64; 1152 // 64 = 18 heads.
DEPTH = 24            # Requirement 1.1: 24 layers
ASPECT_RATIO = 48     # -> model_dim 1152 (nanochat's default 64 would give 1536)
HEAD_DIM = 64         # -> 18 heads. See ARCH_DEVIATIONS for why not 72/16 heads.
WINDOW_PATTERN = "L"  # full attention. nanochat defaults to "SSSL" (sliding window).
MAX_SEQ_LEN = 2048    # Requirement 1.2
VOCAB_SIZE = 32768    # Requirement 1.3 -- asserted against the Gate-1 artifact in preflight

ARCH_DEVIATIONS = [
    # (spec says, nanochat does, why we accept it)
    ("16 query heads / head_dim 72",
     "18 query heads / head_dim 64",
     "head_dim=72 is not a standard Flash-Attention-3 head dimension; a non-supported "
     "head_dim silently drops to the SDPA fallback, which per base_train.py's own "
     "startup warning makes utilization 'terrible' and would fail Gate 3. Parameter "
     "count is unaffected (attention params depend on model_dim, not head count, "
     "while n_kv_head == n_head)."),
    ("4 KV heads (grouped-query attention)",
     "n_kv_head == n_head (no GQA)",
     "base_train.py's build_model_meta() hardcodes n_kv_head=num_heads; GQA is not "
     "reachable from the CLI. Reaching it needs a source edit to nanochat. NOT taken: "
     "an untested architecture patch is the wrong risk to add to a 22h run. Raises the "
     "parameter count (k/v projections stay full-width)."),
    ("SwiGLU feed-forward, dim 3072",
     "ReLU-squared feed-forward, dim 4*model_dim = 4608",
     "gpt.py's MLP is c_fc(n_embd -> 4*n_embd) / ReLU().square() / c_proj, with no "
     "SwiGLU variant and no width flag. Also raises the parameter count."),
    ("full attention (RoPE only, no sliding window)",
     "window_pattern default 'SSSL'",
     "Overridden to 'L' to match the spec. Also required for sane utilization when "
     "FA3 is unavailable."),
]

# The two deviations above that ADD parameters mean the realized N will land
# above the spec's ~400M. base_train.py prints the true parameter count and
# FLOPs/token at startup -- Requirement 1.4's recompute is done by reading
# those two lines off the Gate 3 shakedown, not by trusting an estimate here.
SPEC_PARAM_TARGET = 400_000_000

# ---------------------------------------------------------------- token budget
# research.md's windowed-GPU-access recompute: C_new ~= 3.53e19 FLOPs,
# D_new ~= 14.7B total at N = 400M, split 80/20 stable/decay.
TOTAL_BATCH_TOKENS = 524_288   # Requirement 1.2 (2^19)
DEVICE_BATCH_SIZE = 32         # spec's micro-batch; drop to 16/8 only on OOM

D_TOTAL_TOKENS = 14_700_000_000
TOTAL_STEPS = round(D_TOTAL_TOKENS / TOTAL_BATCH_TOKENS / 100) * 100   # 28,000
DECAY_FRACTION = 0.20                                                  # Requirement 7.5
STABLE_STEPS = round(TOTAL_STEPS * (1 - DECAY_FRACTION))               # 22,400
DECAY_STEPS = TOTAL_STEPS - STABLE_STEPS                               # 5,600 total
DECAY_STEPS_PER_VARIANT = DECAY_STEPS // 2                             # 2,800 each

# ------------------------------------------------------------------- schedules
# nanochat's get_lr_multiplier(it) is a pure function of
# (it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac) -- all
# CLI-settable. That is what makes the stable-then-two-decay-variants plan
# expressible without touching nanochat: see decay_args() below.
WARMUP_STEPS = 40          # nanochat's tuned default
FINAL_LR_FRAC = 0.05       # nanochat's tuned default
WARMDOWN_RATIO_STABLE = DECAY_FRACTION   # 0.20 (nanochat's default is 0.65!)

# ------------------------------------------------------------- checkpoint cadence
# Requirement 6.5: persistent-disk checkpoints at intervals <= 30 minutes.
# At the ~3.6 s/step this budget implies, 500 steps == exactly 30 min, i.e. right
# ON the limit. 400 steps (~24 min) leaves margin for step-time drift.
SAVE_EVERY = 400

# ------------------------------------------------------------------ evaluation
# base_train.py's defaults are a real throughput trap: --eval-every=250 with
# --eval-tokens=80*524288 (42M tokens) re-evaluates every ~15 min of training.
# Cut both; val bpb is a trend line here, not a precision measurement.
EVAL_EVERY = 500
EVAL_TOKENS = 10 * TOTAL_BATCH_TOKENS   # 5.24M, ~8x cheaper than the default

# nanochat's CORE metric is its own English task suite. Running it against a
# Hindi-only model measures noise and costs real minutes. Requirement 9.1's
# "reduced FineTasks subset every ~2B tokens" is satisfied by OUR eval_harness
# against saved checkpoints instead, out-of-band.
CORE_METRIC_EVERY = -1

# Requirement 9.3: hand-sample generations at 25 / 50 / 100% of progress.
SAMPLE_EVERY = STABLE_STEPS // 4   # 5,600 -> lands on 25/50/75/100% of stable

# ------------------------------------------------------------------- Gate 3
MFU_PASS_THRESHOLD = 35.0     # Requirement 6.3, percent
GATE3_STEPS = 60              # enough to clear warmup+compile and average cleanly
SEC_PER_STEP_BUDGET = 3.6     # TOTAL_STEPS in 28 GPU-hours
SEC_PER_STEP_RED_FLAG = 4.5   # spec's own sub-30%-MFU red flag


def _common_args():
    return [
        f"--depth={DEPTH}",
        f"--aspect-ratio={ASPECT_RATIO}",
        f"--head-dim={HEAD_DIM}",
        f"--window-pattern={WINDOW_PATTERN}",
        f"--max-seq-len={MAX_SEQ_LEN}",
        f"--total-batch-size={TOTAL_BATCH_TOKENS}",
        f"--device-batch-size={DEVICE_BATCH_SIZE}",
        f"--warmup-steps={WARMUP_STEPS}",
        f"--final-lr-frac={FINAL_LR_FRAC}",
        f"--core-metric-every={CORE_METRIC_EVERY}",
    ]


def gate3_args():
    """Requirement 6.1: a short shakedown at the real vocab size and depth."""
    return _common_args() + [
        f"--num-iterations={GATE3_STEPS}",
        f"--warmdown-ratio={WARMDOWN_RATIO_STABLE}",
        "--eval-every=-1",        # no eval noise in a throughput measurement
        "--sample-every=-1",
        "--save-every=-1",
        "--model-tag=gate3_shakedown",
    ]


def stable_args(resume_from_step=None):
    """
    Requirement 7.1: the stable phase.

    num_iterations is TOTAL_STEPS (not STABLE_STEPS) with warmdown_ratio 0.20, so
    the LR is flat across every stable step and the warmdown begins exactly at
    STABLE_STEPS. The run is simply stopped at STABLE_STEPS; the decay variants
    then pick that checkpoint up. Setting num_iterations=STABLE_STEPS instead
    would decay the LR *inside* the stable phase and there would be no flat
    checkpoint to branch the two variants from.
    """
    args = _common_args() + [
        f"--num-iterations={TOTAL_STEPS}",
        f"--warmdown-ratio={WARMDOWN_RATIO_STABLE}",
        f"--save-every={SAVE_EVERY}",
        f"--eval-every={EVAL_EVERY}",
        f"--eval-tokens={EVAL_TOKENS}",
        f"--sample-every={SAMPLE_EVERY}",
        "--model-tag=stable",
    ]
    if resume_from_step is not None:
        args.append(f"--resume-from-step={resume_from_step}")   # Requirement 7.7
    return args


def decay_args(variant):
    """
    Requirements 7.3 / 7.4: Decay A and Decay B, both resuming the SAME
    stable checkpoint at STABLE_STEPS.

    num_iterations = STABLE_STEPS + DECAY_STEPS_PER_VARIANT and warmdown_ratio =
    DECAY_STEPS_PER_VARIANT / num_iterations puts warmdown_start at exactly
    STABLE_STEPS, so the LR decays across precisely this variant's steps.

    Caller must first copy the stable step-STABLE_STEPS checkpoint into this
    variant's checkpoint dir -- nanochat resolves checkpoint_dir from
    --model-tag, so a differently-tagged run will not see the stable output.
    """
    assert variant in ("a", "b"), variant
    n_iters = STABLE_STEPS + DECAY_STEPS_PER_VARIANT
    return _common_args() + [
        f"--num-iterations={n_iters}",
        f"--warmdown-ratio={DECAY_STEPS_PER_VARIANT / n_iters:.6f}",
        f"--resume-from-step={STABLE_STEPS}",
        f"--save-every={SAVE_EVERY}",
        f"--eval-every={EVAL_EVERY}",
        f"--eval-tokens={EVAL_TOKENS}",
        f"--sample-every={SAMPLE_EVERY}",
        f"--model-tag=decay_{variant}",
    ]
