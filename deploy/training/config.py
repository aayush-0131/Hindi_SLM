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
#
# PLAN C, 2026-09-06 (project owner's decision). Two prior revisions of this
# block are worth knowing about so the numbers are not re-derived wrongly a
# third time:
#
#  1. It carried TOTAL_STEPS = 28_000, a planning estimate made BEFORE Gate 3
#     measured the real step time. The launched run used 30,500.
#  2. It then capped pretraining at Requirement 7.6's 28 GPU-hours. That
#     figure turned out to be STALE, not a hardware limit -- research.md
#     derived it mid-Round-1 as "~34h remaining minus ~6h for prep", where the
#     ~34h was measured from a point already inside window 1 and so excluded
#     the ~6h of window 1 already spent. The actual allocation is two 20-hour
#     windows = 40 h, exactly the ORIGINAL ceiling in requirements.md line 12,
#     and the 6 h prep reserve has since been spent (Gate 1, Gate 3, corpus,
#     pipeline all done). Requirement 7.6 is therefore re-derived here under
#     the mechanism Requirement 1.4 provides for exactly this.
#
# The binding constraint in Round 2 is the 20-hour WINDOW, not a GPU-hour cap:
# ~6 h of it is needed for Gate 2, decay-variant evaluation, SFT and the demo,
# leaving ~14 h for pretraining.
TOTAL_BATCH_TOKENS = 524_288   # Requirement 1.2 (2^19)
DEVICE_BATCH_SIZE = 32         # spec's micro-batch; drop to 16/8 only on OOM

# MEASURED, not assumed: Gate 3 re-run 2026-09-07 on torch 2.14.0+cu130 after
# the venv rebuild, giving 51.14% MFU / 3.319 s/step (Round 1 on torch 2.9.1
# was 52-53% / 3.20-3.25 s). The re-measurement mattered: the whole Plan C
# budget rests on this number, and the torch version changed underneath it.
MEASURED_SEC_PER_STEP = 3.319
# Round 1's recorded training time AT THE RESUME STEP (meta_022000.json
# loop_state.total_training_time). Was 67,961.98 for step 21,200.
ROUND1_GPU_SECONDS = 70_523.09
# Step 22,000, not 21,200. ROUND2_HANDOFF.md sec 6.1 concluded 21,200 because
# optim_022000_rank0.pt "NEVER TRANSFERRED" to the laptop -- but on 2026-09-06
# it was found intact ON THE SERVER (3,795,102,997 bytes) alongside
# model_022000.pt and meta_022000.json. It was never missing, only never
# downloaded, and the earlier "nothing survived the purge" reading was a
# team02-vs-usr1-iairo permissions artifact. Resuming here recovers 800 steps
# (~43 min) and starts from val bpb 0.34967 rather than 0.35047 -- the best
# weights Round 1 produced.
RESUME_FROM_STEP = 22_000
SPEC_GPU_HOUR_CEILING = 40.0       # requirements.md line 12, the real ceiling

# What Round 1 actually launched with. Kept because it determines the LR the
# model has ALREADY seen, which is what the safety assertion below checks.
ROUND1_NUM_ITERATIONS = 30_500
DECAY_FRACTION = 0.20              # Requirement 7.5's 20% decay phase

# --- Plan C: extend the stable phase onto a GROWN corpus --------------------
# Round 1's stable phase would have ended at 24,400 = 4.09 epochs over a 3.13B
# corpus -- AT the ~4-epoch repetition ceiling, i.e. repetition-limited rather
# than compute-limited. The corpus uses only 7.5% of the ~42B available in the
# seven already-chosen sources, so growing it is nearly free (add_fineweb2.py
# streams ~7,000 docs/sec). Extending stable onto fresh data is a more reliable
# gain than a longer decay, whose supply is dominated by never-inspected OCR'd
# Sangraha PDF.
# 26,800, NOT 27,000. --save-every=400 only checkpoints at multiples of 400,
# and 27,000 is not one (27000/400 = 67.5) -- the nearest are 26,800 and
# 27,200. The stable run was launched with --num-iterations=33750, which puts
# warmdown_start at exactly 27,000, so step 27,200 is already 200 steps INTO
# the anneal: branching a decay variant from it would give the model a
# decaying LR followed by a jump back to 1.0. 26,800 is a real checkpoint and
# is still fully in the flat-LR region. Cost of the 200 steps: 0.18 h.
STABLE_END_STEPS = 26_800
TARGET_CORPUS_TOKENS = 8_000_000_000   # grow to ~8B before resuming

# num_iterations chosen so warmdown_start lands exactly on STABLE_END_STEPS
# while keeping Requirement 7.5's 0.20 ratio.
# The LAUNCHED value, not derived from STABLE_END_STEPS. The run is already
# training with --num-iterations=33750; changing it now would alter the LR
# schedule mid-run. Warmdown therefore starts at 27,000 while we stop at
# 26,800 -- the run simply ends 200 steps before its anneal would begin,
# which is what leaves a flat checkpoint to branch from.
STABLE_NUM_ITERATIONS = 33_750

# --- decay variants ---------------------------------------------------------
# Sized to what the decay DATA actually supports at low upsampling, rather
# than to whatever fills the clock.
DECAY_STEPS_PER_VARIANT = 4_500
DECAY_STEPS = DECAY_STEPS_PER_VARIANT * 2
DECAY_NUM_ITERATIONS = STABLE_END_STEPS + DECAY_STEPS_PER_VARIANT      # 31,500
DECAY_WARMDOWN_RATIO = DECAY_STEPS_PER_VARIANT / DECAY_NUM_ITERATIONS

# Retained for the Requirement 1.4 recompute in run_gate3.py.
TOTAL_STEPS = STABLE_END_STEPS + DECAY_STEPS_PER_VARIANT   # one trajectory
STABLE_STEPS = STABLE_END_STEPS       # back-compat alias
D_TOTAL_TOKENS = TOTAL_STEPS * TOTAL_BATCH_TOKENS

# --- safety assertions ------------------------------------------------------
# THE critical one. Raising num_iterations is only safe because nanochat's
# get_lr_multiplier returns exactly 1.0 for every step between warmup and
# warmdown_start. Every step already trained (<= RESUME_FROM_STEP) sat in that
# flat region under Round 1's config and must STILL sit in it under the new
# one -- otherwise the completed trajectory would be retroactively
# inconsistent with the schedule and the model would end up mis-annealed.
_round1_warmdown_start = ROUND1_NUM_ITERATIONS * (1 - DECAY_FRACTION)   # 24,400
_new_warmdown_start = STABLE_NUM_ITERATIONS * (1 - DECAY_FRACTION)      # 27,000
assert RESUME_FROM_STEP < _round1_warmdown_start, (
    f"step {RESUME_FROM_STEP} was already inside Round 1's warmdown "
    f"({_round1_warmdown_start}); its LR was NOT flat and the schedule cannot "
    f"be extended without mis-annealing the model")
assert RESUME_FROM_STEP < _new_warmdown_start, (
    "resume step must remain in the flat-LR region under the new schedule")
assert STABLE_END_STEPS <= _new_warmdown_start, (
    f"stable must STOP at or before warmdown_start ({_new_warmdown_start:.0f}), "
    f"or the branch checkpoint is already partly annealed")
assert STABLE_END_STEPS % 400 == 0, (
    f"STABLE_END_STEPS={STABLE_END_STEPS} is not a multiple of --save-every=400, "
    f"so no checkpoint would exist there to branch the decay variants from")
assert abs(DECAY_NUM_ITERATIONS * (1 - DECAY_WARMDOWN_RATIO)
           - STABLE_END_STEPS) < 1.0, (
    "decay warmdown must start exactly at STABLE_END_STEPS, or the LR jumps "
    "at the branch point")

TOTAL_PRETRAIN_GPU_HOURS = (
    ROUND1_GPU_SECONDS
    + (STABLE_END_STEPS - RESUME_FROM_STEP) * MEASURED_SEC_PER_STEP
    + DECAY_STEPS * MEASURED_SEC_PER_STEP) / 3600
assert TOTAL_PRETRAIN_GPU_HOURS <= SPEC_GPU_HOUR_CEILING, (
    f"plan needs {TOTAL_PRETRAIN_GPU_HOURS:.2f} h against the spec's "
    f"{SPEC_GPU_HOUR_CEILING} h ceiling")

# Round 2 window accounting, so the plan is checkable against the real limit.
ROUND2_WINDOW_HOURS = 20.0
ROUND2_NON_PRETRAIN_RESERVE_HOURS = 6.0   # Gate 2, decay eval, SFT, demo
ROUND2_PRETRAIN_HOURS = (
    (STABLE_END_STEPS - RESUME_FROM_STEP) + DECAY_STEPS
) * MEASURED_SEC_PER_STEP / 3600
assert ROUND2_PRETRAIN_HOURS <= (
    ROUND2_WINDOW_HOURS - ROUND2_NON_PRETRAIN_RESERVE_HOURS), (
    f"Round 2 pretraining needs {ROUND2_PRETRAIN_HOURS:.2f} h but only "
    f"{ROUND2_WINDOW_HOURS - ROUND2_NON_PRETRAIN_RESERVE_HOURS:.2f} h of the "
    f"window is available after the eval/SFT/demo reserve")

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
# Derived, not hardcoded: the old literal 3.6 was computed against the stale
# TOTAL_STEPS = 28,000 and silently became wrong when the live run used 30,500.
#
# Divide by TOTAL_STEP_EQUIVALENTS, not TOTAL_STEPS. The project trains ONE
# stable trajectory but TWO decay branches, so the decay steps are paid for
# twice in wall-clock while appearing once in a single trajectory's step count.
# Using TOTAL_STEPS here yielded 4.571 s/step -- a "budget" looser than the
# spec's own 4.5 s/step sub-30%-MFU red flag, i.e. meaningless as a bar.
TOTAL_STEP_EQUIVALENTS = STABLE_END_STEPS + DECAY_STEPS      # 36,000
SEC_PER_STEP_BUDGET = round(
    SPEC_GPU_HOUR_CEILING * 3600 / TOTAL_STEP_EQUIVALENTS, 3)   # 4.0
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

    num_iterations is STABLE_NUM_ITERATIONS (33,750), NOT TOTAL_STEPS, with
    warmdown_ratio 0.20 -- so the LR stays flat across every stable step and
    the warmdown begins exactly at STABLE_END_STEPS (27,000). The run is simply
    stopped at STABLE_END_STEPS; both decay variants then branch from that
    checkpoint. Setting num_iterations=STABLE_END_STEPS instead would decay the
    LR *inside* the stable phase, leaving no flat checkpoint to branch from.

    Note this is a LARGER num_iterations than Round 1 launched with (30,500 ->
    33,750), which extends the stable phase from 24,400 to 27,000 steps. That
    is safe only because every already-trained step sits in the flat-LR region
    under both configs -- asserted at module import, not assumed. Confirm
    empirically on the resume test: base_train should print `lrm: 1.00` at the
    resumed step.
    """
    args = _common_args() + [
        f"--num-iterations={STABLE_NUM_ITERATIONS}",
        f"--warmdown-ratio={DECAY_FRACTION}",
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
    # Single-sourced from the module constants, which assert that warmdown
    # begins exactly at STABLE_STEPS.
    return _common_args() + [
        f"--num-iterations={DECAY_NUM_ITERATIONS}",
        f"--warmdown-ratio={DECAY_WARMDOWN_RATIO:.6f}",
        f"--resume-from-step={STABLE_STEPS}",
        f"--save-every={SAVE_EVERY}",
        f"--eval-every={EVAL_EVERY}",
        f"--eval-tokens={EVAL_TOKENS}",
        # SAMPLE_EVERY is scaled to the stable phase (6,100 steps), which is
        # LONGER than a decay variant (3,452) -- using it here would take no
        # samples at all and quietly drop Requirement 9.3 for the decay phase.
        f"--sample-every={max(1, DECAY_STEPS_PER_VARIANT // 4)}",
        f"--model-tag=decay_{variant}",
    ]
