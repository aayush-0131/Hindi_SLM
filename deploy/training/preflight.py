"""
Fails loudly, on CPU, before any GPU time is spent on Gate 3 or pretraining.

Why this exists: nanochat resolves BOTH the tokenizer and the training corpus
from get_base_dir() alone -- neither scripts/base_train.py nor
nanochat/dataloader.py threads an explicit data directory through (verified:
list_parquet_files(data_dir=...) accepts one, but the dataloader calls it with
no argument). So the entire wiring between our Gate-1 tokenizer, our pipeline's
shards, and nanochat is a single environment variable plus an exact directory
layout, and getting it wrong is silent -- nanochat would either train on
ClimbMix English shards or abort mid-startup after the GPU is already reserved.

Required layout:
    $NANOCHAT_BASE_DIR/
        tokenizer/              <- tokenizer.pkl + token_bytes.pt (Gate 1 output)
        base_data_climbmix/     <- our final/<phase>/*.parquet shards
        base_checkpoints/       <- created by nanochat; MUST be persistent disk
"""

import glob
import os
import sys

from training import config as cfg

TOKENIZER_SUBDIR = "tokenizer"
DATA_SUBDIR = "base_data_climbmix"   # nanochat's hardcoded name; ours is a symlink
CHECKPOINT_SUBDIR = "base_checkpoints"


class PreflightError(RuntimeError):
    pass


def _base_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        raise PreflightError(
            "NANOCHAT_BASE_DIR is not set. Without it nanochat defaults to "
            "~/.cache/nanochat -- which on this host is NOT the persistent disk, "
            "so checkpoints would not survive the gap between GPU windows "
            "(Requirement 6.5 / 7.7). Set it to a /data1/... path."
        )
    if not os.path.isdir(base):
        raise PreflightError(f"NANOCHAT_BASE_DIR={base} does not exist.")
    return base


def check_tokenizer(base):
    """Gate-1 artifact present, loadable, and the vocab size Requirement 1.3 fixes."""
    tok_dir = os.path.join(base, TOKENIZER_SUBDIR)
    for fname in ("tokenizer.pkl", "token_bytes.pt"):
        path = os.path.join(tok_dir, fname)
        if not os.path.exists(path):
            raise PreflightError(
                f"Missing {path}. nanochat reads the tokenizer from "
                f"$NANOCHAT_BASE_DIR/{TOKENIZER_SUBDIR}/ and nowhere else -- symlink "
                f"the Gate-1 output directory there."
            )
    try:
        from nanochat.tokenizer import RustBPETokenizer
    except ImportError as e:
        raise PreflightError(
            "Cannot import nanochat. Run this from the nanochat venv with the repo "
            "on PYTHONPATH (that venv has no pip -- use `uv pip install` if a "
            "dependency is missing)."
        ) from e

    tokenizer = RustBPETokenizer.from_directory(tok_dir)
    vocab = tokenizer.get_vocab_size()
    if vocab != cfg.VOCAB_SIZE:
        raise PreflightError(
            f"Tokenizer vocab is {vocab:,}, expected {cfg.VOCAB_SIZE:,} "
            f"(Requirement 1.3). Wrong candidate promoted out of the Gate 1 sweep?"
        )
    print(f"  tokenizer: OK  ({tok_dir}, vocab {vocab:,})")
    return tokenizer


def check_shards(base):
    """Corpus present and shaped the way nanochat's dataloader expects."""
    import pyarrow.parquet as pq

    data_dir = os.path.join(base, DATA_SUBDIR)
    if not os.path.isdir(data_dir):
        raise PreflightError(
            f"Missing {data_dir}. nanochat's DATA_DIR is hardcoded to "
            f"<base_dir>/{DATA_SUBDIR} -- symlink the phase directory you want to "
            f"train on (e.g. final/stable) to exactly that name."
        )
    shards = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not shards:
        raise PreflightError(f"No .parquet shards found in {data_dir}.")
    if len(shards) < 2:
        raise PreflightError(
            f"Only {len(shards)} shard in {data_dir}. nanochat treats the LAST shard "
            f"in a split as the validation set, so a single shard leaves zero "
            f"training data."
        )

    schema = pq.read_schema(shards[0])
    if "text" not in schema.names:
        raise PreflightError(
            f"{shards[0]} has no 'text' column (columns: {schema.names}). "
            f"nanochat's dataloader requires it."
        )
    rows = sum(pq.read_metadata(s).num_rows for s in shards)
    print(f"  corpus:    OK  ({len(shards)} shards, {rows:,} rows, "
          f"train={len(shards) - 1} / val=1)")
    return shards, rows


def estimate_tokens(tokenizer, shards, total_rows, sample_rows=2000):
    """
    Extrapolated token count, from a sample. This is deliberately NOT
    shard/token_count_verify.py: that one encodes every document to check the
    >=12B floor and takes hours. Here we only need to know whether the corpus
    covers the step budget or whether the dataloader will be looping epochs.
    """
    import pyarrow.parquet as pq

    seen, chars = 0, 0
    tokens = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["text"])
        for row in table.to_pylist():
            text = row["text"] or ""
            tokens += len(tokenizer.encode(text))
            chars += len(text)
            seen += 1
            if seen >= sample_rows:
                break
        if seen >= sample_rows:
            break
    if seen == 0:
        raise PreflightError("Sampled 0 rows -- shards contain no documents.")

    per_doc = tokens / seen
    estimated = int(per_doc * total_rows)
    print(f"  tokens:    ~{estimated / 1e9:.2f}B estimated "
          f"({per_doc:,.0f} tok/doc over {seen:,} sampled docs, "
          f"{chars / seen:,.0f} chars/doc)")
    return estimated


def report_budget(estimated_tokens):
    """Turn the token estimate into the two numbers that actually drive decisions."""
    stable_tokens = cfg.STABLE_STEPS * cfg.TOTAL_BATCH_TOKENS
    print()
    print(f"  stable phase consumes {stable_tokens / 1e9:.2f}B tokens "
          f"({cfg.STABLE_STEPS:,} steps x {cfg.TOTAL_BATCH_TOKENS:,})")
    if estimated_tokens <= 0:
        return
    epochs = stable_tokens / estimated_tokens
    print(f"  implied epochs over the available corpus: {epochs:.2f}")
    if epochs <= 1.0:
        print("  -> single pass, corpus is larger than the budget needs.")
    elif epochs <= 4.0:
        print("  -> repeats, but under the ~4-epoch envelope where repeated tokens "
              "are near-equivalent to fresh ones (Muennighoff et al. 2023). "
              "Acceptable; no need to block training on more corpus.")
    else:
        print(f"  -> WARNING: {epochs:.1f} epochs exceeds the ~4-epoch envelope. "
              "Past that, additional passes teach progressively less. Grow the "
              "corpus or cut the step budget.")

    hours = cfg.STABLE_STEPS * cfg.SEC_PER_STEP_BUDGET / 3600
    print(f"  stable phase at the budgeted {cfg.SEC_PER_STEP_BUDGET}s/step: "
          f"{hours:.1f} GPU-hours (Requirement 7.1 allows 22.4h)")


def print_deviations():
    print()
    print("Architecture deviations from the build spec (Requirement 1.4):")
    for spec, actual, why in cfg.ARCH_DEVIATIONS:
        print(f"  - spec: {spec}")
        print(f"    ours: {actual}")
        print(f"    why:  {why}")
    print()
    print("  base_train.py prints the realized parameter count and FLOPs/token at "
          "startup. Read both off the Gate 3 shakedown and record them -- that is "
          "Requirement 1.4's C=6ND recompute, and it cannot be done from this "
          "script because the added k/v and wider MLP params are nanochat's, not "
          "the spec's.")


def main():
    print("=" * 72)
    print("Pretraining preflight (CPU only -- no GPU time spent here)")
    print("=" * 72)
    base = _base_dir()
    print(f"  NANOCHAT_BASE_DIR = {base}")

    ckpt_dir = os.path.join(base, CHECKPOINT_SUBDIR)
    print(f"  checkpoints ->      {ckpt_dir}")
    if not base.startswith("/data1"):
        print("  NOTE: base dir is not under /data1 -- confirm this path is the "
              "persistent disk that survives the gap between GPU windows.")

    tokenizer = check_tokenizer(base)
    shards, rows = check_shards(base)
    estimated = estimate_tokens(tokenizer, shards, rows)
    report_budget(estimated)
    print_deviations()

    print()
    print("=" * 72)
    print("Commands (run from the nanochat repo root, inside tmux):")
    print("=" * 72)
    print("\n# Gate 3 -- throughput shakedown (Requirement 6)")
    print(f"python -m scripts.base_train {' '.join(cfg.gate3_args())}")
    print("\n# Stable phase (Requirement 7.1) -- fresh start")
    print(f"python -m scripts.base_train {' '.join(cfg.stable_args())}")
    print("\n# Stable phase -- resume after the window boundary (Requirement 7.7)")
    print(f"python -m scripts.base_train {' '.join(cfg.stable_args(resume_from_step='<LAST_SAVED_STEP>'))}")
    print("\n# Decay A / Decay B (Requirements 7.3 / 7.4)")
    # Derived, not hardcoded: this read *022400* long after STABLE_STEPS moved
    # to 27,000, so it named a checkpoint the decay commands below would not
    # then find.
    print(f"#   first: cp base_checkpoints/stable/*{cfg.STABLE_STEPS:06d}* "
          f"into base_checkpoints/decay_{{a,b}}/")
    print(f"python -m scripts.base_train {' '.join(cfg.decay_args('a'))}")
    print(f"python -m scripts.base_train {' '.join(cfg.decay_args('b'))}")
    print()
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    try:
        main()
    except PreflightError as e:
        print(f"\nPREFLIGHT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
