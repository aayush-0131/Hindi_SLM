"""
Counts unique tokens across a phase's final shards using the Requirement-2
Gate-1 tokenizer artifact, READ-ONLY. This module does not train or select the
tokenizer -- it only consumes whatever Gate 1 already produced.

Depends on TOKENIZER_DIR containing tokenizer.pkl + token_bytes.pt (the
`tokenizer_gate/` component in design.md, not yet implemented as of this
notebook). Running this before that exists is expected to raise -- that's not
a bug in this file.
"""

import glob
import pyarrow.parquet as pq

REQUIRED_UNIQUE_TOKENS = 12_000_000_000
ENCODE_BATCH_SIZE = 1000  # bounds peak memory; the parallelism comes from the list dispatch  # reduced from 17B: see research.md's windowed-GPU-access recompute


def count_unique_tokens(phase_dir, tokenizer_dir):
    try:
        from nanochat.tokenizer import RustBPETokenizer
    except ImportError as e:
        raise ImportError(
            "nanochat is not installed/importable in this environment. "
            "Install/clone nanochat before running token_count_verify.py."
        ) from e

    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    total_tokens = 0
    for shard_path in sorted(glob.glob(f"{phase_dir}/*.parquet")):
        table = pq.read_table(shard_path, columns=["text"])
        texts = [t for t in table.column("text").to_pylist() if t]
        # Batch the encode calls. RustBPETokenizer.encode() dispatches on input
        # type: a str goes to enc.encode_ordinary() (one Python->Rust call per
        # document, single-threaded), while a LIST goes to
        # enc.encode_ordinary_batch(texts, num_threads=8) -- tiktoken's parallel
        # Rust batch encoder. Encoding one document at a time left 7 of 8
        # threads idle and paid per-call overhead on every row, which is the
        # dominant cost of this pass at corpus scale.
        for start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch = texts[start:start + ENCODE_BATCH_SIZE]
            total_tokens += sum(len(ids) for ids in tokenizer.encode(batch))
    passed = total_tokens >= REQUIRED_UNIQUE_TOKENS
    print(f"{phase_dir}: {total_tokens:,} tokens "
          f"({'PASS' if passed else 'SHORT of'} {REQUIRED_UNIQUE_TOKENS:,} floor)")
    return total_tokens, passed
