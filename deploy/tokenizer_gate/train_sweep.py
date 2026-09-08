"""
Trains BPE candidates at 24576/32768/49152 vocab sizes (Requirement 2.1) via
nanochat's own RustBPETokenizer.train_from_iterator -- called directly, not
through scripts/tok_train.py's CLI, since that script has no data-directory
flag and always reads from a fixed base_dir/base_data_climbmix location
(confirmed via nanochat source). Calling the Python API directly lets us
train on our own acquired sample without fighting that convention.

Requires nanochat's repo to be importable -- run with the nanochat repo on
PYTHONPATH (e.g. from within its activated venv, with the repo directory
itself also on sys.path).
"""

import os

import nanochat.tokenizer as nanochat_tokenizer_module
from nanochat.tokenizer import RustBPETokenizer

VOCAB_SIZES = [24576, 32768, 49152]

# Mitigation pattern (research.md Design Decision): widen the letter-run term
# from \p{L}+ to [\p{L}\p{M}]+ so combining marks (matras, nukta, virama)
# stay grouped with their base consonant before BPE merging runs. Only used
# as a one-time retry if the default pattern misses Gate 1's fertility bar
# at 32K (Requirement 2.4).
WIDENED_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+[\p{L}\p{M}]+"""
    r"""|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def train_candidate(text_iterator, vocab_size, tokenizer_dir, pattern_override=None):
    """Trains one candidate, optionally overriding nanochat's module-level
    SPLIT_PATTERN for the duration of this call (train_from_iterator reads it
    by name at call time, so reassigning the module attribute here is
    sufficient -- no fork of nanochat needed), and saves it to tokenizer_dir."""
    original_pattern = nanochat_tokenizer_module.SPLIT_PATTERN
    if pattern_override is not None:
        nanochat_tokenizer_module.SPLIT_PATTERN = pattern_override
    try:
        tokenizer = RustBPETokenizer.train_from_iterator(text_iterator, vocab_size)
    finally:
        nanochat_tokenizer_module.SPLIT_PATTERN = original_pattern
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(tokenizer_dir)
    return tokenizer
