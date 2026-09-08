"""
Acquires the 2GB Hindi FineWeb-2 training sample and a disjoint held-out
sample for Gate 1 (Requirement 2.1, 2.2). Independent of the Data Pipeline
component -- this pulls its own small, unfiltered sample directly, per
design.md's Boundary Commitments.
"""

from datasets import load_dataset

TRAIN_MAX_CHARS = 2_000_000_000  # matches nanochat's own tok_train.py default
HELDOUT_DOC_COUNT = 20_000
DOC_CAP_CHARS = 10_000  # matches nanochat's own tok_train.py default


def _stream_fineweb2_train():
    return load_dataset("HuggingFaceFW/fineweb-2", "hin_Deva", split="train", streaming=True)


def iter_train_sample(max_chars=TRAIN_MAX_CHARS, doc_cap=DOC_CAP_CHARS):
    """Yields raw text, capped at doc_cap chars/doc, until max_chars total is reached."""
    nchars = 0
    for row in _stream_fineweb2_train():
        text = row["text"][:doc_cap]
        nchars += len(text)
        yield text
        if nchars >= max_chars:
            return


def collect_heldout_sample(after_chars=TRAIN_MAX_CHARS, doc_count=HELDOUT_DOC_COUNT, doc_cap=DOC_CAP_CHARS):
    """
    Returns a list of `doc_count` documents guaranteed disjoint from
    iter_train_sample()'s output: skips ahead past `after_chars` worth of the
    same stream before collecting held-out documents (Requirement 2.2).
    """
    heldout = []
    nchars = 0
    skipping = True
    for row in _stream_fineweb2_train():
        text = row["text"][:doc_cap]
        if skipping:
            nchars += len(text)
            if nchars >= after_chars:
                skipping = False
            continue
        heldout.append(text)
        if len(heldout) >= doc_count:
            break
    return heldout
