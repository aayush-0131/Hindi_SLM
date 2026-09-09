"""
Generates the `token_bytes.pt` companion file for an already-trained tokenizer.

Why this is needed as a separate script: Gate 1 deliberately bypassed
nanochat's `scripts/tok_train.py` and called
`RustBPETokenizer.train_from_iterator` directly, because tok_train.py has no
data-directory flag and always reads from a fixed `base_data_climbmix`
location (see research.md). That was the right call for training -- but
`RustBPETokenizer.save()` writes ONLY `tokenizer.pkl`. The `token_bytes.pt`
file is produced by separate code living in tok_train.py itself (lines ~71-91),
which the bypass therefore skipped.

The result: Gate 1 legitimately passed its own criteria -- fertility 1.2339
(held-out FineWeb-2) / 1.2705 (FLORES dev+devtest), round-trip clean on all
10,014 documents -- because none of those checks touch token_bytes.pt. But
`scripts/base_train.py` calls `get_token_bytes()`, which hard-asserts the file
exists, so pretraining cannot start without it. design.md and research.md both
describe the Gate 1 artifact as "tokenizer.pkl + token_bytes.pt"; the second
half was never actually produced.

What the file is: a vocab_size-length int32 tensor mapping each token id to the
number of raw BYTES that token decodes to, with special tokens set to 0. It
exists so validation loss can be reported as bits-per-byte, which -- unlike
mean token loss -- is comparable across tokenizers with different vocab sizes.
Getting it wrong silently corrupts the primary training metric rather than
crashing, so this replicates tok_train.py's logic exactly, including its
decode_single_token_bytes() call (decoding to a str first would corrupt tokens
that are not valid standalone UTF-8, e.g. raw bytes >= 0x80 -- which is a real
concern for Devanagari, where most characters are 3-byte sequences that BPE
may split).

Usage:
    python3 tokenizer_gate/make_token_bytes.py --tokenizer-dir <dir>
"""

import argparse
import os
import sys


def make_token_bytes(tokenizer_dir, overwrite=False):
    import torch
    from nanochat.tokenizer import RustBPETokenizer

    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"No tokenizer.pkl in {tokenizer_dir} -- point this at the Gate 1 "
            f"output directory."
        )
    out_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    if os.path.exists(out_path) and not overwrite:
        print(f"{out_path} already exists; pass --overwrite to regenerate.")
        return out_path

    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer from {tokenizer_dir} (vocab_size={vocab_size:,})")

    # Exactly tok_train.py's construction.
    special_ids = set(
        tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()
    )
    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes.append(0)  # special tokens are not counted
        else:
            # Raw bytes, NOT decode-to-str-then-len: a token that is not valid
            # standalone UTF-8 would be corrupted by a str round-trip.
            token_bytes.append(len(tokenizer.decode_single_token_bytes(token_id)))

    tensor = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")

    # Sanity checks -- a wrong token_bytes corrupts bits-per-byte silently
    # rather than failing, so verify before writing.
    assert tensor.numel() == vocab_size, (tensor.numel(), vocab_size)
    n_special = int((tensor == 0).sum())
    assert n_special == len(special_ids), (
        f"{n_special} zero-byte entries but {len(special_ids)} special tokens -- "
        f"a non-special token decoding to 0 bytes would corrupt bits-per-byte."
    )
    non_special = tensor[tensor > 0]
    print(f"  special tokens (0 bytes): {n_special}")
    print(f"  bytes/token over the rest: min={int(non_special.min())} "
          f"max={int(non_special.max())} mean={float(non_special.float().mean()):.2f}")

    with open(out_path, "wb") as f:
        torch.save(tensor, f)
    print(f"Saved token_bytes to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="directory holding the Gate 1 tokenizer.pkl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        make_token_bytes(args.tokenizer_dir, overwrite=args.overwrite)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
