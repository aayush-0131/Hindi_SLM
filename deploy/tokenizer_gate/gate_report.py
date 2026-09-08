"""
Applies Gate 1's pass/fail decision (Requirement 2.3, 2.4, 2.6): trains the
three vocab-size candidates, measures fertility, owns the ONE pre-planned
mitigation retry for the 32K candidate if the default pre-tokenizer pattern
misses the 1.45 bar, requires the round-trip check to fully pass, and only
then promotes the tokenizer to the canonical directory downstream components
(nanochat training, the Data Pipeline's token_count_verify.py) consume.
"""

import os
import shutil

from nanochat.tokenizer import RustBPETokenizer

from tokenizer_gate import acquire_samples, fertility_eval, roundtrip_verify, train_sweep

FERTILITY_THRESHOLD = 1.45
ROUNDTRIP_SAMPLE_SIZE = 10_000


def run_gate1(artifacts_dir, canonical_tokenizer_dir):
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Acquiring 2GB training sample + held-out sample + FLORES-200 dev+devtest...")
    train_texts = list(acquire_samples.iter_train_sample())
    heldout_texts = acquire_samples.collect_heldout_sample()
    flores_texts = fertility_eval.load_flores_hin_deva_combined()
    print(
        f"Train sample: {len(train_texts)} docs. Held-out: {len(heldout_texts)} docs. "
        f"FLORES dev+devtest: {len(flores_texts)} sentences."
    )

    results = {}
    for vocab_size in train_sweep.VOCAB_SIZES:
        key = f"{vocab_size}_default"
        tok_dir = os.path.join(artifacts_dir, key)
        print(f"Training vocab_size={vocab_size} (default pattern)...")
        tokenizer = train_sweep.train_candidate(iter(train_texts), vocab_size, tok_dir)
        fert = fertility_eval.evaluate_candidate(tokenizer, heldout_texts, flores_texts)
        results[key] = {"tokenizer_dir": tok_dir, **fert}
        print(
            f"  fertility: fineweb2={fert['fertility_fineweb2']:.4f} "
            f"flores={fert['fertility_flores']:.4f}"
        )

    target = results["32768_default"]
    passed_fertility = (
        target["fertility_fineweb2"] <= FERTILITY_THRESHOLD
        and target["fertility_flores"] <= FERTILITY_THRESHOLD
    )
    pattern_used = "default"

    if not passed_fertility:
        print(
            "32K default pattern missed the 1.45 bar -- retrying once with the "
            "widened Devanagari-combining-mark pattern (research.md Design Decision)."
        )
        key = "32768_widened_marks"
        tok_dir = os.path.join(artifacts_dir, key)
        tokenizer = train_sweep.train_candidate(
            iter(train_texts), 32768, tok_dir, pattern_override=train_sweep.WIDENED_PATTERN
        )
        fert = fertility_eval.evaluate_candidate(tokenizer, heldout_texts, flores_texts)
        results[key] = {"tokenizer_dir": tok_dir, **fert}
        print(
            f"  retry fertility: fineweb2={fert['fertility_fineweb2']:.4f} "
            f"flores={fert['fertility_flores']:.4f}"
        )
        target = results[key]
        passed_fertility = (
            target["fertility_fineweb2"] <= FERTILITY_THRESHOLD
            and target["fertility_flores"] <= FERTILITY_THRESHOLD
        )
        pattern_used = "widened_marks"

    if not passed_fertility:
        print(
            "GATE 1 FAILED: 32K vocabulary misses the 1.45 fertility bar even with "
            "the widened pattern. Escalate to the project owner -- do not proceed "
            "to pretraining (Requirement 2.4)."
        )
        return {"passed": False, "results": results}

    print(
        f"Fertility bar passed (pattern={pattern_used}). Running round-trip verification "
        f"on {ROUNDTRIP_SAMPLE_SIZE} documents + script edge cases..."
    )
    winning_tokenizer = RustBPETokenizer.from_directory(target["tokenizer_dir"])
    roundtrip_docs = roundtrip_verify.sample_documents(iter(train_texts), n=ROUNDTRIP_SAMPLE_SIZE)
    failures = roundtrip_verify.verify_roundtrip(winning_tokenizer, roundtrip_docs)

    if failures:
        print(
            f"GATE 1 FAILED: {len(failures)} round-trip mismatches (Requirement 2.6). "
            f"First failure: {failures[0]}"
        )
        return {"passed": False, "results": results, "roundtrip_failures": failures}

    total_checked = len(roundtrip_docs) + len(roundtrip_verify.SCRIPT_EDGE_CASES)
    print(f"Round-trip verification passed on all {total_checked} documents.")

    if os.path.isdir(canonical_tokenizer_dir):
        shutil.rmtree(canonical_tokenizer_dir)
    shutil.copytree(target["tokenizer_dir"], canonical_tokenizer_dir)
    print(f"GATE 1 PASSED. Tokenizer promoted to {canonical_tokenizer_dir}.")
    return {
        "passed": True,
        "results": results,
        "pattern_used": pattern_used,
        "tokenizer_dir": canonical_tokenizer_dir,
    }
