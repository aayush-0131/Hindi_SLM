#!/usr/bin/env python3
"""Build nanochat/scripts/chat_sft_history.py deterministically from upstream chat_sft.py.

This repairs the previously documented broken helper. It is deliberately conservative:
- refuses to overwrite an existing trainer unless --force is supplied;
- validates exact source anchors before editing;
- replaces only task imports and train/validation dataset construction;
- compiles the generated trainer before accepting it.

It does not load model weights, train, evaluate, or access HistoryBench.
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

TASK_IMPORTS = '''from tasks.common import TaskMixture
from tasks.history_hi_sft import HistoryHiSFT
'''

UPSTREAM_IMPORTS = '''from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
'''

UPSTREAM_DATA_START = '''train_tasks = [
    SmolTalk(split="train"), # 460K rows of general conversations
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
]
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
'''

HISTORY_DATA = '''train_tasks = [HistoryHiSFT(split="train")]
train_dataset = TaskMixture(train_tasks)
print0(f"History-HI SFT training set: {len(train_dataset):,} rows")
val_dataset = TaskMixture([HistoryHiSFT(split="val")])
print0(f"History-HI SFT validation set: {len(val_dataset):,} rows")
'''


def replace_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Refusing generation: expected exactly one {label} anchor, found {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--force", action="store_true", help="replace an existing generated trainer")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    src = repo / "nanochat/scripts/chat_sft.py"
    task = repo / "nanochat/tasks/history_hi_sft.py"
    dst = repo / "nanochat/scripts/chat_sft_history.py"

    if not src.is_file():
        raise SystemExit(f"Missing upstream trainer: {src}")
    if not task.is_file():
        raise SystemExit(
            f"Missing custom task: {task}. Preserve/recover the validated HistoryHiSFT task first; "
            "this builder will not invent dataset semantics."
        )
    if dst.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing trainer: {dst} (use --force only after review)")

    text = src.read_text(encoding="utf-8")
    text = replace_exact_once(text, UPSTREAM_IMPORTS, TASK_IMPORTS, "upstream task-import")
    text = replace_exact_once(text, UPSTREAM_DATA_START, HISTORY_DATA, "upstream data-mixture")

    # Defensive checks: generated trainer must use only the capstone dataset in the SFT mixture.
    required = ['HistoryHiSFT(split="train")', 'HistoryHiSFT(split="val")']
    for marker in required:
        if marker not in text:
            raise SystemExit(f"Generated trainer missing required marker: {marker}")
    for forbidden in ["SmolTalk(split=", "MMLU(subset=", "GSM8K(subset="]:
        if forbidden in text:
            raise SystemExit(f"Generated trainer still contains forbidden upstream dataset marker: {forbidden}")

    tmp = dst.with_suffix(".py.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(tmp), doraise=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # Atomic final placement; destination absence/--force was checked above.
    tmp.replace(dst)
    print(f"BUILT={dst}")
    print("PY_COMPILE=PASS")
    print("DATASET_WIRING=HistoryHiSFT(train)+HistoryHiSFT(val)")


if __name__ == "__main__":
    main()
