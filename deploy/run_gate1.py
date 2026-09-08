#!/usr/bin/env python3
"""
Gate 1 -- Tokenizer Construction and Validation, for the SSH-accessible GPU box.

Trains BPE candidates at 24576/32768/49152, measures fertility against a
held-out FineWeb-2 sample and FLORES-200 Hindi dev+devtest combined, and
verifies exact round-trip on 10,000 documents + script edge cases
(design.md's Tokenizer component, Requirement 2).

IMPORTANT: this script imports `nanochat`, so it must run with the nanochat
repo importable -- e.g. from inside nanochat's activated venv, with the
nanochat repo directory added to PYTHONPATH:

    cd ~/nanochat && source .venv/bin/activate
    PYTHONPATH=~/nanochat:~/hindi-lm python3 ~/hindi-lm/run_gate1.py

This is CPU-only (no GPU needed) and should take well under the ~1 hour
budgeted in the execution sequence.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer_gate import gate_report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("HINDI_LM_TOKENIZER_DIR", os.path.expanduser("~/hindi_lm_tokenizer_gate")),
        help="Root directory for candidate artifacts and the final promoted tokenizer.",
    )
    args = parser.parse_args()

    artifacts_dir = os.path.join(args.base_dir, "candidates")
    canonical_dir = os.path.join(args.base_dir, "tokenizer")

    result = gate_report.run_gate1(artifacts_dir, canonical_dir)

    serializable = dict(result)
    report_path = os.path.join(args.base_dir, "gate1_report.json")
    os.makedirs(args.base_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(serializable, f, default=str, indent=2)
    print(f"Report written to {report_path}")

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
