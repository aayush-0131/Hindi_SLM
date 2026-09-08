#!/usr/bin/env python3
"""
Gate 2 -- Evaluation Harness validation (Requirement 5), for the
SSH-accessible GPU box.

Requires `lighteval` installed (pip install lighteval) and network access to
pull the baseline model + Hindi task datasets from Hugging Face.

Run:
    python3 run_gate2.py

Scope decision (see eval_harness/config.py and research.md): this validates
the harness against Goldfish-Hindi (the causal-LM "primary comparison"
baseline, same architecture family as the target model) only. MuRIL and
IndicBERT v2 are masked-LM encoders that can't be scored via lighteval's
standard causal-LM loglikelihood multiple-choice method without a different
scoring approach not yet implemented -- flagged as a deliberate, not
accidental, scope limitation for now.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_harness import gate_report, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo", default=config.PRIMARY_BASELINE["hf_repo"])
    parser.add_argument(
        "--report-path",
        default=os.environ.get("HINDI_LM_GATE2_REPORT", os.path.expanduser("~/hindi_lm_gate2_report.json")),
    )
    args = parser.parse_args()

    result = gate_report.run_gate2(hf_repo=args.hf_repo)

    with open(args.report_path, "w") as f:
        json.dump(result, f, default=str, indent=2)
    print(f"\nReport written to {args.report_path}")

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
