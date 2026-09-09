#!/usr/bin/env python3
"""Requirement 8.1: run Gate 2's benchmark items against our own trained
checkpoint (Decay A / Decay B), not the Goldfish-Hindi baseline.

See eval_harness/nanochat_gate2.py's module docstring for why this bridges
lighteval's task data into nanochat's own native scoring path instead of
going through lighteval's `model_name=<hf_repo>` / HF-transformers loading.

Usage:
    python3 run_gate2_nanochat.py --model-tag decay_a --step 31300
    python3 run_gate2_nanochat.py --model-tag decay_b --step 31300
    python3 run_gate2_nanochat.py --model-tag decay_a --step 31300 --smoke-test
"""

import argparse
import json
import os

import torch

import nanochat.checkpoint_manager as ckpt
from eval_harness.nanochat_gate2 import evaluate_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag", required=True, choices=["decay_a", "decay_b", "stable"])
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--json-out", default=None,
                     help="default: hindi_lm_gate2_nanochat_<model-tag>.json in cwd")
    ap.add_argument("--smoke-test", action="store_true",
                     help="only 5 docs per task, plus print one rendered prompt "
                          "per task -- run this FIRST to sanity-check formatting "
                          "before trusting a full run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_tag} @ step {args.step} on {device}...")
    model, tokenizer, meta = ckpt.load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step)

    results = evaluate_checkpoint(model, tokenizer, device, smoke_test=args.smoke_test)

    report = {
        "model_tag": args.model_tag,
        "step": args.step,
        "smoke_test": args.smoke_test,
        "results": results,
        "aggregate_acc": sum(r["acc"] for r in results.values()) / len(results) if results else None,
    }
    out_path = args.json_out or f"hindi_lm_gate2_nanochat_{args.model_tag}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.abspath(out_path)}")
    print(f"aggregate acc: {report['aggregate_acc']}")


if __name__ == "__main__":
    main()
