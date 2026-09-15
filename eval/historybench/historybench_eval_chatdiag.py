#!/usr/bin/env python3
"""
HistoryBench-HI v1 development-set baseline evaluator for nanochat checkpoints.

What this script does
---------------------
1. Loads the project's native nanochat checkpoint + tokenizer.
2. Scores all development MCQs by conditional log-likelihood of A/B/C/D.
3. Generates deterministic-ish outputs for the open-ended development items.
4. Writes machine-readable predictions and a compact summary.

Important:
- This script uses ONLY the 75-item development split.
- It never reads the 25 hidden items.
- Open-ended responses are generated but NOT automatically judged here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True,
                   help="Root of the Hindi_SLM repo, e.g. ~/Desktop/code_export")
    p.add_argument("--base-dir", required=True,
                   help="Directory containing tokenizer/, e.g. ~/Hindi_SLM_backup")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing model_033750.pt + meta_033750.json")
    p.add_argument("--step", type=int, default=33750)
    p.add_argument("--benchmark-csv", default=str(here / "HistoryBench_HI_v1_DEV_75.csv"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Run only 3 MCQs + 1 open-ended item to validate wiring.")
    return p.parse_args()

def choose_device(torch, requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 75:
        raise ValueError(f"Expected exactly 75 DEV rows, found {len(rows)} in {path}")
    hidden = [r["id"] for r in rows if r.get("split") != "dev"]
    if hidden:
        raise ValueError(f"Non-development rows found in DEV file: {hidden[:5]}")
    return rows

def build_mcq_query(row):
    return (
        f"प्रश्न: {row['question_hi']}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n"
        "सही उत्तर:"
    )

def score_mcq(model, tokenizer, device, row, render_prompts_mc,
               batch_sequences_mc, stack_sequences, forward_model):
    query = build_mcq_query(row)

    tokens, start_idxs, end_idxs = [], [], []

    for choice in ["A", "B", "C", "D"]:
        conversation = {
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": ""},
            ]
        }

        prompt_ids = tokenizer.render_for_completion(conversation)
        choice_ids = tokenizer.encode(choice)

        ids = prompt_ids + choice_ids
        tokens.append(ids)
        start_idxs.append(len(prompt_ids))
        end_idxs.append(len(ids))

    max_seq_len = (
        getattr(model, "max_seq_len", None)
        or getattr(getattr(model, "config", None), "sequence_len", None)
    )

    if max_seq_len is not None:
        new_t, new_s, new_e = [], [], []

        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                cut = len(t) - max_seq_len
                t = t[-max_seq_len:]
                s -= cut
                e -= cut

            if s <= 0:
                raise ValueError(
                    f"{row['id']}: answer continuation was cropped from context"
                )

            new_t.append(t)
            new_s.append(s)
            new_e.append(e)

        tokens, start_idxs, end_idxs = new_t, new_s, new_e

    pad_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_id).to(device)

    losses, _ = forward_model(model, input_ids)

    mean_losses = [
        losses[i, start_idxs[i] - 1:end_idxs[i] - 1].mean().item()
        for i in range(len(tokens))
    ]

    pred_idx = min(range(4), key=lambda i: mean_losses[i])
    pred = "ABCD"[pred_idx]

    return pred, mean_losses, query

def generate_open(model, tokenizer, row, args):
    prompt = f"प्रश्न: {row['question_hi']}\nउत्तर:"

    conversation = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
    }

    ids = tokenizer.render_for_completion(conversation)

    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    generated_only = []

    for tok in model.generate(
        list(ids),
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    ):
        if tok == assistant_end:
            break
        generated_only.append(tok)

    text = tokenizer.decode(generated_only).strip()
    return prompt, text

def acc(correct, total):
    return correct / total if total else None

def main():
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    benchmark_csv = Path(args.benchmark_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # nanochat is a nested package in this repo.
    nanochat_src = repo_root / "nanochat"
    if not nanochat_src.exists():
        raise FileNotFoundError(f"nanochat source not found: {nanochat_src}")
    sys.path.insert(0, str(nanochat_src))

    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)

    import torch
    import nanochat.checkpoint_manager as ckpt
    from nanochat.core_eval import (
        render_prompts_mc,
        batch_sequences_mc,
        stack_sequences,
        forward_model,
    )

    device = choose_device(torch, args.device)
    print(f"[HistoryBench] device={device}")
    print(f"[HistoryBench] checkpoint={checkpoint_dir} step={args.step}")
    print(f"[HistoryBench] base_dir={base_dir}")

    if not (checkpoint_dir / f"model_{args.step:06d}.pt").exists():
        raise FileNotFoundError(checkpoint_dir / f"model_{args.step:06d}.pt")
    if not (checkpoint_dir / f"meta_{args.step:06d}.json").exists():
        raise FileNotFoundError(checkpoint_dir / f"meta_{args.step:06d}.json")
    if not (base_dir / "tokenizer" / "tokenizer.pkl").exists():
        raise FileNotFoundError(base_dir / "tokenizer" / "tokenizer.pkl")

    rows = load_rows(benchmark_csv)

    print("[HistoryBench] loading model...")
    model, tokenizer, meta = ckpt.build_model(
        str(checkpoint_dir), args.step, device, phase="eval"
    )
    print("[HistoryBench] model loaded.")

    mcq_rows = [r for r in rows if r["item_type"] == "mcq"]
    open_rows = [r for r in rows if r["item_type"] == "open_ended"]

    if args.smoke:
        mcq_rows = mcq_rows[:3]
        open_rows = open_rows[:1]
        print("[HistoryBench] SMOKE MODE: 3 MCQ + 1 open-ended")

    predictions = []
    by_subdomain = defaultdict(lambda: [0, 0])
    by_difficulty = defaultdict(lambda: [0, 0])

    print(f"[HistoryBench] scoring {len(mcq_rows)} MCQs...")
    for idx, row in enumerate(mcq_rows, 1):
        pred, losses, rendered = score_mcq(
            model, tokenizer, device, row,
            render_prompts_mc, batch_sequences_mc, stack_sequences, forward_model
        )
        gold = row["answer_key"].strip().upper()
        ok = pred == gold
        by_subdomain[row["subdomain"]][0] += int(ok)
        by_subdomain[row["subdomain"]][1] += 1
        by_difficulty[row["difficulty"]][0] += int(ok)
        by_difficulty[row["difficulty"]][1] += 1

        predictions.append({
            "id": row["id"],
            "subdomain": row["subdomain"],
            "difficulty": row["difficulty"],
            "item_type": "mcq",
            "gold": gold,
            "prediction": pred,
            "correct": ok,
            "choice_mean_losses": dict(zip("ABCD", losses)),
            "rendered_query": rendered,
        })
        print(f"  [{idx:02d}/{len(mcq_rows)}] {row['id']}  pred={pred} gold={gold}  {'✓' if ok else '✗'}")

    print(f"[HistoryBench] generating {len(open_rows)} open-ended responses...")
    open_outputs = []
    for idx, row in enumerate(open_rows, 1):
        prompt, response = generate_open(model, tokenizer, row, args)
        rec = {
            "id": row["id"],
            "subdomain": row["subdomain"],
            "difficulty": row["difficulty"],
            "item_type": "open_ended",
            "prompt": prompt,
            "response": response,
            "reference_answer_hi": row["reference_answer_hi"],
            "grading_rubric": row["grading_rubric"],
            "judge_status": "NOT_YET_JUDGED",
        }
        open_outputs.append(rec)
        predictions.append(rec)
        print(f"  [{idx:02d}/{len(open_rows)}] {row['id']}  chars={len(response)}")

    total_correct = sum(int(p["correct"]) for p in predictions if p.get("item_type") == "mcq")
    total_mcq = sum(1 for p in predictions if p.get("item_type") == "mcq")

    summary = {
        "benchmark": "HistoryBench-HI-v1",
        "split": "dev",
        "checkpoint_step": args.step,
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "smoke": bool(args.smoke),
        "mcq": {
            "correct": total_correct,
            "n": total_mcq,
            "accuracy": acc(total_correct, total_mcq),
            "by_subdomain": {
                k: {"correct": v[0], "n": v[1], "accuracy": acc(v[0], v[1])}
                for k, v in sorted(by_subdomain.items())
            },
            "by_difficulty": {
                k: {"correct": v[0], "n": v[1], "accuracy": acc(v[0], v[1])}
                for k, v in sorted(by_difficulty.items())
            },
        },
        "open_ended": {
            "n": len(open_outputs),
            "judge_status": "NOT_YET_JUDGED",
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seed": args.seed,
            }
        },
        "note": (
            "Development baseline only. Hidden 25 were not read. "
            "Open-ended outputs require blind rubric-based judging before any L3-style score is computed."
        ),
    }

    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    open_path = output_dir / "open_ended_for_judging.jsonl"
    with open(open_path, "w", encoding="utf-8") as f:
        for r in open_outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== HistoryBench-HI DEV BASELINE ===")
    if total_mcq:
        print(f"MCQ accuracy: {total_correct}/{total_mcq} = {total_correct/total_mcq:.4f}")
    else:
        print("MCQ accuracy: n/a")
    for k, v in sorted(summary["mcq"]["by_difficulty"].items()):
        print(f"  {k:>6}: {v['correct']}/{v['n']} = {v['accuracy']:.4f}")
    print(f"Open-ended outputs waiting for judge: {len(open_outputs)}")
    print(f"\nWrote:\n  {summary_path}\n  {pred_path}\n  {open_path}")

if __name__ == "__main__":
    main()
