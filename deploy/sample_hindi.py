#!/usr/bin/env python3
"""Qualitative check: sample real Hindi completions from a checkpoint.

The training log's own periodic generation samples are nanochat's hardcoded
English sanity prompts (base_train.py's fixed list) -- they say nothing
about Hindi quality. This prompts the model with real Devanagari text
instead, so Hindi fluency/coherence can actually be eyeballed rather than
inferred from benchmark numbers alone.

Usage:
    python3 sample_hindi.py --model-tag stable --step 33750
    python3 sample_hindi.py --model-tag decay_a --step 31300 --max-tokens 80
"""

import argparse

import torch

import nanochat.checkpoint_manager as ckpt

PROMPTS = [
    "भारत की राजधानी",
    "सूरज पूर्व दिशा से",
    "मेरा नाम",
    "आज का मौसम",
    "एक समय की बात है",
    "हिंदी भाषा में",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True, choices=["decay_a", "decay_b", "stable"])
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_tag} @ step {args.step} on {device}...")
    model, tokenizer, meta = ckpt.load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step)

    bos_id = tokenizer.get_bos_token_id()
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, prepend=bos_id)
        generated = list(ids)
        for tok in model.generate(list(ids), max_tokens=args.max_tokens,
                                   temperature=args.temperature, top_k=args.top_k,
                                   seed=args.seed):
            generated.append(tok)
        text = tokenizer.decode(generated)
        print(f"\n--- prompt: {prompt!r} ---")
        print(text)


if __name__ == "__main__":
    main()
