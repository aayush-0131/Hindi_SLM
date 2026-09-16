# Evaluation findings on `history_partial_c2` (step 120)

Found while integrating this repo's SFT work into the parent Hindi LM project and verifying `model_000120.pt` locally (hash-checked against `FINAL_MODEL_LOCK.txt`, loads cleanly through a from-scratch HF wrapper). Recording here since these bear directly on this repo's own submission.

## 1. General-capability comparison (Requirement 11.2 style, vs. Goldfish-Hindi)

Ran the shipped checkpoint through the parent project's Evaluation Harness (`lighteval accelerate` against an HF export, same method used for the Goldfish-Hindi baseline) on the 5-task FineTasks Hindi suite. Full numbers in `hindi_lm_gate2_history_partial_c2.json`.

| task | Goldfish-Hindi (125M) | base `stable@33,750` | `history_partial_c2` |
|---|---|---|---|
| ARC-hi easy | 0.2513 | 0.2563 | 0.2471 |
| ARC-hi challenge | 0.2363 | 0.2483 | 0.2415 |
| BoolQ-hi | 0.5881 | 0.5685 | **0.3804** |
| HellaSwag-hi | 0.2431 | 0.2530 | 0.2536 |
| Belebele-hin | 0.2600 | 0.2744 | 0.2644 |
| **Aggregate** | ~0.316 | 0.3201 | **0.2774** |

The SFT'd model's aggregate is now below both the base model and the tiny Goldfish baseline. **None of the five scores reflect real reasoning about task content** — reconstructed per-item predictions from lighteval's own cached logprobs show:

- **BoolQ**: predicts "no" (नहीं) on **99.48%** of items (3253/3270) regardless of question content — a near-total mode collapse, not partial competence.
- **ARC-easy/challenge**: picks the first-listed choice ("A") on **62.6%/65.4%** of items against a genuinely ~25%-each gold distribution.
- **HellaSwag/Belebele**: same first-option skew, milder.

## 2. Candidate root cause for the BoolQ collapse

`history_hi_sft_v1_train.jsonl`'s `behavior` category (80/275 rows, 29%) is refusal/decline-style examples, heavily duplicated (only ~5-6 unique prompts repeated many times to fill the count), and nearly all *open* with negation ("मैं ... नहीं करूँगा" / "नहीं।"). Plausible mechanism for the "always no" bias — not proven by ablation, but a concrete, quantifiable candidate.

The `balanced_mcq` category (60 rows, same A/B/C/D format ARC uses) does **not** show an equivalent skew (A 25%, B 33.3%, C 25%, D 16.7%) — so the ARC positional bias is *not* explained by an obvious label imbalance in the matching SFT category. It may predate SFT entirely (untested against the base model's own per-item predictions).

## 3. Safety: the benign-prompt explicit-content issue is broader than documented

`CAPSTONE_STATE.md` already documents explicit content triggered by the benign prefix "मेरा नाम" on the base model. Confirmed independently that a second, even more generic prompt — plain "नमस्ते" ("hello") — also triggers unprompted graphic sexual content on the **shipped SFT'd checkpoint**, not just the base model. The SFT's `behavior` refusal training does not generalize to this — it only covers the specific adversarial phrasings it saw, not the underlying failure mode.

Added `tools/safety_filter.py`: an output-side keyword guardrail (not a fix) that scans generated text and substitutes a refusal on match. Verified against the actual captured explicit output. Real fixes would need either broader/more varied SFT safety data + retraining, or pretraining-corpus decontamination for NSFW content — the latter needs GPU access neither of us currently has.

## 4. HistoryBench-HI v1's own answer-position skew — flagging the risk to the headline number

`CAPSTONE_STATE.md`'s own "Known v1 benchmark limitation" section documents the DEV set's gold answers are severely skewed (B: 41/55, 74.5%; C: 7/55; A: 7/55; D: 0/55). Given (2)'s confirmed general positional-bias pattern, the reported 27.27% → 41.82% DEV MCQ improvement has **not been verified** to reflect real domain reasoning rather than the model learning to answer "B" more often. Neither the frozen DEV CSV nor per-item predictions for this run were available to check either way from the parent project's side — worth checking directly against the actual per-item predictions if you still have them.
