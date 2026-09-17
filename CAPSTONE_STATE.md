# Hindi SLM Capstone — canonical state

Updated 2026-09-17. Project: IAIRO PRAMANA SLM++ Bootcamp; Indian History & Cultural Heritage. Repository: aayush-0131/Hindi_SLM.

## Final decision and restart instructions

**C2 step120 is permanently locked.** No retraining, retuning, alternative checkpoint selection, or Hidden-25 rerun. Hidden-25 was consumed once; the completion lock records 2026-09-17T08:35:21Z. Resume documentation, artifact preservation, Git reconciliation and submission preparation only. Old wandb/SFT-smoke blockers and old instructions to evaluate Hidden are superseded.

The project is at documentation/package preparation, not confirmed submitted. The historical deadline recorded in earlier state files was 2026-09-16 00:00 IST; it has passed. Current acceptance/extension status is unverified. Do not invent a revised deadline.

## Identity

- Base: Stable-33750, 910,690,922 parameters; 24 layers, width 1152, 18 attention and KV heads, vocabulary 32768; full attention, RoPE, ReLU², RMSNorm/QK norm, untied embeddings/head.
- Base context: 2048. Final SFT and frozen HF export declare 256; do not advertise a validated final 2048-token context.
- C2: partial SFT of final four transformer blocks plus LM head; 101,450,160 trainable parameters (11.14%), 27 trainable tensors, AdamW at 5e-6, T4/FP16. Step label 120. Dataset: 275 train / 36 validation conversations; data seed 42; sequence length 256; batch 1, configured total batch 256 tokens.
- Metadata reports final validation BPB 0.3812000506303815. This is SFT validation, not HistoryBench accuracy or directly comparable to base-corpus BPB.
- Native checkpoint SHA256: `c744a163be8b70ac2f2e840e3d0659ac33569816bf9a4ff4eb4b32918e7250a7`.
- Metadata SHA256: `36beab06660d0b5815808f5e2715211686011c01da4114fd647fa7f4e31835ef`.
- All frozen HF file hashes are recorded in [manifest](submission/manifest.yaml); exact tensor parity 175/175 and one FP32 CPU diagnostic forward match are recorded in [parity report](submission/evidence/HF_PARITY_REPORT.md).

## Results and boundaries

| Evidence | Result | Interpretation |
|---|---|---|
| Base DEV MCQ, historical state | 15/55 = 27.27% | Historical baseline; original run not bundled here |
| Locked C2 DEV MCQ, saved summary | 23/55 = 41.82% | Canonical T4 reproduction |
| Locked C2 Hidden MCQ, saved summary | 8/15 = 53.33% | One-shot; severe answer-position imbalance |
| Hidden open-ended generations | 10 | Archived; not regenerated |
| Hidden rubric judgment reported in project handoff | 14/80 = 17.5% | Provisional: item-level scoring record not recovered |
| All-100 exact token 8-gram SFT screen | 0/275 train; 0/36 validation overlaps | PASS_NO_EXACT_NGRAM_OVERLAP; not semantic/pretraining decontamination |

DEV has gold positions A=7, B=41, C=7, D=0: always-B scores 74.55%, above both models. Hidden imbalance is reported but its exact counts are not provided here. Neither raw MCQ result establishes strong reasoning. Both generation summaries still say NOT_YET_JUDGED; preserve them and attach a separate verified judging record rather than silently rewriting them. DEV open-ended judging is unverified.

The eight saved safety/general diagnostics show repetition, weak instruction compliance, and explicit sexual generation from a benign prefix in C2. No deployment-safety claim is justified. See [results](submission/RESULTS.md).

## Evidence locations

- Drive root: `/content/drive/MyDrive/Hindi_SLM_Capstone_Submission`.
- Drive native files: `model_000120.pt`, `meta_000120.json`, `tokenizer.pkl`.
- Drive evaluation archive: `final_eval_evidence/c2_dev_reproduction/` and `final_eval_evidence/c2_hidden25_one_shot/`; completion locks also at `final_eval_evidence/`.
- Drive frozen benchmark: `recovery/HistoryBench_HI_v1_FROZEN.csv`, SHA256 `21dc840c8d4fc07288ddbeaf61c4b8cf41e930b0504131e3252ab45d4e224fbe`. Private: do not commit it or Hidden/prediction content to Git.
- Drive audit: `contamination_audit/20260917T120424_031844Z/all100_contamination_report.json`, SHA256 `52696d5b8af0f1a968ff1370fb5c2e5f4ead63ad4e043e0aad93375302ff3ddc`, with saved checksum sidecar.
- Lightning export: `/teamspace/studios/this_studio/Hindi_SLM/hf_export/`. All eight export hashes verified in the user-run batch. A complete independent backup/access link still needs confirmation.
- Evaluation runtime: Tesla T4 FP16, PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002. The current runtime is not automatically evidence of historical dependency versions.

## Remaining work

Follow [SUBMISSION_CHECKLIST](submission/SUBMISSION_CHECKLIST.md). C2 source is recovered; recover pinned historical environment evidence; preserve HF export; recover judging ledger; reconcile Git before committing; confirm weights access and evaluator custom-code contract; finalize intake/presentation. Do not rerun models to conceal missing records.

Lightning evidence commit: `44cd791203be73d576471e9ffffc4a43380d64c3`; Colab clone: `05d66d66bbe622a149429518ed7a5299e51d0edb`. Tree comparison is complete: upstream retains the C2 source; isolated integration uses upstream as its base. Lightning contains untracked work. No reset, clean, force-push, blanket add, or branch replacement.

Do not publish weights, submit externally, purchase compute, or load the model on the 8 GB Mac without the applicable explicit authorization. Git cleanup/push is requested, but only after reviewing exact changes. Check [evidence review](submission/EVIDENCE_REVIEW.md) for unresolved reproducibility limitations.
