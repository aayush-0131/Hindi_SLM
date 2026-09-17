# Hindi SLM — Indian History & Cultural Heritage

A from-scratch Hindi-first 910.7M-parameter language-model capstone for IAIRO PRAMANA SLM++. The intended tasks are Hindi question answering, explanation and summarization for learners. The locked research checkpoint is **C2 step120**, derived by partial supervised fine-tuning of Stable-33750.

**Research artifact, not a reliable student-facing assistant.** Saved outputs show repetition, factual/reasoning weaknesses and unsafe sexual continuation from a benign prompt. No deployment-ready or safety-aligned claim is made.

## Verified engineering and evaluation

- Frozen HF export: custom architecture and tokenizer; saved validation reports 175/175 tensors identical to native weights, with exact forward parity on one CPU FP32 diagnostic.
- C2 DEV MCQ: **23/55 (41.82%)**. Hidden MCQ: **8/15 (53.33%)**, evaluated once after locking.
- Severe answer-position imbalance limits MCQ interpretation. Always-B is 41/55 (74.55%) on DEV; the model does not beat that baseline.
- All-100 exact 8-token SFT overlap audit: zero overlapping training or validation rows. This does not establish absence of semantic or pretraining contamination.
- Hidden open-ended score reported in the handoff: **14/80 (17.5%), provisional pending recovery of its scoring ledger**. Generation itself is complete; no rerun is allowed.

## Model and data

24 layers, width 1152, 18 attention/KV heads, vocabulary 32768, RoPE, RMSNorm/QK norm, ReLU², untied embeddings/head. Base context 2048; final export declares 256. C2 trained the final four transformer blocks and LM head (101,450,160 parameters) with AdamW, LR 5e-6, on 275 training conversations with 36 validation conversations. Historical corpus figures (approximately 3.13B unique tokens, 82% FineWeb-2/18% other) are reported from prior state, not independently reconstructed from corpus artifacts in this bundle. Licensing/source inventory remains incomplete.

## Documentation

- [Canonical state](CAPSTONE_STATE.md)
- [Model card](MODEL_CARD.md)
- [Manifest](submission/manifest.yaml)
- [Results and limitations](submission/RESULTS.md)
- [Evidence review](submission/EVIDENCE_REVIEW.md)
- [Submission checklist](submission/SUBMISSION_CHECKLIST.md)
- [HF loading contract](submission/INFERENCE.md)

The frozen HF folder is currently recorded on Lightning. A recipient-accessible weights link is not yet recorded. The documentation bundle contains no weights or private HistoryBench data. Documentation installation does not publish or submit the project.

C2 step120 is permanently locked. Never retrain, retune, select a replacement checkpoint, or rerun Hidden-25 as part of finalization.
