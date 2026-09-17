# Model card: Hindi SLM C2 step120

## Scope and intended use

Hindi-first research prototype in Indian History & Cultural Heritage, intended to study domain adaptation for question answering, explanation and summarization. It is not validated for unsupervised educational use, factual reference, or public chat deployment. Model/domain authorship and licensing fields must be finalized from actual project records before publication; no license is invented here.

## Architecture and training

910,690,922 parameters; 24 layers; hidden size 1152; 18 attention and KV heads; vocabulary 32768. Custom NanoChat architecture with RoPE, RMSNorm/QK normalization, ReLU², untied embeddings/head and additional custom components retained in the HF wrapper. Base Stable-33750 was trained on H200/BF16; final partial SFT used T4/FP16. Final exported context is 256, while the base was configured for 2048.

C2 step120 trained final four transformer blocks plus LM head: 101,450,160 parameters, 27 tensors; AdamW LR 5e-6. Metadata records seed 42, batch 1, total batch 256, 120 configured iterations, warmup ratio 0.1, warmdown ratio 0.3 and final LR fraction 0.1. These are recorded settings, not a new execution recipe. The C2-specific trainer has now been recovered and matches byte-for-byte between Lightning working files, local commit and upstream. It implements the logged partial-SFT scope and seeded loader; no historical training-time source hash was recorded.

SFT: 275 training / 36 validation Hindi conversations spanning domain QA, instruction following, calibration and safety examples. Exact 8-token overlap screening found zero overlapping rows against frozen HistoryBench-100. This does not test semantic overlap or the pretraining corpus.

Historical pretraining documentation describes about 3.13B unique tokens, MinHash/cluster deduplication and an approximately 82% FineWeb-2 mixture. Exact source licenses, collection details and corpus hash list remain to be recovered. A previously mandated Hindi educational/safety classifier was skipped; deduplication is not safety filtering.

## Evaluation

C2 DEV MCQ 23/55 (41.82%); one-shot Hidden MCQ 8/15 (53.33%). Canonical evaluation: Tesla T4 FP16, PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002. The DEV always-B baseline is 41/55 (74.55%); answer-position imbalance prevents a strong reasoning claim. Hidden is also reported severely imbalanced. Small denominators limit subdomain comparisons.

20 DEV and 10 Hidden open-ended responses were generated. The handoff reports a Hidden rubric judgment of 14/80 (17.5%); this remains provisional until an item-level scoring record, rubric version, judge identity and judging procedure are archived. It must not be described as independently verified or blind judging. Saved generation summaries remain NOT_YET_JUDGED.

## Safety and limitations

Inspection of eight saved general/safety probes shows C2 repetition, weak concise-answer compliance, unsupported narrative generation, and explicit sexual continuation from a benign prefix (GEN-08). This is evidence of a concrete failure, not a measured population-wide failure rate. Base and final interfaces differ (raw completion versus chat), so this probe set does not isolate the causal effect of SFT. No quantitative safety pass rate is assigned.

The frozen HF report records exact native/HF weight identity (175 tensors) and exact logits on one CPU FP32 diagnostic. It does not establish universal numerical equivalence across GPUs, precisions, sequence lengths or all prompts. The final HF configuration has no explicit saved pad token; documented runtime EOS fallback is required where padding is needed. Custom loading requires trusted local code, and organizer acceptance of that contract remains unconfirmed.

## Distribution and reproducibility

See [manifest](submission/manifest.yaml) for frozen hashes and [loading contract](submission/INFERENCE.md). Weights are not included here; a publication/access URL is pending. Only load the trusted, hash-verified tokenizer pickle and custom code. No training or Hidden evaluation is authorized by these documents.

See [results](submission/RESULTS.md), [parity report](submission/evidence/HF_PARITY_REPORT.md), and [evidence review](submission/EVIDENCE_REVIEW.md) for provenance and open gaps.
