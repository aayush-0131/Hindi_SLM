# Hindi SLM Capstone — Canonical Project State

**Project:** IAIRO PRAMANA SLM++ Bootcamp Capstone  
**Repository:** `aayush-0131/Hindi_SLM`  
**Canonical state timestamp:** 2026-09-14 12:27 IST  
**Hard submission deadline:** 2026-09-16 00:00 IST  
**Purpose of this file:** single source of truth for project state across long chats, machines, and fresh sessions. Read this file before making strategic or execution decisions.

---

## 1. Non-negotiable project decisions

- **Model:** from-scratch Hindi-first small language model, approximately 910M parameters.
- **Chosen domain:** Indian History & Cultural Heritage.
- **Primary tasks:** question answering, explanation, summarization.
- **Target user:** Hindi-medium students and learners.
- **Subdomains:**
  1. Ancient India
  2. Medieval India
  3. Modern India & Freedom Movement
  4. Indian Art, Architecture & Heritage
  5. Literature, Society & Cultural Traditions
- **Do not rerun full pretraining.** Current base checkpoint is the rescued Stable-33750 checkpoint.
- **Do not change the frozen benchmark during model selection.** Any correction after freeze requires a new benchmark version and new hashes.
- **Do not use hidden benchmark items during iterative model selection or SFT data creation.**
- **Do not publish weights, submit externally, purchase compute, or make a major model/domain strategy change without explicit approval.**

---

## 2. Base model identity

### Stable-33750

- Training step: **33,750**
- Exact parameter count: **910,690,922**
- Layers: **24**
- Hidden size / d_model: **1152**
- Attention heads: **18**
- KV heads: **18** (effective MHA for this instantiated model; code supports GQA)
- Vocabulary: **32,768**
- Context length: **2,048**
- Attention window pattern: **`L`** (full attention)
- Activation: **ReLU²**
- Normalization: **RMSNorm**, with QK norm in model code
- Positional encoding: **RoPE**
- Embedding / LM head: **untied**
- Training precision: **BF16**
- Optimizer family: **Muon / MuonAdamW**
- Total batch size: **524,288 tokens/step**
- Hardware used for main training: **NVIDIA H200, world size 1**
- Stable checkpoint final validation BPB: **0.3177861748987491**
- Stable checkpoint smoothed train loss: **2.159572934694303**
- Stable cumulative training time in metadata: **108,903.187 seconds (~30.25 h)**
- Initial documented train loss: **10.398**

**Interpretation:** Stable-33750 is a **base/pretrained completion model**, not an instruction/chat model and not safe for unrestricted deployment.

---

## 3. Rescued model artifacts and canonical hashes

Local backup root:

```text
~/Hindi_SLM_backup
```

Critical files:

| Artifact | Canonical SHA-256 |
|---|---|
| `ckpt_stable_final/model_033750.pt` | `1ad56ea60a5e28d6899146a56f63b1833837dbe6f6cef9128bb2bf4a51faa00d` |
| `ckpt_stable_final/meta_033750.json` | `8f2d7c586d0c9ddc30633dee4310c4759ca9068b5c256595f124f57a43563013` |
| `tokenizer/tokenizer.pkl` | `25be857870b2cebbc1dfc729738d6e03dd1399821cdeb1d1d85a86be6ab59823` |
| `tokenizer/token_bytes.pt` | `96e1e74bbbfae0c277a5b8644f64487325745ed2f26cf3a3a267271c98999234` |

The four critical artifacts were uploaded to the Google Drive folder:

```text
Hindi_SLM_Capstone_Submission
```

Remote/cloud execution should verify these hashes before evaluation or SFT.

---

## 4. Tokenizer and corpus facts

- Tokenizer: byte-level BPE
- Vocabulary: **32,768**
- Devanagari-aware regex optimization includes `[` + `\p{L}\p{M}` + `]+`
- Historical tokenizer fertility: approximately **1.23**
- Pretraining corpus: approximately **3.13B unique tokens** in Round 1
- Approximately **127 parquet shards** (126 train + 1 validation)
- Approximate mixture: **82% FineWeb-2 / 18% other sources**
- Corpus deduplication included MinHash/cluster filtering.

Important safety limitation: bulk FineWeb-2 ingestion skipped the mandated `finepdfs_edu_classifier_hin_Deva` classifier because its estimated compute cost exceeded the project budget. Deduplication is not equivalent to safety filtering.

---

## 5. Safety finding that must remain in final reporting

Qualitative generation from Stable-33750 exposed explicit sexual/adult continuation on a benign Hindi prefix (`मेरा नाम`). Decay A/B checkpoints showed similar contamination.

**Mitigation plan:**

- no full pretraining rerun;
- document the issue transparently;
- create explicit safety evaluation prompts;
- include safety-aligned examples in SFT;
- compare Base vs Instruct safety behavior;
- add inference/demo guardrails;
- release/describe base and instruct models separately;
- do not claim the raw base model is deployment-safe.

---

## 6. Training health audit status

Formal conclusion on available historical evidence:

> **LEVEL 1 — PASS on available evidence, with one documentation limitation (`grad_norm` unavailable).**

Supporting facts:

- no NaN/Inf found in available logs;
- initial loss 10.398 -> final smoothed loss 2.15957 (>79% reduction);
- final validation BPB improved relative to earlier checkpoints;
- approximate validation/train loss gap was below the 20% threshold used in the audit;
- resume test passed;
- historical H200 throughput/MFU evidence exceeded the efficiency gate;
- historical base logs do **not** contain `grad_norm`, and this must be reported as unavailable rather than fabricated;
- an OOM seen in a Round 2 capture came from external evaluation, not the training loop.

---

## 7. HistoryBench-HI v1 — frozen evaluation benchmark

**Status:** FROZEN before SFT.

- Total items: **100**
- DEV: **75**
- Hidden: **25**
- Subdomains: **5 x 20 items**
- Difficulty mix: **30 Easy / 40 Medium / 30 Hard**
- Language: Hindi-first
- Hidden placement: 5 hidden items per subdomain
- Hard/open-ended rubric includes correctness and reasoning assessment.

### Canonical benchmark hashes

| File | SHA-256 |
|---|---|
| `HistoryBench_HI_v1_FROZEN.xlsx` | `aa057dd846da02dae7d43d40d7307d3d7ba621cd0554d5941b0d3c63798f6c30` |
| `HistoryBench_HI_v1_FROZEN.csv` | `21dc840c8d4fc07288ddbeaf61c4b8cf41e930b0504131e3252ab45d4e224fbe` |
| `HistoryBench_HI_v1_FROZEN.jsonl` | `49d4505fdb4d4c81c32133c4a62ef4072c0614c8e15630a828fa0f9f6a8c35a3` |
| `HistoryBench_HI_v1_DEV_75.csv` | `ac69ea72ba0721e9a934825e93070e4117f3554d66acb693aa4e4033ef5d78e5` |
| `HistoryBench_HI_v1_HIDDEN_25_KEEP_PRIVATE.csv` | `9c8b56d2f528fdcede9a1a835a31a6479a791a0b786408745a614c167dfc90b2` |

### Freeze policy

During training/model selection, do not change:

- wording;
- answers;
- difficulty labels;
- dev/hidden split;
- reference answers;
- grading rubrics.

If a correction is unavoidable, create **HistoryBench-HI v1.1** (or later) with fresh hashes and document the change.

### Leakage policy

- Do not feed benchmark questions, reference answers, hidden items, or benchmark-derived prompts into SFT data generation.
- After the final SFT corpus is created, run an n-gram contamination scan against **all 100 benchmark items** before training/final reporting.
- Hidden 25 should be used only for the final locked model comparison, not iterative tuning.

---

## 8. Evaluation methodology

### DEV baseline

- Run on Stable-33750 before SFT.
- MCQ: zero-shot conditional mean token loss over A/B/C/D choices using native NanoChat evaluation utilities.
- Open-ended: generate deterministic/locked outputs and save for later blind rubric judging.
- Save prompts, outputs, scores, environment metadata, precision, code commit, and benchmark hash.

### Final comparison

At minimum:

1. Stable-33750 Base on DEV
2. Final instruction/domain/safety-tuned model on DEV
3. Final locked model on Hidden-25
4. Safety comparison Base vs Instruct
5. General/instruction capability checks required by bootcamp rubric

Do not use hidden performance to select checkpoints.

---

## 9. Current local-compute incident and decision

### Mac incident

Machine: Apple Silicon MacBook Air with **8 GB unified memory**.

The first native MPS smoke test reached:

```text
[HistoryBench] loading model...
```

and the machine kernel-panicked/restarted. The panic report showed a watchdog timeout and low swap headroom.

NanoChat's checkpoint loader converts BF16 model tensors to FP32 when loading on CPU/MPS. On an 8 GB unified-memory machine this creates unacceptable memory pressure for a ~910M model.

**Decision:** do not attempt another full Stable-33750 load on the Mac.

Use the Mac for:

- control/SSH/browser;
- artifact handling;
- documentation;
- Git/Drive operations;
- lightweight data preparation.

Use remote NVIDIA GPU for:

- base evaluation;
- SFT;
- final inference/evaluation.

---

## 10. Current remote GPU status

Google Colab Free currently assigned:

- GPU: **Tesla T4**
- Compute capability: **7.5 (SM75)**
- VRAM: **~14.6 GB**

NanoChat auto-selects BF16 on CUDA SM80+; the T4 is pre-Ampere and does not provide the canonical BF16 path expected for this project.

**Current decision:**

- T4 may be retained as an emergency engineering/smoke option only if explicitly chosen;
- do **not** use T4 numbers as the canonical evaluation while the intake requires evaluation precision to match training;
- first seek a free/available **SM80+** GPU (examples: L4, A10/A40/A5000/A6000, A100, H100/H200);
- do not purchase compute without explicit approval.

**Current blocker:** obtaining a valid BF16-capable remote GPU for the canonical Stable-33750 baseline and subsequent SFT.

---

## 11. Post-training strategy

Goal: produce an instruction/domain/safety-tuned model without rerunning pretraining.

Preferred sequence:

1. Run and save Stable-33750 baseline.
2. Build Hindi domain/instruction/safety SFT corpus with strict benchmark leakage isolation.
3. Run contamination scan against HistoryBench-HI v1.
4. Fine-tune from Stable-33750 on remote GPU.
5. Log SFT properly: step, train loss, validation loss where applicable, LR, pre-clip grad norm, clip status, tokens seen, wall time/timestamps, precision, batch/accumulation, seed, GPU, dependency versions, save events.
6. Compare Base vs Instruct on DEV + safety/general capability.
7. Lock final model.
8. Run Hidden-25 once on the final locked model.

NanoChat's native `chat_sft.py` can load a base checkpoint with a fresh optimizer. Parameter-efficient tuning is desirable only if integration is reliable under the deadline; do not spend hours building a custom LoRA bridge if a short native SFT is simpler on adequate hardware.

---

## 12. Optional differentiator: Knowledge-Graph guardrail

After the SFT/evaluation critical path is secure, build a compact Indian History/Cultural Heritage KG demonstration:

- neural model handles language;
- graph supplies explicit factual triples;
- graph can ground prompts before generation;
- generated claims can be marked supported / contradicted / unverifiable afterward.

This is a differentiator, **not a blocker for SFT**.

---

## 13. Required submission/reproducibility artifacts

Bootcamp intake expects, at minimum:

- architecture/model details;
- exact parameter count;
- tokenizer details;
- domain/task/end-user/use case;
- training data sources/size/license/collection description;
- train/validation/test methodology;
- contamination/test-set dedup methodology;
- training corpus/file hash evidence;
- optimizer/LR/schedule/warmup/decay;
- batch/steps/tokens seen;
- GPU type/hours;
- precision;
- random seed;
- SFT method/dataset;
- `manifest.yaml`;
- `train_log.jsonl`;
- checkpoint + tokenizer/config;
- inference script;
- exact dependency versions;
- corpus/hash list;
- model weights access link;
- README/model card/final report/demo/results.

Historical fields that do not exist (especially base-training `grad_norm`) must be marked unavailable rather than invented. New SFT logs should include them correctly.

---

## 14. Critical path to deadline

Current order of operations:

1. **Secure valid BF16-capable remote GPU.**
2. **Stable-33750 DEV baseline.**
3. **Create leakage-safe Hindi history/instruction/safety SFT dataset.**
4. **Contamination check.**
5. **SFT on persistent GPU.**
6. **Base vs Instruct DEV + safety/general evaluation.**
7. **Lock final model and run Hidden-25 once.**
8. **Compact KG demo if time remains.**
9. **Finalize model card, README, report, intake, manifest, logs, inference/demo and submission package.**

No rubric-critical requirement should be silently dropped. Moving fast means parallelizing and simplifying implementation, not fabricating evidence or weakening the evaluation protocol.

---

## 15. Repository / checkpoint hygiene

Checkpoint files should not be committed to Git. Recommended `.gitignore` entries:

```gitignore
base_checkpoints/
chatsft_checkpoints/
chatrl_checkpoints/
model_*.pt
optim_*_rank*.pt
```

Do not blanket-ignore every `.pt` file because the tokenizer uses `token_bytes.pt`.

---

## 16. Recovery protocol for a new ChatGPT thread

If the current conversation becomes too long or a fresh thread is started, begin with:

> Read `CAPSTONE_STATE.md` in `aayush-0131/Hindi_SLM` first. Treat it as the canonical project state. Do not change the frozen benchmark, domain, base checkpoint, hidden-set policy, or paid-compute policy without explicit approval. Then continue from the `Current blocker` and `Critical path` sections.

Then verify the newest repo commit and update this file when a material project decision changes.

---

## 17. Update policy for this file

Update `CAPSTONE_STATE.md` whenever any of these changes:

- canonical checkpoint;
- artifact hash;
- benchmark version/hash;
- compute environment;
- SFT dataset/version/hash;
- chosen fine-tuning method;
- baseline/final scores;
- locked final checkpoint;
- current blocker;
- critical path;
- submission status.

Do not rewrite historical facts to make later results look cleaner. Append or explicitly mark superseded decisions where needed.
