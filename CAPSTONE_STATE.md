# Hindi SLM Capstone — Canonical Project State

**Project:** IAIRO PRAMANA SLM++ Bootcamp Capstone  
**Repository:** `aayush-0131/Hindi_SLM`  
**Canonical state timestamp:** 2026-09-15 (after midnight IST)  
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
- **Do not load Stable-33750 on the 8 GB Mac again.** Use the Mac only as control/artifact/documentation machine.

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

The architecture is not a clean GPT-2/Llama clone. Important custom components include value embeddings/gates, residual/x0 scalars, smear/backout terms, RoPE + QK norm and ReLU² MLPs. Do not perform a lossy rename-only conversion into GPT-2 or Llama.

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

The four critical artifacts were uploaded to the private Google Drive folder:

```text
Hindi_SLM_Capstone_Submission
```

### Lightning rescue status — COMPLETE

Lightning organization/project: **Hindi SLM Capstone**  
Studio: **`sudden-teal-clz7`**

Persistent artifact root:

```text
/teamspace/studios/this_studio/artifacts/
```

The large checkpoint upload was resumed over SSH after a browser-upload interruption. The remote partial file was verified as an exact prefix of the local canonical file, then the missing tail was appended. Final remote size is **2,661,366,058 bytes** and final SHA-256 matches the canonical model hash exactly:

```text
1ad56ea60a5e28d6899146a56f63b1833837dbe6f6cef9128bb2bf4a51faa00d
```

Runtime layout:

```text
~/hindi_slm_runtime/tokenizer/tokenizer.pkl
~/hindi_slm_runtime/tokenizer/token_bytes.pt
~/hindi_slm_runtime/ckpt_stable_final/model_033750.pt
~/hindi_slm_runtime/ckpt_stable_final/meta_033750.json
```

A base-loader symlink is also present:

```text
~/hindi_slm_runtime/base_checkpoints/stable -> ~/hindi_slm_runtime/ckpt_stable_final
```

Tokenizer runtime test passed and meta-device architecture reconstruction produced the exact **910,690,922** parameter count.

---

## 4. Tokenizer and corpus facts

- Tokenizer: byte-level BPE
- Vocabulary: **32,768**
- Devanagari-aware tokenizer logic is present in the repository.
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
- include explicit safety-aligned examples in SFT;
- create explicit safety evaluation prompts;
- compare Base vs Instruct safety behavior;
- add inference/demo guardrails if time permits;
- describe base and instruct models separately;
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

During training/model selection, do not change wording, answers, difficulty labels, split, reference answers or grading rubrics. If a correction is unavoidable, create **HistoryBench-HI v1.1** (or later) with fresh hashes and document the change.

### Leakage policy

- Do not feed benchmark questions, reference answers, hidden items, or benchmark-derived prompts into SFT data generation.
- Hidden 25 is for the final locked model only, never iterative model selection.
- Final reporting still requires a contamination audit against all 100 items; merely possessing the hidden set for this audit must not turn it into tuning data.

### Known v1 benchmark limitation discovered after freeze

The DEV MCQ answer key is strongly answer-position imbalanced:

```text
Gold B: 41 / 55
Gold C:  7 / 55
Gold A:  7 / 55
Gold D:  0 / 55
```

Because the benchmark was already frozen before SFT, **do not silently rebalance v1**. Report this limitation explicitly. HistoryBench-HI v1 remains useful for paired Base-vs-Instruct comparison and open-ended reasoning assessment, but raw MCQ accuracy must not be described as if the answer positions were balanced.

---

## 8. Stable-33750 DEV baseline — COMPLETE

HistoryBench DEV baseline successfully ran on Lightning using **Tesla T4 / FP16 engineering fallback**.

Environment:

```text
PyTorch 2.8.0+cu128
CUDA available: True
GPU: Tesla T4
NANOCHAT_DTYPE=float16
```

The checkpoint loaded successfully, MCQ conditional-loss scoring completed, and all 20 open-ended generations completed.

### MCQ results

```text
15 / 55 = 0.272727...
Easy:   6 / 25 = 0.2400
Medium: 9 / 30 = 0.3000
```

By subdomain:

| Subdomain | Accuracy |
|---|---:|
| Ancient India | 3/11 = 27.27% |
| Art, Architecture & Heritage | 4/11 = 36.36% |
| Literature, Society & Cultural Traditions | 1/11 = 9.09% |
| Medieval India | 5/11 = 45.45% |
| Modern India & Freedom Movement | 2/11 = 18.18% |

Model answer-position behavior:

```text
Predictions: B=26, A=11, C=10, D=8
Gold:        B=41, C=7, A=7, D=0
Gold A correct: 0/7
Gold B correct: 15/41
Gold C correct: 0/7
```

The raw 27.27% score must **not** be described as simply "above 25% chance" because the gold labels are highly imbalanced.

### Open-ended baseline behavior

All **20/20** DEV open-ended responses were generated. The first inspected hard examples showed a clear base-model failure mode: the model repeatedly copied/rephrased the prompt instead of answering and entered repetition loops. This is consistent with Stable-33750 being a pretrained completion model rather than an instruction model.

Saved under:

```text
~/Hindi_SLM/results/historybench/base_33750/
```

including `summary.json`, `predictions.jsonl`, `open_ended_for_judging.jsonl`, and `run.log`.

### Precision caveat

Base training was BF16 on H200, while this DEV baseline was FP16 on a T4 because free BF16-capable compute was unavailable without adding a payment method. Treat the current run as the saved engineering baseline and **report the precision mismatch explicitly**. If a BF16-capable GPU becomes available in time, a precision-matched rerun is desirable, but do not purchase compute without approval.

The Hidden-25 has **not** been used for model selection or iterative evaluation.

---

## 9. SFT dataset v1 — CREATED AND VALIDATED

Current SFT seed package:

```text
History_HI_SFT_v1_seed_package.zip
```

Package SHA-256:

```text
9d78b9a9fce9c3a8d7343195ea1c49a848b1c63214c445fb8cfdadef09587f0f
```

Installed in Lightning under:

```text
~/Hindi_SLM/data/history_hi_sft_v1/
```

Dataset sizes:

```text
Train: 275 conversations
Val:    36 conversations
Total: 311 conversations
```

Training data includes:

- Hindi-first domain QA across all five subdomains;
- explicit direct-answer / anti-repetition behavior;
- balanced A/B/C/D MCQ answer-position practice;
- epistemic calibration / do-not-invent behavior;
- generic safety-aligned examples, including refusal/redirection for explicit sexual-content steering and avoidance of insulting/hateful framing.

This dataset was created independently of HistoryBench question wording; hidden benchmark items were not used to generate it.

### DEV contamination scan

Exact 8-gram scan against the frozen DEV-75:

```text
train_rows=275 benchmark_rows=75 n=8
rows_with_overlap=0
STATUS=PASS_NO_EXACT_NGRAM_OVERLAP
```

This is a DEV-only pretraining gate. A final audit against all 100 frozen items is still required for submission reporting.

### Tokenization validation

All 311 conversations rendered through the real NanoChat tokenizer successfully.

Train:

```text
min_tokens: 30
median_tokens: 52
p95_tokens: 128
max_tokens: 134
```

Validation:

```text
min_tokens: 30
median_tokens: 51
p95_tokens: 66
max_tokens: 67
```

Therefore **`max_seq_len=256` is sufficient for every current SFT conversation** and is preferred for the T4 memory test/training attempt instead of wasting VRAM at 2048.

---

## 10. Custom SFT integration status

Custom task module installed:

```text
nanochat/tasks/history_hi_sft.py
```

Custom trainer currently present and syntax-validated:

```text
nanochat/scripts/chat_sft_history.py
```

It is wired to:

```python
HistoryHiSFT(split="train")
HistoryHiSFT(split="val")
```

and no longer uses the default SmolTalk/MMLU/GSM8K mixture for this capstone SFT run.

**Important implementation warning:** the packaged helper `tools/make_history_sft_script.py` generated literal `\n` sequences in the replacement block and produced an invalid trainer on first use. The current `chat_sft_history.py` was manually rebuilt correctly and `python -m py_compile` passed. **Do not rerun `tools/make_history_sft_script.py` until that helper itself is fixed**, or it may regenerate the broken trainer.

---

## 11. Current compute environment and exact blocker

Primary remote environment is Lightning Studio `sudden-teal-clz7`.

Last confirmed GPU:

```text
Tesla T4
PyTorch 2.8.0+cu128
CUDA: True
```

Lightning can auto-sleep after roughly 10 minutes of inactivity, so verify the machine/GPU before every long run.

A 1-step native full-SFT smoke test was prepared with:

```text
model: Stable-33750
precision: FP16 fallback
max_seq_len: 256
device_batch_size: 1
total_batch_size: 256
num_iterations: 1
load_optimizer: 0
chatcore: disabled
```

The test **did not reach model loading or GPU memory allocation** because it failed immediately at import time:

```text
ModuleNotFoundError: No module named 'wandb'
```

### CURRENT BLOCKER

Install `wandb` in the Lightning cloudspace environment, verify import, then rerun the exact 1-step SFT smoke test. The next command should be:

```bash
python -m pip install -q wandb && python -c "import wandb; print('WANDB_IMPORT=PASS', wandb.__version__)"
```

Only after that rerun the 1-step SFT memory smoke. The outcome decides strategy:

- **If full native SFT fits T4 16 GB:** proceed with short native SFT using `max_seq_len=256` and carefully chosen steps/LRs.
- **If CUDA OOM:** immediately switch to a memory-efficient tuning approach; do not spend hours repeatedly trying full SFT settings.

The previous plan to require a BF16-capable GPU before doing anything is **superseded by deadline reality**. T4/FP16 is now the available engineering fallback. Precision mismatch must be disclosed; paid compute still requires explicit approval.

---

## 12. New mandatory Hugging Face submission requirement

A new bootcamp instruction arrived after baseline/SFT setup:

> Final model submission must be loadable using `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()` without a manual conversion step at evaluation time. The submitted checkpoint files and HF weights link must point to the same final `output_dir`.

Required final-export checks include:

- wrap the custom NanoChat architecture as a Hugging Face `PreTrainedModel` / `PretrainedConfig` implementation rather than performing a lossy GPT-2/Llama rename;
- load and map final trained state dict;
- `model.save_pretrained(output_dir)`;
- `tokenizer.save_pretrained(output_dir)`;
- set tokenizer `pad_token` / `pad_token_id` explicitly (or documented EOS fallback if required);
- ensure `config.json` `vocab_size` is exactly **32768**;
- verify a clean reload from the final folder with `AutoModelForCausalLM.from_pretrained(...)` and `AutoTokenizer.from_pretrained(...)`;
- preferably use safetensors for final weights if practical;
- run native-NanoChat vs HF-wrapper parity checks before accepting the conversion.

Because the architecture contains custom modules/features, do **not** assume GPT-2/Llama key renaming is sufficient. Build a faithful wrapper.

One contract detail remains to verify with the organizers/evaluator: whether custom model code may require `trust_remote_code=True`. Their instruction explicitly allows wrapping a custom architecture as a `PreTrainedModel`, so a custom HF implementation is the intended route, but the final clean-load command must match the evaluator's actual contract.

HF export is now **rubric-critical**, not optional. Prototype the wrapper before the final submission window, but do not let it block the immediate SFT memory test.

---

## 13. Mac incident and local-compute decision

Machine: Apple Silicon MacBook Air with **8 GB unified memory**.

The first native MPS smoke test kernel-panicked/restarted after reaching model load. The panic report showed watchdog timeout and low swap headroom. NanoChat converts BF16 weights to FP32 on CPU/MPS, which is unsafe for this machine at ~910M parameters.

**Decision:** never load Stable-33750 on the Mac again.

Use the Mac for:

- SSH/control/browser;
- artifact transfer;
- documentation;
- Git/Drive operations;
- lightweight data preparation.

Use remote NVIDIA GPU for model loading, SFT and inference/evaluation.

---

## 14. Post-training evaluation strategy

At minimum:

1. Stable-33750 Base on DEV — **done**.
2. Final instruction/domain/safety-tuned model on DEV.
3. Compare Base vs Instruct safety behavior.
4. Run required general/instruction capability checks.
5. Lock the final model.
6. Run Hidden-25 **once** on the locked final model only.
7. Judge open-ended outputs using the frozen rubric; do not derive scores from vibes.
8. Document HistoryBench answer-position imbalance and T4/FP16 precision caveat.

Do not use hidden performance to select checkpoints.

---

## 15. Required submission/reproducibility artifacts

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
- final HF-loadable checkpoint + tokenizer/config in one folder;
- inference script;
- exact dependency versions;
- corpus/hash list;
- model weights access link;
- README/model card/final report/demo/results.

Historical fields that do not exist (especially base-training `grad_norm`) must be marked unavailable rather than invented. New SFT logs should include the requested fields wherever the trainer makes them available.

---

## 16. Critical path to deadline

Current order of operations:

1. **Install `wandb` in Lightning.**
2. **Rerun the 1-step T4/FP16 native SFT memory smoke** at seq len 256, batch 1.
3. **Choose fine-tuning strategy from measured memory result:** native full SFT if it fits; otherwise memory-efficient tuning immediately.
4. **Run SFT** on the 275-row leakage-controlled Hindi history/instruction/safety dataset.
5. **Evaluate Instruct on DEV + safety/general checks** and compare with saved Base baseline.
6. **Prototype/finalize faithful HF `PreTrainedModel` + tokenizer export** and verify AutoModel/AutoTokenizer clean reload and parity.
7. **Lock the final model.**
8. **Run Hidden-25 once** on the final locked model.
9. **Run final all-100 contamination audit for reporting.**
10. **Compact KG guardrail/demo only if time remains.**
11. **Finalize model card, README, report, intake, manifest, logs, inference/demo, HF weights link and submission package.**

No rubric-critical requirement should be silently dropped. Moving fast means simplifying implementation and parallelizing CPU/documentation work, not fabricating evidence or weakening the evaluation protocol.

---

## 17. Optional differentiator: Knowledge-Graph guardrail

After SFT/evaluation/HF-export critical path is secure, a compact Indian History/Cultural Heritage KG can be used as a differentiator:

- neural model handles language;
- graph supplies explicit factual triples;
- graph can ground prompts before generation;
- generated claims can be marked supported / contradicted / unverifiable afterward.

This is a differentiator, **not a blocker**.

---

## 18. Repository / checkpoint hygiene

Checkpoint files should not be committed to Git. Recommended `.gitignore` entries:

```gitignore
base_checkpoints/
chatsft_checkpoints/
chatrl_checkpoints/
model_*.pt
optim_*_rank*.pt
```

Do not blanket-ignore every `.pt` file because the tokenizer uses `token_bytes.pt`.

Do not commit the Hidden-25 benchmark file.

---

## 19. Recovery protocol for a new ChatGPT thread

The current long chat became UI-heavy/laggy and should be replaced with a fresh chat inside the **SLM++ Bootcamp Capstone Project**.

Start the new chat with:

> Read `CAPSTONE_STATE.md` in `aayush-0131/Hindi_SLM` first and treat it as the canonical state of my IAIRO SLM++ capstone. Also use the relevant context from this project. Continue exactly from the current blocker without redoing completed work. Do not change the frozen benchmark, use the hidden set for model selection, purchase compute, publish weights, or change the model/domain strategy without my approval.

Then continue from **Section 11 — CURRENT BLOCKER**. The immediate next action is installing `wandb`, then rerunning the 1-step SFT memory smoke.

---

## 20. Update policy for this file

Update `CAPSTONE_STATE.md` whenever any of these changes:

- canonical checkpoint;
- artifact hash;
- benchmark version/hash;
- compute environment;
- SFT dataset/version/hash;
- chosen fine-tuning method;
- baseline/final scores;
- HF export/load status;
- locked final checkpoint;
- current blocker;
- critical path;
- submission status.

Do not rewrite historical facts to make later results look cleaner. Append or explicitly mark superseded decisions where needed.
