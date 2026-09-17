# Submission completion checklist

## Complete or recorded

- [x] Official team identity recorded: **Team 2 — Qwenspiracy**.
- [x] C2 step120 locked; native checkpoint and metadata hashes verified in Drive.
- [x] HF export located on Lightning; all eight frozen file hashes verified.
- [x] Frozen HF ZIP reconstructed locally, SHA256-verified (`92ca56b23cf439e76ea6d6cde44356e72a69d966f129177f6f8ea0cb47bdd516`), `unzip -t` passed, and Drive backup completed.
- [x] Saved exact tensor/diagnostic parity evidence recovered.
- [x] Canonical DEV result recovered; one-shot Hidden archive verified.
- [x] All-100 exact 8-token SFT audit complete; report checksum saved; semantic/pretraining-overlap limitation disclosed.
- [x] C2-specific trainer recovered; working/local/upstream bytes agree and settings match the log.
- [x] C2 raw training log preserved; retrospective structured log prepared separately as `c2_train_log_derived.jsonl`.
- [x] Current Lightning dependency snapshot captured and explicitly labeled current rather than historical.
- [x] Final evaluator-ready Hugging Face package created from the locked C2 weights with explicit EOS-as-pad fallback.
- [x] `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` reloaded successfully from the Hugging Face Hub.
- [x] `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` reloaded successfully from the Hugging Face Hub.
- [x] Public model repository established: `https://huggingface.co/Shadow0131/hindi-slm-c2-step120`.
- [x] Anonymous access verified with HTTP 200.
- [x] Verified Hub commit recorded: `5d5821284962ece831b8f62ea2dddef89bb68101`.
- [x] Final intake PPT updated with the public HF checkpoint and recovered training/reproducibility evidence.
- [x] Git ancestry/tree differences inspected; upstream includes the C2-specific trainer unchanged.
- [x] Final documentation branch reconciled after HF publication; manifest updated with public checkpoint, C2 schedule/log facts, current dependency capture, and disclosed gaps.

## Remaining before actual organizer submission

- [ ] Recover the item-level Hidden open-ended scoring ledger/rubric provenance for the reported 14/80 if it still exists. Until then keep the score explicitly provisional; do not regenerate Hidden responses.
- [ ] If available, add the exact source revision/license inventory and complete pretraining corpus file-hash list. Current records are incomplete; do not infer missing licenses/hashes.
- [ ] Historical base gradient norms, pre-clip norms/clip flags, historical dependency lock and pretraining random seed were not recovered. Keep them marked unavailable rather than fabricating values.
- [ ] Full multi-branch/SFT/evaluation GPU-hour total was not reconstructed. Keep base stable-run time (~30.25h) and C2 logged optimizer-step time (0.13m) clearly scoped.
- [ ] Confirm whether the organizer requires any additional inference entrypoint beyond the tested Hugging Face Auto* loading path with `trust_remote_code=True`.
- [ ] Confirm current organizer deadline/submission channel from the actual active instructions.
- [ ] Submit the final intake/presentation and any requested artifact links/files.
- [ ] Record the final delivery commit separately after all final documentation edits (avoid self-referential commit hashes).
- [ ] Save the submission receipt/status only after the actual organizer submission succeeds.

## Canonical final checkpoint access

- Team: **Team 2 — Qwenspiracy**
- Hugging Face: `https://huggingface.co/Shadow0131/hindi-slm-c2-step120`
- Verified Hub revision: `5d5821284962ece831b8f62ea2dddef89bb68101`
- Frozen archive: `C2_step120_HF_frozen.zip`
- Frozen archive SHA256: `92ca56b23cf439e76ea6d6cde44356e72a69d966f129177f6f8ea0cb47bdd516`

## Resume after a reset

**COLAB:** run the existing START HERE recovery cell only; reconnect Drive and resume the outstanding documentation/package task. Do not Run all. Missing `/content` source files do not invalidate Drive archives.

**LIGHTNING:** documentation/package checks do not require retraining. Preserve the frozen HF archive bytes exactly; the evaluator-ready Hub package is a separate compatibility distribution of the same weights.

**MAC:** control/download/upload/Git/document review only. No model load required.
