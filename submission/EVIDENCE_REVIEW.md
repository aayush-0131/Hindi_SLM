# Evidence review and unresolved fields

## Evidence actually inspected

Input ZIP SHA256: `dbceb3699b2331cd43ea6470c2e371ec705b1ac7bae01e150ed9a46d08850456`. Every included file was verified against EVIDENCE_INDEX.json before use. No AGENTS.md was present in the supplied bundle. The original index is preserved under evidence/.

Native/HF weights were not uploaded here or loaded. Their successful checks are recorded from user-run batches; the original parity report and index were inspected locally. DEV/Hidden summaries, hashes, metadata and audit status were supplied as terminal output in this conversation. These are accurately transcribed in the manifest, not represented as newly downloaded original files.

## Material gaps

1. **C2 trainer recovered:** `nanochat/scripts/chat_sft_history_partial_c2.py` matches byte-for-byte in Lightning working files, local commit and upstream. SHA256 `1ddd87c92ea982aaadbb04d0510f5836e2a8f085c007db623357b641c435144f`. It freezes all parameters then unfreezes the final four blocks and LM head; AdamW betas=(0.9,0.95), eps=1e-8, weight_decay=0.01, foreach=False. Data seed is 42 with per-epoch shuffling. This resolves the generic-trainer mismatch; historical execution identity is inferred from matching source/log/metadata, not a contemporaneous source hash.
2. **Dependencies:** supplied nanochat/pyproject.toml pins torch 2.9.1, while canonical evaluation used 2.8.0+cu128. deploy/requirements.txt is unpinned. Do not install the generic project dependencies over the canonical runtime or call them its lockfile. Recover the actual Transformers/tokenizer/safetensors and other package versions from contemporaneous records. A fresh environment capture must be labeled current, not historical.
3. **Padding:** frozen tokenizer/model configs omit an explicit pad token. Document an in-memory EOS fallback for inference; do not rewrite/hash-replace the frozen export. Whether this meets the evaluator contract remains to confirm.
4. **Context:** final config declares 256, not base 2048. Native generation max_new_tokens=256 does not establish HF support for arbitrary longer prompts/generation.
5. **Judging:** reported Hidden 14/80 lacks recovered scoring ledger; NOT_YET_JUDGED in generation summaries is not evidence the separate judgment never happened. Preserve both facts; recover rubric, judge, item scores and provenance.
6. **Git reconciled by tree comparison:** local commit 44cd791 is patch-equivalent upstream; local ff5a74a is not patch-equivalent, but the final tree diff shows only CAPSTONE_STATE.md modified and 13 paths added upstream. No source path is local-only or source-modified. Use an isolated integration worktree from upstream 05d66d6 and retain the original checkout/untracked files. Existing upstream pretraining log/schema remain unchanged; derived C2 records have distinct filenames. No reset, clean or force-push.
7. **Delivery:** no verified recipient-accessible model link, independent complete HF backup, exact data/license inventory, corpus hash list or confirmed deadline extension is present. Historical base gradient norms are unavailable; do not manufacture them.

## Derived training log

`c2_train_log_derived.jsonl` is parsed retrospectively from the original C2 console log, with source line numbers and original SHA256. It is not an original structured training log. The accompanying JSON schema is newly authored for this derived representation, distinct from the recovered upstream schema, which requires timestamps unavailable in per-step C2 console lines. Fields retain printed precision; loss is stored as logged loss because exact historical smoothing implementation is not recovered. Missing gradient norms and true supervised-token counts remain null. MFU is stored as an unavailable estimate. Nominal batch capacity is not true unmasked token count.

## Source retention

Historical Lightning state is retained under history/ with an explicit superseded banner. Original private evaluation archives remain on Drive unchanged. Original HF export is unchanged. No prediction text, hidden questions, model weights, optimizer state, or tokenizer pickle is included in this documentation package.

## Newly recovered current environment

See evidence/CURRENT_LIGHTNING_ENVIRONMENT.json. These package versions describe the capture time; they are not retroactively certified historical pins. The generic PyTorch 2.9.1 pin remains inconsistent with canonical evaluation 2.8.0+cu128.

## Existing upstream records retained

`submission/train_log.jsonl` and its original schema remain byte-identical to upstream. The former contains two reconstructed pretraining records, including explicitly labeled placeholder reconstruction timestamps. The derived C2 log and newly authored schema use separate filenames. The upstream `inference_config.yaml` is also retained unchanged: it is a pre-final template, not the settings used for final evaluation. Use the new observed evaluation record to interpret final results; do not silently mutate a file with a freeze policy.
