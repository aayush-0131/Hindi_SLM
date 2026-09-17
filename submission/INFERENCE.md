# Frozen HF loading contract

Status: saved AutoConfig/AutoTokenizer/AutoModel loading passed in the parity report. Full dependency lock and evaluator acceptance of custom code are unresolved. This document is a contract reference, not a request to execute a new model run.

The eight frozen files must stay together in the original `hf_export` folder. Verify against evidence/HF_SHA256SUMS.txt. Use the custom Auto mappings with `trust_remote_code=True` and the local export directory, loading only verified trusted code/tokenizer pickle. Do not rename the custom architecture into Llama/GPT-2. No weights are included in this documentation bundle.

For future authorized, non-benchmark loading, the existing contract is:

```python
# LIGHTNING or COLAB — reference only; do not run during finalization.
from transformers import AutoTokenizer, AutoModelForCausalLM
output_dir = "/absolute/path/to/frozen/hf_export"
tokenizer = AutoTokenizer.from_pretrained(
    output_dir, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    output_dir, trust_remote_code=True, local_files_only=True
)
# Explicit runtime-only EOS fallback; do not save_pretrained or alter export.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

This snippet was not executed during documentation preparation. Device placement, model dtype and package versions must follow the validated runtime; the frozen wrapper has explicit FP16/BF16 internals, so blanket recasting is not justified by a config label alone. Respect the declared 256-token context. The chat template supports user/assistant turns; a system-role behavior is not established. Review attention-mask/batching behavior before assuming padded batches work. No inference CLI or demo is certified by this document. Recover the existing validated inference entrypoint rather than executing Hidden again. Never load the 910M model on the 8 GB Mac.

## Historical template

The upstream `inference_config.yaml` remains unchanged to preserve its freeze policy. It was a pending-checkpoint template (2048 context, greedy 128-token defaults); it does not describe the final saved native generation. See [observed settings](final_evaluation_observed.yaml), a new post-hoc record, not a newly locked pre-run protocol.
