"""
Works around a `transformers` tokenizer-loading bug that is specific to
Goldfish-Hindi's shipped tokenizer files, not to lighteval or our own code.

`goldfish-models/hin_deva_1000mb`'s tokenizer_config.json / special_tokens_map.json
declare cls_token="[CLS]" (id 50000) and sep_token="[SEP]" (id 50001) -- real
tokens, but ADDED tokens sitting beyond the base SentencePiece vocab, not base
vocab entries. `transformers.models.albert.tokenization_albert.AlbertTokenizer.
__init__` builds a BERT-style pair-encoding post-processor
(`self._tokenizer.post_processor = processors.TemplateProcessing(...,
special_tokens=[(self.cls_token, self.cls_token_id), ...])`) at construction
time, and at that exact point the added tokens have not yet been registered
against the underlying Rust tokenizer, so `self.cls_token_id` / `self.
sep_token_id` resolve to None. The Rust binding then rejects
`(token_str, None)` with:

    TypeError: Expected Union[Tuple[str, int], Tuple[int, str], dict]

on EVERY load of this repo -- confirmed via `AutoTokenizer.from_pretrained`
directly, independent of lighteval, and independent of `use_fast`.

Confirmed-safe workaround: override cls_token/sep_token at load time to any
BASE-vocab token ("<s>"/"</s>", ids 1/2, resolvable immediately). This is
harmless for evaluation: lighteval only uses this tokenizer for causal-LM
loglikelihood multiple-choice scoring, never the BERT pair-encoding template
cls/sep feed. Verified post-load and unaffected by the override: bos_token_id
(50000), eos_token_id (50001), pad_token_id, unk_token_id, and plain text
encode/decode (correct "[CLS] ... [SEP]" wrapping) are all still correct --
only the cls_token/sep_token attributes themselves change, and nothing in the
eval path reads them.

Implementation note: lighteval's `TransformersModelConfig` resolves the
tokenizer independently of the model weights --
`models/transformers/transformers_model.py`'s `_create_auto_tokenizer` does
`tokenizer_name = self.config.tokenizer or self.config.model_name` -- so we
can pass `tokenizer=<local patched dir>` as an extra `model_args` key while
`model_name=<hf_repo>` still pulls the real weights from the Hub, unpatched.
"""

import json
import os
import shutil

# Only override these two keys -- bos/eos/pad/unk are already correct and
# actively used elsewhere (e.g. lighteval sets pad_token = eos_token), so
# touching them would risk breaking something that currently works.
DEFAULT_OVERRIDES = {"cls_token": "<s>", "sep_token": "</s>"}

CACHE_ROOT = os.path.expanduser("~/.cache/hindi_lm_tokenizer_patches")

PATCHED_FILES = ("tokenizer_config.json", "special_tokens_map.json")


def get_patched_tokenizer_dir(hf_repo, overrides=None, force=False):
    """Materialize a local, working copy of hf_repo's tokenizer files.

    Cached under CACHE_ROOT by repo name -- safe and cheap to call on every
    run_task() invocation; only does real work (snapshot download + patch)
    once per repo, ever, unless force=True.

    Returns the local directory path, suitable for lighteval's `tokenizer=`
    model_args key.
    """
    overrides = overrides or DEFAULT_OVERRIDES
    local_dir = os.path.join(CACHE_ROOT, hf_repo.replace("/", "__"))
    marker = os.path.join(local_dir, ".patched")
    if os.path.exists(marker) and not force:
        return local_dir

    # Imported lazily: callers that never need a patch (e.g. tasks that don't
    # touch this baseline) shouldn't need huggingface_hub importable.
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(hf_repo)
    os.makedirs(local_dir, exist_ok=True)
    for name in os.listdir(snapshot):
        src = os.path.join(snapshot, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(local_dir, name))
        elif os.path.isdir(src):
            # Snapshot dirs from the HF cache are usually flat for tokenizer
            # files, but be defensive rather than silently skip a subdir.
            shutil.copytree(src, os.path.join(local_dir, name), dirs_exist_ok=True)

    patched_any = False
    for fname in PATCHED_FILES:
        path = os.path.join(local_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        data.update(overrides)
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        patched_any = True

    if not patched_any:
        raise RuntimeError(
            f"get_patched_tokenizer_dir({hf_repo!r}): none of {PATCHED_FILES} "
            f"were found in the downloaded snapshot -- this repo's tokenizer "
            f"files may not need this patch, or may be laid out differently. "
            f"Do not blindly trust the cache; investigate before retrying."
        )

    with open(marker, "w") as f:
        f.write(f"patched {list(overrides.keys())} in {PATCHED_FILES}: "
                 f"{json.dumps(overrides)}")
    return local_dir
