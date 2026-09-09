"""
Requirement 8.1: evaluate our own trained checkpoints (Decay A / Decay B)
against the same Gate 2 benchmark items already used to score the
Goldfish-Hindi baseline (research.md Bug 15's CANDIDATE_TASKS).

WHY THIS ISN'T JUST `lighteval_runner.run_task()` AGAIN
--------------------------------------------------------
`run_task()` builds `model_args = f"model_name={hf_repo}"` and hands it to
lighteval's `TransformersModel`, which loads via HuggingFace's
`AutoModelForCausalLM`. Our checkpoint is nanochat's own custom architecture
(per-layer `resid_lambdas`/`x0_lambdas`, the "smear" bigram gate, "backout",
ResFormer-style `value_embeds`) -- none of these have an HF `transformers`
equivalent, so there is no `model_name=` string that could ever load it.
Converting to HF format would mean re-implementing this entire architecture
as a new `PreTrainedModel`, which is more work than this bridge and far
riskier (a subtle mismatch in any of those mechanisms silently produces
wrong logits, with no error to catch it).

Instead, this reuses nanochat's own already-correct, already-tested
multiple-choice scoring path (`nanochat/core_eval.py`, originally built for
DCLM's CORE benchmark) and just swaps in Gate 2's actual Hindi benchmark
items in place of CORE's own task data. Only the DATA SOURCE changes;
tokenization, batching, and the forward-pass/loss computation are exactly
nanochat's own code, not a reimplementation -- so this does not carry the
same "did I get the model's forward pass right" risk an HF conversion would.

Getting Gate 2's exact benchmark items (not a re-derivation): each task spec
in `eval_harness.config.CANDIDATE_TASKS` (e.g. "community_arc_hin_mcf:easy|0")
is resolved via `lighteval.tasks.registry.Registry(...).load_tasks()`, then
`.eval_docs()` returns the same `Doc` objects (`.query`, `.choices`,
`.gold_index`) that were used to score Goldfish-Hindi -- so the accuracy
numbers this produces are directly comparable to the ones already recorded
for the baseline.

CONFIRMED BY SMOKE TEST (2026-09-09): a fixed `" "` delimiter double-spaces
against these tasks' `choices`, which already carry a leading space (the
standard MCF "score just the answer letter" convention, e.g. `" A"`/`" B"`) --
smoke-test output showed `"...:  A"` (two spaces) before the fix below. Now
picked per-item: empty string if the first choice already starts with
whitespace, else a single space -- mirrors lighteval's own
`_check_continuations_start_space` guard for exactly this reason. Still use
`--smoke-test` to eyeball formatting before trusting a full run.
"""

import torch

from lighteval.tasks.registry import Registry

from nanochat.core_eval import render_prompts_mc, batch_sequences_mc, stack_sequences, forward_model

from eval_harness import config
from eval_harness.gate_report import _is_non_random


def _load_task_docs(task_spec):
    registry = Registry(tasks=task_spec, load_multilingual=True)
    task = registry.load_tasks()[task_spec]
    return task.eval_docs()


def _continuation_delimiter(choices):
    """Empty if choices already start with whitespace (the standard MCF
    "score just the answer letter" convention, e.g. " A"/" B"), else a single
    space -- avoids double-spacing, which tokenizes differently from
    anything the model saw during training. Mirrors lighteval's own
    `_check_continuations_start_space` guard."""
    return "" if choices and choices[0][:1].isspace() else " "


def _score_doc(model, tokenizer, device, doc):
    """Returns (predicted_index, gold_index) for one Doc, using nanochat's own
    multiple-choice scoring convention (lowest mean loss over the
    continuation span wins) -- mirrors core_eval.py's evaluate_example."""
    gold = doc.gold_index if isinstance(doc.gold_index, int) else doc.gold_index[0]
    item = {"query": doc.query, "choices": doc.choices}
    delimiter = _continuation_delimiter(doc.choices)
    prompts = render_prompts_mc(item, delimiter, fewshot_examples=None)
    tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)

    max_seq_len = getattr(model, "max_seq_len", None) or getattr(getattr(model, "config", None), "sequence_len", None)
    if max_seq_len is not None:
        cropped_tokens, cropped_starts, cropped_ends = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                cut = len(t) - max_seq_len
                cropped_tokens.append(t[-max_seq_len:])
                cropped_starts.append(s - cut)
                cropped_ends.append(e - cut)
            else:
                cropped_tokens.append(t)
                cropped_starts.append(s)
                cropped_ends.append(e)
        tokens, start_idxs, end_idxs = cropped_tokens, cropped_starts, cropped_ends

    pad_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_id).to(device)
    losses, _ = forward_model(model, input_ids)
    mean_losses = [
        losses[i, start_idxs[i] - 1:end_idxs[i] - 1].mean().item()
        for i in range(len(tokens))
    ]
    pred = mean_losses.index(min(mean_losses))
    return pred, gold


def evaluate_checkpoint(model, tokenizer, device, tasks=None, smoke_test=False):
    """Runs every (task_spec, num_choices) in `tasks` (default:
    config.CANDIDATE_TASKS) against an already-loaded nanochat model.
    Returns {task_spec: {"acc": float, "n": int, "verdict": "OK"|"DEGENERATE"}}.
    """
    tasks = tasks if tasks is not None else config.CANDIDATE_TASKS
    results = {}
    for task_spec, num_choices in tasks:
        docs = _load_task_docs(task_spec)
        if smoke_test:
            docs = docs[:5]
            item = {"query": docs[0].query, "choices": docs[0].choices}
            delimiter = _continuation_delimiter(docs[0].choices)
            sample_prompt = render_prompts_mc(item, delimiter)[0]
            print(f"[smoke] {task_spec} sample rendered prompt "
                  f"(delimiter={delimiter!r}):\n  {sample_prompt!r}")

        correct = 0
        for doc in docs:
            pred, gold = _score_doc(model, tokenizer, device, doc)
            correct += int(pred == gold)
        n = len(docs)
        acc = correct / n if n else 0.0
        verdict = "OK" if _is_non_random(acc, num_choices) else "DEGENERATE"
        results[task_spec] = {"acc": acc, "n": n, "correct": correct, "verdict": verdict}
        print(f"{task_spec}: acc={acc:.4f} ({correct}/{n})  "
              f"chance={1/num_choices:.3f}  -> {verdict}")
    return results
