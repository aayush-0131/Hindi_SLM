"""
Gate 2 (Requirement 5): validates the evaluation harness works and produces
non-random scores against baselines, BEFORE any pretraining GPU time is
spent. Two phases:

  Phase A (suite curation): smoke-test every CANDIDATE task at a tiny sample
  count. A task that fails to even load here is dropped from the suite with
  the reason logged -- this is a scoping decision made *before* Gate 2's
  formal run, not a silent skip during it (research.md flags two open
  lighteval issues where blog-post-claimed tasks don't actually resolve for
  some languages, so this step is expected to matter, not just defensive).

  Phase B (the actual gate): runs the curated suite for real against the
  primary baseline (Goldfish-Hindi, same causal-LM architecture family as
  the target model). Per Requirement 5.5, ANY task in the curated suite
  that fails or produces a degenerate/random-chance score here fails Gate 2
  outright -- no further dropping once Phase B starts.
"""

from eval_harness import config
from eval_harness.lighteval_runner import run_task, extract_task_metrics

METRIC_PRIORITY = ["acc_norm", "acc", "em", "f1"]
RANDOM_MARGIN = 1.15  # score must exceed chance-baseline by at least this factor

# Cache of hf_repo -> resolved local patched-tokenizer dir (or None if no
# patch is configured for that repo), so a full suite run only pays for
# tokenizer_patch's snapshot_download/patch step once, not once per task.
_TOKENIZER_PATH_CACHE = {}


def _resolve_tokenizer_path(hf_repo):
    """Return a local patched-tokenizer directory for hf_repo, or None if no
    patch is configured. See tokenizer_patch.py and PRIMARY_BASELINE's
    "tokenizer_patch" key for why this exists (a transformers-internal bug
    in Goldfish-Hindi's tokenizer, unrelated to lighteval or our own code)."""
    if hf_repo in _TOKENIZER_PATH_CACHE:
        return _TOKENIZER_PATH_CACHE[hf_repo]
    overrides = None
    if hf_repo == config.PRIMARY_BASELINE.get("hf_repo"):
        overrides = config.PRIMARY_BASELINE.get("tokenizer_patch")
    path = None
    if overrides:
        from eval_harness.tokenizer_patch import get_patched_tokenizer_dir
        path = get_patched_tokenizer_dir(hf_repo, overrides)
    _TOKENIZER_PATH_CACHE[hf_repo] = path
    return path


def _primary_metric(metrics_dict):
    for name in METRIC_PRIORITY:
        if name in metrics_dict:
            return name, metrics_dict[name]
    for k, v in metrics_dict.items():
        if not k.endswith("_stderr"):
            return k, v
    return None, None


def _is_non_random(score, num_choices):
    if score is None:
        return False
    chance = 1.0 / num_choices
    return score > chance * RANDOM_MARGIN


def curate_suite(hf_repo=None):
    """Phase A.

    Returns (curated, non_gating, dropped):
      curated    -- [(task_spec, num_choices)] that Phase B gates on strictly
      non_gating -- [(task_spec, num_choices, reason)] that Phase B still RUNS
                    and RECORDS, but is not allowed to fail the gate
      dropped    -- [(task_spec, reason)] that did not resolve at all

    The `non_gating` bucket exists for tasks whose chance baseline is too high
    to discriminate against a size-matched baseline -- see
    config.MIN_DISCRIMINATIVE_CHOICES. Putting that decision here, in curation,
    is what keeps Phase B strictly all-or-nothing (Requirement 5.5) instead of
    silently skipping a task mid-gate.
    """
    hf_repo = hf_repo or config.PRIMARY_BASELINE["hf_repo"]
    curated, non_gating, dropped = [], [], []
    min_choices = getattr(config, "MIN_DISCRIMINATIVE_CHOICES", 0)
    gate_on_binary = getattr(config, "GATE_ON_BINARY_TASKS", True)

    tokenizer_path = _resolve_tokenizer_path(hf_repo)
    for task_spec, num_choices in config.CANDIDATE_TASKS:
        print(f"[Phase A] smoke-testing {task_spec} (limit={config.SMOKE_TEST_LIMIT})...")
        success, results, output = run_task(hf_repo, task_spec,
                                            max_samples=config.SMOKE_TEST_LIMIT,
                                            tokenizer_path=tokenizer_path)
        if not success:
            dropped.append((task_spec, f"failed to run: {output[-500:]}"))
            print(f"  DROPPED -- did not run successfully.")
            continue
        metrics = extract_task_metrics(results, task_spec)
        if not metrics:
            dropped.append((task_spec, "ran but no metrics found under this exact task key"))
            print(f"  DROPPED -- no metrics found under key '{task_spec}' in results JSON.")
            continue
        if not gate_on_binary and num_choices < min_choices:
            reason = (
                f"{num_choices}-choice task: chance is {1 / num_choices:.3f}, so "
                f"_is_non_random demands >{1 / num_choices * RANDOM_MARGIN:.3f}. "
                f"A 125M size-matched baseline scoring near chance here is the "
                f"expected outcome, not a harness defect, so this task is "
                f"measured but not gated on.")
            non_gating.append((task_spec, num_choices, reason))
            print(f"  KEPT (measured, NOT gating) -- {reason}")
            continue
        curated.append((task_spec, num_choices))
        print(f"  KEPT -- resolved and produced metrics.")
    return curated, non_gating, dropped


def run_gate2(hf_repo=None):
    hf_repo = hf_repo or config.PRIMARY_BASELINE["hf_repo"]

    curated, non_gating, dropped = curate_suite(hf_repo)
    print(f"\n[Phase A summary] {len(curated)} gating, "
          f"{len(non_gating)} measured-only, {len(dropped)} dropped.")
    for task_spec, reason in dropped:
        print(f"  dropped: {task_spec} -- {reason}")
    for task_spec, _, reason in non_gating:
        print(f"  measured-only: {task_spec} -- {reason}")

    if not curated:
        return {"passed": False,
                "reason": "no tasks survived Phase A curation as gating tasks",
                "dropped": dropped,
                "non_gating": [(t, r) for t, _, r in non_gating]}

    print(f"\n[Phase B] Running curated suite for real against {hf_repo}...")
    tokenizer_path = _resolve_tokenizer_path(hf_repo)
    task_results = {}
    for task_spec, num_choices in curated:
        success, results, output = run_task(hf_repo, task_spec, max_samples=None,
                                            tokenizer_path=tokenizer_path)
        if not success:
            print(f"  FAILED: {task_spec} did not run in Phase B (was fine in Phase A -- flaky?).")
            return {
                "passed": False,
                "reason": f"{task_spec} failed in Phase B after passing Phase A (Requirement 5.5)",
                "output": output[-1000:],
                "curated": curated,
                "dropped": dropped,
            }
        metrics = extract_task_metrics(results, task_spec)
        metric_name, score = _primary_metric(metrics) if metrics else (None, None)
        non_random = _is_non_random(score, num_choices)
        task_results[task_spec] = {"metric_name": metric_name, "score": score,
                                   "non_random": non_random, "gating": True}
        status = "OK" if non_random else "DEGENERATE"
        print(f"  {task_spec}: {metric_name}={score} (chance={1/num_choices:.3f}) -> {status}")
        if not non_random:
            return {
                "passed": False,
                "reason": f"{task_spec} produced a random/degenerate score (Requirement 5.5)",
                "task_results": task_results,
                "curated": curated,
                "dropped": dropped,
            }

    # Measured-only tasks run AFTER the gating set, so a slow or flaky
    # non-gating task cannot delay or obscure the gate's own verdict. Their
    # scores are recorded for the Requirement 11.2 comparison table.
    for task_spec, num_choices, reason in non_gating:
        success, results, output = run_task(hf_repo, task_spec, max_samples=None,
                                            tokenizer_path=tokenizer_path)
        if not success:
            print(f"  {task_spec}: did not run (measured-only, not gating)")
            task_results[task_spec] = {"metric_name": None, "score": None,
                                       "non_random": None, "gating": False,
                                       "note": "failed to run in Phase B"}
            continue
        metrics = extract_task_metrics(results, task_spec)
        metric_name, score = _primary_metric(metrics) if metrics else (None, None)
        task_results[task_spec] = {
            "metric_name": metric_name, "score": score,
            "non_random": _is_non_random(score, num_choices),
            "gating": False, "note": reason,
        }
        print(f"  {task_spec}: {metric_name}={score} "
              f"(chance={1 / num_choices:.3f}) -> measured only, not gating")

    print(f"\nGATE 2 PASSED -- all {len(curated)} gating task(s) produced a "
          f"non-random score against the primary baseline.")
    return {"passed": True, "task_results": task_results, "curated": curated,
            "non_gating": [(t, r) for t, _, r in non_gating],
            "dropped": dropped}
