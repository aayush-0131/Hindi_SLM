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
    """Phase A. Returns (curated_tasks: list[(task_spec, num_choices)], dropped: list[(task_spec, reason)])."""
    hf_repo = hf_repo or config.PRIMARY_BASELINE["hf_repo"]
    curated, dropped = [], []
    for task_spec, num_choices in config.CANDIDATE_TASKS:
        print(f"[Phase A] smoke-testing {task_spec} (limit={config.SMOKE_TEST_LIMIT})...")
        success, results, output = run_task(hf_repo, task_spec, max_samples=config.SMOKE_TEST_LIMIT)
        if not success:
            dropped.append((task_spec, f"failed to run: {output[-500:]}"))
            print(f"  DROPPED -- did not run successfully.")
            continue
        metrics = extract_task_metrics(results, task_spec)
        if not metrics:
            dropped.append((task_spec, "ran but no metrics found under this exact task key"))
            print(f"  DROPPED -- no metrics found under key '{task_spec}' in results JSON.")
            continue
        curated.append((task_spec, num_choices))
        print(f"  KEPT -- resolved and produced metrics.")
    return curated, dropped


def run_gate2(hf_repo=None):
    hf_repo = hf_repo or config.PRIMARY_BASELINE["hf_repo"]

    curated, dropped = curate_suite(hf_repo)
    print(f"\n[Phase A summary] {len(curated)} task(s) curated, {len(dropped)} dropped.")
    for task_spec, reason in dropped:
        print(f"  dropped: {task_spec} -- {reason}")

    if not curated:
        return {"passed": False, "reason": "no tasks survived Phase A curation", "dropped": dropped}

    print(f"\n[Phase B] Running curated suite for real against {hf_repo}...")
    task_results = {}
    for task_spec, num_choices in curated:
        success, results, output = run_task(hf_repo, task_spec, max_samples=None)
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
        task_results[task_spec] = {"metric_name": metric_name, "score": score, "non_random": non_random}
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

    print("\nGATE 2 PASSED -- every curated task produced a non-random score against the primary baseline.")
    return {"passed": True, "task_results": task_results, "curated": curated, "dropped": dropped}
