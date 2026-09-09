"""
Thin wrapper around lighteval's CLI (confirmed pattern from lighteval's own
docs: `lighteval accelerate "model_name=<repo>" "<task>|<n_shots>"`). Runs
one task per subprocess call rather than batching multiple tasks into one
call -- simpler to isolate failures per task, and sidesteps uncertainty
about the exact multi-task argument syntax.

The sample-limiting flag name USED to be an unverified assumption
(`--max-samples`, a plausible lighteval convention that was never confirmed
against the docs). ROUND2_HANDOFF.md sec 5 lists it as "the first thing to
check if it errors" -- but Gate 2 has never actually been run, so a wrong
guess would have burned a GPU-window slot discovering a one-word problem.
It is now RESOLVED AT RUNTIME against `lighteval accelerate --help`
(see `detect_sample_limit_flag`), so the harness adapts instead of guessing.
"""

import glob
import json
import os
import re
import subprocess
import tempfile

# Tried in order against the installed lighteval's --help output. `--max-samples`
# stays first so behaviour is unchanged where that flag really exists.
SAMPLE_LIMIT_FLAG_CANDIDATES = (
    "--max-samples", "--max_samples", "--limit", "--num-samples",
)

_DETECTED_FLAG = None      # cache: resolved once per process
_DETECTION_FAILED = False


def detect_sample_limit_flag(force=False):
    """Return the flag this lighteval build uses to cap samples, or None.

    Returning None is meaningful, not an error: Phase A's smoke test simply
    runs unlimited, which is slower but still correct. Failing the whole gate
    over a flag name would be the wrong trade.
    """
    global _DETECTED_FLAG, _DETECTION_FAILED
    if _DETECTED_FLAG is not None and not force:
        return _DETECTED_FLAG
    if _DETECTION_FAILED and not force:
        return None
    try:
        proc = subprocess.run(["lighteval", "accelerate", "--help"],
                              capture_output=True, text=True, timeout=120)
        help_text = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[lighteval] could not run --help ({exc}); "
              f"proceeding without a sample limit")
        _DETECTION_FAILED = True
        return None
    for flag in SAMPLE_LIMIT_FLAG_CANDIDATES:
        # Word-boundary match so `--limit` does not match `--limit-tokens`.
        if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text):
            _DETECTED_FLAG = flag
            print(f"[lighteval] sample-limit flag detected: {flag}")
            return flag
    print(f"[lighteval] none of {SAMPLE_LIMIT_FLAG_CANDIDATES} found in "
          f"`lighteval accelerate --help`; smoke tests will run unlimited")
    _DETECTION_FAILED = True
    return None


def run_task(hf_repo, task_spec, max_samples=None, extra_args=None, tokenizer_path=None):
    """
    Runs a single lighteval task against a HF model repo.

    tokenizer_path: optional local directory of a patched tokenizer (see
    tokenizer_patch.py) to load INSTEAD of hf_repo's own tokenizer files,
    while model weights still load from hf_repo. Passed via lighteval's
    `tokenizer=` model_args key, which TransformersModelConfig resolves
    independently of `model_name=` (`tokenizer_name = self.config.tokenizer
    or self.config.model_name`).

    Returns (success: bool, results_dict_or_none, raw_stdout_stderr: str).
    """
    with tempfile.TemporaryDirectory(prefix="lighteval_") as output_dir:
        model_args = f"model_name={hf_repo}"
        if tokenizer_path:
            model_args += f",tokenizer={tokenizer_path}"
        cmd = [
            "lighteval", "accelerate",
            model_args,
            task_spec,
            "--output-dir", output_dir,
            # Every CANDIDATE_TASKS entry lives under lighteval's `multilingual`
            # task suite (community_arc_hin_mcf, community_boolq_hin, etc.),
            # which is NOT loaded by default -- `lighteval accelerate --help`
            # shows `--load-tasks-multilingual / --no-load-tasks-multilingual`
            # defaulting to off. Without this flag every one of these tasks
            # fails identically with "Cannot find task X in task list or in
            # custom task registry", which looks exactly like a wrong task-name
            # string (and WAS partly that too, see research.md Bug 15) but is
            # actually this: the whole suite is invisible until asked for.
            "--load-tasks-multilingual",
        ]
        if max_samples is not None:
            flag = detect_sample_limit_flag()
            if flag is not None:
                cmd += [flag, str(max_samples)]
            # else: run unlimited rather than pass an unrecognised argument,
            # which lighteval would reject outright.
        if extra_args:
            cmd += extra_args

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        combined_output = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            return False, None, combined_output

        result_files = glob.glob(os.path.join(output_dir, "results", "**", "results_*.json"), recursive=True)
        if not result_files:
            return False, None, combined_output + "\n[lighteval_runner] no results_*.json found in output dir"

        newest = max(result_files, key=os.path.getmtime)
        with open(newest) as f:
            data = json.load(f)
        return True, data, combined_output


def extract_task_metrics(results_json, task_spec):
    """Pulls the metrics dict for one task out of a lighteval results JSON
    (keyed by the exact 'task_name|n_shots' string, per lighteval's
    documented output format)."""
    return results_json.get("results", {}).get(task_spec)
