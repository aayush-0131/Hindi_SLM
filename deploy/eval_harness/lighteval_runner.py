"""
Thin wrapper around lighteval's CLI (confirmed pattern from lighteval's own
docs: `lighteval accelerate "model_name=<repo>" "<task>|<n_shots>"`). Runs
one task per subprocess call rather than batching multiple tasks into one
call -- simpler to isolate failures per task, and sidesteps uncertainty
about the exact multi-task argument syntax.

UNVERIFIED: the sample-limiting flag is assumed to be `--max-samples`
(a common lighteval CLI convention) -- this was not directly confirmed
against current lighteval docs. If a real run errors on this flag with
"unrecognized argument", that's the first thing to check and fix.
"""

import glob
import json
import os
import subprocess
import tempfile


def run_task(hf_repo, task_spec, max_samples=None, extra_args=None):
    """
    Runs a single lighteval task against a HF model repo.
    Returns (success: bool, results_dict_or_none, raw_stdout_stderr: str).
    """
    with tempfile.TemporaryDirectory(prefix="lighteval_") as output_dir:
        cmd = [
            "lighteval", "accelerate",
            f"model_name={hf_repo}",
            task_spec,
            "--output-dir", output_dir,
        ]
        if max_samples is not None:
            cmd += ["--max-samples", str(max_samples)]
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
