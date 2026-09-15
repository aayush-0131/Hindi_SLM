#!/usr/bin/env python3
"""Non-destructive preflight for the Hindi SLM capstone.

Checks only local files/environment and prints one machine-readable report. It never
loads the 2.6 GB model, starts training, evaluates Hidden-25, publishes, or overwrites
artifacts. Run from the repository root (or pass --repo-root).
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

BASE_SHA = "1ad56ea60a5e28d6899146a56f63b1833837dbe6f6cef9128bb2bf4a51faa00d"
META_SHA = "8f2d7c586d0c9ddc30633dee4310c4759ca9068b5c256595f124f57a43563013"
TOK_SHA = "25be857870b2cebbc1dfc729738d6e03dd1399821cdeb1d1d85a86be6ab59823"
TOK_BYTES_SHA = "96e1e74bbbfae0c277a5b8644f64487325745ed2f26cf3a3a267271c98999234"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str):
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--runtime-root", default=os.path.expanduser("~/hindi_slm_runtime"))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    repo, runtime = Path(a.repo_root).resolve(), Path(a.runtime_root).resolve()

    checks = []
    def add(name, status, detail): checks.append({"check": name, "status": status, "detail": detail})

    # Cheap environment checks first: these explain failures before any GPU/model work.
    for mod in ("torch", "wandb"):
        spec = importlib.util.find_spec(mod)
        add(f"python_import:{mod}", "PASS" if spec else "BLOCKER", "available" if spec else "missing")

    # Canonical source/config presence.
    required_repo = [
        "CAPSTONE_STATE.md",
        "nanochat/pyproject.toml",
        "nanochat/uv.lock",
    ]
    for rel in required_repo:
        p = repo / rel
        add(f"repo_file:{rel}", "PASS" if p.is_file() else "BLOCKER", str(p))

    # The custom trainer/task may currently exist only on Lightning; report that explicitly.
    for rel in ("nanochat/scripts/chat_sft_history.py", "nanochat/tasks/history_hi_sft.py"):
        p = repo / rel
        add(f"custom_sft:{rel}", "PASS" if p.is_file() else "ATTENTION", str(p))

    # Critical rescued runtime artifacts. Hash only when present; this is read-only.
    expected = {
        "ckpt_stable_final/model_033750.pt": BASE_SHA,
        "ckpt_stable_final/meta_033750.json": META_SHA,
        "tokenizer/tokenizer.pkl": TOK_SHA,
        "tokenizer/token_bytes.pt": TOK_BYTES_SHA,
    }
    for rel, expected_sha in expected.items():
        p = runtime / rel
        if not p.is_file():
            add(f"runtime_artifact:{rel}", "BLOCKER", f"missing: {p}")
        else:
            actual = sha256(p)
            add(f"runtime_artifact:{rel}", "PASS" if actual == expected_sha else "BLOCKER",
                {"path": str(p), "sha256": actual, "expected": expected_sha})

    # SFT data: canonical state says 275 train / 36 validation; accept common filenames.
    data_dir = repo / "data/history_hi_sft_v1"
    train = next((p for p in (data_dir/"train.jsonl", data_dir/"train.csv") if p.is_file()), None)
    val = next((p for p in (data_dir/"val.jsonl", data_dir/"validation.jsonl", data_dir/"val.csv") if p.is_file()), None)
    add("sft_train_file", "PASS" if train else "BLOCKER", str(train) if train else str(data_dir))
    add("sft_val_file", "PASS" if val else "BLOCKER", str(val) if val else str(data_dir))

    head = git(repo, "rev-parse", "HEAD")
    dirty = git(repo, "status", "--porcelain")
    report = {
        "schema_version": "1.0",
        "purpose": "preflight only; no model load/training/hidden evaluation",
        "repo_root": str(repo), "runtime_root": str(runtime),
        "git_head": head, "git_dirty": bool(dirty) if dirty is not None else None,
        "checks": checks,
    }
    blockers = [x for x in checks if x["status"] == "BLOCKER"]
    report["summary"] = {"blockers": len(blockers), "attention": sum(x["status"] == "ATTENTION" for x in checks),
                         "ready_for_next_execution_stage": len(blockers) == 0}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if a.json_out:
        dest = Path(a.json_out)
        if dest.exists():
            raise SystemExit(f"Refusing to overwrite existing report: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
    return 0 if not blockers else 2

if __name__ == "__main__":
    raise SystemExit(main())
