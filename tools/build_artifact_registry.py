#!/usr/bin/env python3
"""Build a deterministic SHA-256 registry for capstone submission artifacts.

The registry hashes only files that actually exist. Missing required/optional paths
are recorded explicitly instead of guessed. The output is written atomically and
is never overwritten unless --force is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = [
    "CAPSTONE_STATE.md",
    "submission/manifest.yaml",
    "submission/train_log.schema.json",
    "submission/train_log.jsonl",
    "submission/eval_protocol.yaml",
    "submission/inference_config.yaml",
    "nanochat/tasks/history_hi_sft.py",
    "eval/safety_hi_v1.jsonl",
    "eval/safety_hi_v1_rubric.md",
]
OPTIONAL = [
    "nanochat/scripts/chat_sft_history.py",
    "data/history_hi_sft_v1/train.jsonl",
    "data/history_hi_sft_v1/val.jsonl",
    "results/runtime_capture_pre_sft/runtime.json",
    "results/runtime_capture_pre_sft/pip_freeze.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def entry(repo: Path, rel: str, required: bool) -> dict:
    p = repo / rel
    out = {"path": rel, "required": required, "present": p.is_file()}
    if p.is_file():
        out.update({"bytes": p.stat().st_size, "sha256": sha256(p)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--output", default="submission/artifact_registry.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    out_path = repo / args.output
    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out_path}; pass --force after reviewing the existing registry.")

    files = [entry(repo, p, True) for p in REQUIRED]
    files += [entry(repo, p, False) for p in OPTIONAL]
    missing_required = [x["path"] for x in files if x["required"] and not x["present"]]

    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "source": {
            "git_commit": git(repo, "rev-parse", "HEAD"),
            "git_branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(git(repo, "status", "--porcelain")),
        },
        "files": files,
        "validation": {
            "required_count": len(REQUIRED),
            "required_present": len(REQUIRED) - len(missing_required),
            "missing_required": missing_required,
            "status": "PASS" if not missing_required else "INCOMPLETE",
        },
        "notes": [
            "This registry records byte-level integrity, not semantic correctness.",
            "Runtime-only/final-checkpoint artifacts must be added or captured after the final run; absent files are never assigned synthetic hashes.",
            "Regenerate only after reviewing changes; --force is intentionally explicit.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out_path.name + ".", dir=out_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print(json.dumps(payload["validation"], ensure_ascii=False))
    return 0 if not missing_required else 2


if __name__ == "__main__":
    raise SystemExit(main())
