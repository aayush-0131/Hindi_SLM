#!/usr/bin/env python3
"""Capture the exact capstone runtime without modifying the environment.

Writes a JSON manifest plus pip-freeze text. Refuses to overwrite by default.
Run this on the machine used for SFT/evaluation so submission evidence reflects
that runtime rather than the repository's declared dependency constraints.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone


def cmd(args):
    try:
        return subprocess.run(args, text=True, capture_output=True, check=False)
    except Exception as e:
        return type("R", (), {"returncode": -1, "stdout": "", "stderr": repr(e)})()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/runtime_capture")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze_path = out / "pip_freeze.txt"
    json_path = out / "runtime_env.json"
    for p in (freeze_path, json_path):
        if p.exists() and not a.force:
            raise SystemExit(f"Refusing to overwrite existing {p}; use a new --out-dir or --force intentionally")

    freeze = cmd([sys.executable, "-m", "pip", "freeze"])
    freeze_path.write_text(freeze.stdout, encoding="utf-8")
    git = cmd(["git", "rev-parse", "HEAD"])
    status = cmd(["git", "status", "--porcelain"])
    nvcc = cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])

    modules = {}
    for name in ["torch", "wandb", "numpy", "tiktoken", "rustbpe", "pyarrow", "psutil", "filelock"]:
        try:
            m = __import__(name)
            modules[name] = getattr(m, "__version__", "UNKNOWN")
        except Exception as e:
            modules[name] = f"IMPORT_ERROR: {type(e).__name__}: {e}"

    torch_info = {}
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as e:
        torch_info = {"error": repr(e)}

    data = {
        "schema_version": "1.0",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Hindi SLM capstone SFT/evaluation runtime evidence",
        "python": {"version": sys.version, "executable": sys.executable, "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "platform": platform.platform()},
        "git": {"commit": git.stdout.strip() if git.returncode == 0 else None, "dirty": bool(status.stdout.strip()), "status_porcelain": status.stdout.splitlines()},
        "nvidia_smi": {"returncode": nvcc.returncode, "gpus": nvcc.stdout.splitlines(), "stderr": nvcc.stderr.strip()},
        "modules": modules,
        "torch": torch_info,
        "pip_freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path), "returncode": freeze.returncode, "stderr": freeze.stderr.strip()},
        "environment_policy": "Secrets and arbitrary environment variables are intentionally not captured.",
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_env": str(json_path), "runtime_env_sha256": sha256(json_path), "pip_freeze": str(freeze_path), "pip_freeze_sha256": sha256(freeze_path)}, indent=2))

if __name__ == "__main__":
    main()
