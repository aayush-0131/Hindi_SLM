#!/usr/bin/env bash
# Guarded one-step native SFT smoke for the capstone critical path.
# Non-destructive: writes a fresh timestamped result directory and refuses to proceed
# unless preflight passes. Does not touch Hidden-25 or publish anything.
set -euo pipefail

REPO="${REPO:-$HOME/Hindi_SLM}"
RUNTIME="${RUNTIME:-$HOME/hindi_slm_runtime}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/results/sft_smoke/$STAMP"
mkdir -p "$OUT"

cd "$REPO"

python tools/capstone_preflight.py \
  --runtime-root "$RUNTIME" \
  --json-out "$OUT/preflight.json"

python - <<'PY' > "$OUT/runtime.json"
import json, os, platform, subprocess, sys
mods={}
for name in ("torch","wandb"):
    try:
        m=__import__(name); mods[name]=getattr(m,"__version__","UNKNOWN")
    except Exception as e:
        mods[name]={"import_error":repr(e)}
try:
    import torch
    cuda={"available":torch.cuda.is_available(),"cuda":torch.version.cuda,
          "devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
except Exception as e:
    cuda={"error":repr(e)}
print(json.dumps({"python":sys.version,"platform":platform.platform(),"modules":mods,
 "cuda":cuda,"NANOCHAT_DTYPE":os.environ.get("NANOCHAT_DTYPE")},indent=2))
PY

git rev-parse HEAD > "$OUT/repo_commit.txt"
git status --porcelain > "$OUT/git_status.txt"

export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The custom history task/trainer are expected to be present from the canonical project state.
# Keep this invocation deliberately tiny: one optimization step, batch size 1, no W&B run.
# tee preserves stdout for train_log reconstruction even if the process OOMs.
set +e
python -m nanochat.scripts.chat_sft \
  --task=HistorySFT \
  --run=dummy \
  --num_iterations=1 \
  --device_batch_size=1 \
  --sequence_len=256 \
  2>&1 | tee "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

python - <<PY > "$OUT/result.json"
import json
print(json.dumps({"exit_code":$RC,"result_dir":"$OUT",
 "interpretation":"PASS if exit_code=0; CUDA OOM means use prepared memory-efficient fallback. Other errors are environment/integration blockers, not evidence that full SFT does not fit."},indent=2))
PY

sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null || true
cat "$OUT/result.json"
exit "$RC"
