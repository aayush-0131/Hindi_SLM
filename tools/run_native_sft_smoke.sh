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
import json, os, platform, sys
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

# Canonical capstone state specifies nanochat/scripts/chat_sft_history.py as the
# custom trainer wired directly to HistoryHiSFT(train/val). Refuse to fall back
# to the upstream SmolTalk/MMLU/GSM8K trainer: doing so would test the wrong task.
TRAINER="$REPO/nanochat/scripts/chat_sft_history.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "BLOCKER: canonical History SFT trainer missing: $TRAINER" | tee "$OUT/run.log"
  python - <<PY > "$OUT/result.json"
import json
print(json.dumps({"exit_code":4,"result_dir":"$OUT",
 "interpretation":"INTEGRATION_BLOCKER: chat_sft_history.py is missing. Do not infer native-SFT OOM/fit from this run."},indent=2))
PY
  sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null || true
  cat "$OUT/result.json"
  exit 4
fi

python -m py_compile "$TRAINER"

# Deliberately tiny: one optimization step, batch size 1, sequence length 256.
# The history-specific trainer owns the HistoryHiSFT wiring; no generic --task
# override is supplied here. tee preserves stdout even on OOM.
set +e
python "$TRAINER" \
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
 "trainer":"nanochat/scripts/chat_sft_history.py",
 "interpretation":"PASS if exit_code=0; CUDA OOM means use prepared memory-efficient fallback. Other errors are environment/integration blockers, not evidence that full SFT does not fit."},indent=2))
PY

sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null || true
cat "$OUT/result.json"
exit "$RC"
