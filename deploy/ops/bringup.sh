#!/usr/bin/env bash
# Rebuild the Round 2 working environment on the GPU host, from an uploaded
# backup/ directory. Run this ON THE SERVER, in an interactive shell.
#
#   bash deploy/ops/bringup.sh          # build (idempotent, safe to re-run)
#   bash deploy/ops/bringup.sh --check  # verify only, change nothing
#
# WHY THIS EXISTS
# ---------------
# /workspace is purged between rounds, and ROUND2_HANDOFF.md sec 8 Phase B says
# only "re-upload backup/, recreate the directory layout and symlinks". That
# understates the job: hindi_env.sh sets PYTHONPATH and its venv to
# $TEAM/nanochat, which is INSIDE the purged directory. So the nanochat clone,
# its uv-managed venv, and the uv-installed Python 3.10 that CPATH points at
# are all gone too, and every one of them is load-bearing for training.
#
# Doing this by hand under a ticking 20-hour window is how Round 1 produced two
# undetected stale uploads and a header-only CSV written against a
# non-existent path. Hence a script.
#
# HARD-WON FACTS THIS ENCODES (ROUND2_HANDOFF.md sec 2 and sec 7)
# ---------------------------------------------------------------
#  - SSH login is team02; the interactive shell identity is usr1-iairo with a
#    DIFFERENT $HOME. Never assume ~ resolves where you expect -- this script
#    prints both and uses explicit paths.
#  - Uploads land in /workspace (the team02 login's home). The working layout
#    lives outside it. Same filesystem, so artifacts are HARD-LINKED rather
#    than copied -- instant, and no second copy of 21 GB.
#  - python3.10-dev is missing on the host, so Triton cannot compile and
#    torch.compile dies with an opaque InductorError. The fix is a uv-installed
#    Python 3.10 plus CPATH. This is the single most important line here.
#  - nanochat resolves BOTH the tokenizer and the corpus from
#    NANOCHAT_BASE_DIR alone, via the exact directory names tokenizer/ and
#    base_data_climbmix/. Symlinks are fine.
#  - No sudo. No pip in the nanochat venv -- use `uv pip install`.

set -uo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

ROOT="/data1/users/usr1-iairo"
TEAM="$ROOT/pramana-bootcamp/nosudo/workspaces/team02"   # == /workspace
BACKUP="$TEAM/backup"
NANOCHAT="$TEAM/nanochat"
TOKDIR="$ROOT/hindi_lm_tokenizer_gate/tokenizer"
CORPUS="$ROOT/hindi_lm_data_pipeline/data_pipeline_artifacts/final/stable"
TRAIN="$ROOT/hindi_lm_train"
CKPT="$TRAIN/base_checkpoints/stable"
ENVSH="$ROOT/hindi_env.sh"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
fail=0
ok()   { echo "  ${GRN}OK${RST}   $*"; }
warn() { echo "  ${YEL}WARN${RST} $*"; }
bad()  { echo "  ${RED}FAIL${RST} $*"; fail=$((fail+1)); }
hdr()  { echo; echo "=== $* ==="; }

hdr "identity and paths (never assume ~)"
echo "  whoami : $(whoami)"
echo "  \$HOME  : ${HOME:-unset}"
echo "  pwd    : $(pwd)"
echo "  ROOT   : $ROOT"
echo "  TEAM   : $TEAM  (this is /workspace; purged between rounds)"
[[ -d "$ROOT" ]] || { echo "${RED}ROOT does not exist -- is this the right host?${RST}"; exit 1; }
[[ -d "$TEAM" ]] || warn "TEAM missing; uploads may not have landed yet"

hdr "uploaded backup"
for d in tokenizer ckpt corpus logs; do
  if [[ -d "$BACKUP/$d" ]]; then
    n=$(find "$BACKUP/$d" -type f | wc -l | tr -d ' ')
    sz=$(du -sh "$BACKUP/$d" 2>/dev/null | cut -f1)
    ok "backup/$d present ($n files, $sz)"
  else
    # Only a WARNing: an absent staging directory is fine when the artifact is
    # already at its destination. On 2026-09-06 the whole 42 GB pipeline tree
    # and 331 GB of checkpoints survived the purge, so nothing needed staging
    # -- and treating that as FAIL made a clean bring-up look broken. The
    # authoritative checks are in VERIFY below, against the real paths.
    warn "backup/$d not staged -- fine IF the artifact is already in place; VERIFY below decides"
  fi
done

if [[ $CHECK_ONLY -eq 0 ]]; then
  hdr "directory layout"
  mkdir -p "$TOKDIR" "$CORPUS" "$CKPT" || bad "mkdir failed"
  ok "created $TOKDIR"
  ok "created $CORPUS"
  ok "created $CKPT"

  # Hard-link instead of copy: same filesystem, instant, no extra disk for
  # 21 GB. `ln -f` makes re-running idempotent.
  hdr "linking artifacts into the working layout"
  link_dir() {  # <src> <dst> <label>
    local src="$1" dst="$2" label="$3"
    if [[ ! -d "$src" ]]; then warn "$label: source $src missing, skipped"; return; fi
    local n=0
    shopt -s nullglob
    for f in "$src"/*; do
      [[ -f "$f" ]] || continue
      ln -f "$f" "$dst/$(basename "$f")" 2>/dev/null \
        || cp -f "$f" "$dst/$(basename "$f")" \
        || { bad "$label: could not place $(basename "$f")"; continue; }
      n=$((n+1))
    done
    shopt -u nullglob
    ok "$label: $n file(s) -> $dst"
  }
  link_dir "$BACKUP/tokenizer" "$TOKDIR"  "tokenizer"
  link_dir "$BACKUP/corpus"    "$CORPUS"  "corpus"
  link_dir "$BACKUP/ckpt"      "$CKPT"    "checkpoint"

  hdr "nanochat (gone with the purge -- rebuilding)"
  if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "$ROOT/.local/bin/uv" ]]; then
      export PATH="$ROOT/.local/bin:$PATH"
      ok "uv found at $ROOT/.local/bin/uv"
    else
      warn "uv not found -- installing (no sudo needed)"
      curl -LsSf https://astral.sh/uv/install.sh | sh || bad "uv install failed"
      export PATH="$ROOT/.local/bin:$PATH"
    fi
  else
    ok "uv on PATH: $(command -v uv)"
  fi

  if [[ -d "$NANOCHAT/.git" ]]; then
    ok "nanochat already cloned at $NANOCHAT"
  else
    mkdir -p "$(dirname "$NANOCHAT")"
    git clone https://github.com/karpathy/nanochat "$NANOCHAT" \
      || bad "nanochat clone failed (network? check with: curl -sI https://github.com)"
  fi

  # Python 3.10 headers: the whole reason CPATH exists. Missing headers ->
  # Triton cannot build cuda_utils.c -> torch.compile dies with an opaque
  # InductorError and the real gcc error scrolled off screen.
  hdr "python 3.10 + CPATH (mandatory for torch.compile)"
  uv python install 3.10 >/dev/null 2>&1 \
    && ok "uv python 3.10 installed" \
    || warn "uv python install 3.10 returned non-zero (may already exist)"
  PYINC=$(find "$ROOT/.local/share/uv/python" -maxdepth 3 -type d \
            -name "python3.10" -path "*/include/*" 2>/dev/null | head -1)
  if [[ -n "$PYINC" && -f "$PYINC/Python.h" ]]; then
    ok "Python.h found: $PYINC/Python.h"
  else
    bad "Python.h NOT found under $ROOT/.local/share/uv/python -- torch.compile WILL fail"
    PYINC="${PYINC:-/MISSING-SET-THIS-BY-HAND}"
  fi

  if [[ -d "$NANOCHAT" ]]; then
    hdr "nanochat venv (uv-managed; there is no pip in it)"
    # Do NOT recreate a venv that already imports nanochat. `uv venv` on an
    # existing directory replaces it, and the Round 1 clone can survive the
    # purge with a working install -- recreating it would throw away a large
    # torch download for nothing.
    if [[ -x "$NANOCHAT/.venv/bin/python3" ]] \
       && "$NANOCHAT/.venv/bin/python3" -c "import nanochat, datasets" 2>/dev/null; then
      ok "existing venv already imports nanochat + datasets -- left alone"
    else
      # Output is NOT suppressed. This is the step most likely to fail (torch
      # is a large download) and hiding its error is how a failure here first
      # surfaced three commands later as ModuleNotFoundError.
      #
      # DEPENDENCIES ARE INSTALLED FROM pyproject.toml DIRECTLY, not as a
      # side effect of `uv pip install -e .`. On a dirty clone the editable
      # build FAILS (Round 1 leaves runs/ and dev/ at the repo root and
      # setuptools refuses flat-layout discovery with multiple top-level
      # packages) -- and when it fails, NONE of the declared dependencies get
      # installed either. That produced three separate ModuleNotFoundErrors in
      # a row (datasets, rustbpe, wandb), each found only by running the next
      # command. Reading the dependency array out of pyproject.toml cannot
      # miss one. torch is excluded so its working CUDA build is preserved.
      ( cd "$NANOCHAT" \
        && { [[ -x .venv/bin/python3 ]] || uv venv --python 3.10; } \
        && sed -n '/^dependencies = \[/,/^]/p' pyproject.toml \
             | grep -oE '"[^"]+"' | tr -d '"' \
             | grep -viE '^torch([=<>!~]|$)' > /tmp/hindi_nc_deps.txt \
        && echo "  installing $(wc -l < /tmp/hindi_nc_deps.txt) declared deps (torch excluded)" \
        && uv pip install -r /tmp/hindi_nc_deps.txt ) \
        && ok "nanochat dependencies installed from pyproject.toml" \
        || warn "dependency install had a problem -- see error above. PYTHONPATH is how nanochat is imported (see hindi_env.sh), so the package need not be pip-installed. It fails on a dirty clone because Round 1 leaves runs/ and dev/ at the repo root and setuptools refuses flat-layout auto-discovery with multiple top-level packages. Verify with: python3 -c 'import nanochat; print(nanochat.__file__)' from OUTSIDE the repo root."

      # The data pipeline needs datasets/pyarrow/datasketch/transformers,
      # which are NOT nanochat dependencies -- so `uv pip install -e .` alone
      # leaves run_pipeline.py and run_decay_corpus.py unable to import.
      if [[ -f "$TEAM/deploy/requirements.txt" ]]; then
        # torch is EXCLUDED deliberately. requirements.txt lists it unpinned,
        # and nanochat pins its own version -- installing an unpinned torch on
        # top could replace a working training install to satisfy a data
        # pipeline dependency. nanochat's torch is the one that matters.
        grep -viE '^[[:space:]]*torch([=<>!~[:space:]]|$)' \
          "$TEAM/deploy/requirements.txt" > /tmp/hindi_reqs_notorch.txt
        ( cd "$NANOCHAT" && uv pip install -r /tmp/hindi_reqs_notorch.txt ) \
          && ok "pipeline requirements installed (torch left to nanochat's pin)" \
          || bad "pipeline requirements FAILED (error above)"
        "$NANOCHAT/.venv/bin/python3" -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())" \
          || bad "torch broken after requirements install -- reinstall nanochat: cd $NANOCHAT && uv pip install -e ."
      else
        warn "deploy/requirements.txt not found -- upload deploy/ first"
      fi
    fi
    # rustbpe is a PUBLISHED PyPI package (pyproject declares
    # "rustbpe>=0.1.0"); there is no rustbpe/ subdirectory to build and no Rust
    # toolchain on this host (no cargo/rustc). An earlier version of this
    # script tried `maturin develop` and dismissed a failure as "only needed to
    # TRAIN a tokenizer" -- both wrong. nanochat/tokenizer.py does
    # `import rustbpe` at MODULE level, so base_train cannot even import
    # without it. It is required for training.
    if ! "$NANOCHAT/.venv/bin/python3" -c "import rustbpe" 2>/dev/null; then
      ( cd "$NANOCHAT" && uv pip install rustbpe ) \
        && ok "rustbpe installed from PyPI" \
        || bad "rustbpe install FAILED -- base_train cannot import the tokenizer without it"
    else
      ok "rustbpe importable"
    fi
    # Flash Attention 3 arrives via the `kernels` package (prebuilt from the
    # Hub), NOT a local compile, and nanochat wraps it as
    # nanochat.flash_attention -- so checking for a top-level `flash_attn`
    # module gives a false negative. MFU depends on this: the SDPA fallback is
    # far slower, so Gate 3 is what actually confirms it.
    "$NANOCHAT/.venv/bin/python3" -c "
from nanochat.flash_attention import flash_attn
print('  flash_attn wrapper resolved:', hasattr(flash_attn,'flash_attn_func'))" \
      2>/dev/null || warn "nanochat.flash_attention did not resolve -- expect SDPA fallback and much lower MFU; confirm with run_gate3.py before the long run"
  fi

  hdr "hindi_env.sh"
  # Two corrections versus the Round 1 file:
  #  1. NANOCHAT_BASE_DIR defaulted to hindi_lm_gate3 -- the THROWAWAY Gate 3
  #     base dir. Sourcing it and launching training without an override would
  #     train against the wrong tokenizer/corpus wiring. Now defaults to the
  #     real training base dir.
  #  2. CPATH is resolved at build time here rather than hardcoded to one
  #     cpython patch version, which changes whenever uv updates.
  cat > "$ENVSH" <<EOF
# source this before any training command.  Regenerated by ops/bringup.sh
export TEAM=$TEAM
source \$TEAM/nanochat/.venv/bin/activate

# MANDATORY. The host has no python3.10-dev, so Triton cannot compile
# cuda_utils.c and torch.compile dies with an opaque InductorError. gcc honours
# CPATH as an extra include path even though Triton hardcodes
# -I/usr/include/python3.10.  See ROUND2_HANDOFF.md sec 7.1.
export CPATH=$PYINC

export CUDA_VISIBLE_DEVICES=0          # only ONE GPU may be used
# deploy/ is on the path too, so `python3 deploy/training/preflight.py`
# can do `from training import config` without a manual PYTHONPATH prefix.
export PYTHONPATH=\$TEAM/nanochat:\$TEAM/deploy

# Round 1's version defaulted this to hindi_lm_gate3 (the throwaway Gate 3 dir).
# It now points at the real training base dir; override per-command if needed.
export NANOCHAT_BASE_DIR=\${NANOCHAT_BASE_DIR:-$TRAIN}

echo "env ready | venv=\$(which python3) | base_dir=\$NANOCHAT_BASE_DIR | gpu=\$CUDA_VISIBLE_DEVICES | cpath=\$CPATH"
EOF
  ok "wrote $ENVSH"

  hdr "symlinks nanochat resolves from NANOCHAT_BASE_DIR"
  ln -sfn "$TOKDIR" "$TRAIN/tokenizer"           && ok "$TRAIN/tokenizer -> $TOKDIR"
  ln -sfn "$CORPUS" "$TRAIN/base_data_climbmix"  && ok "$TRAIN/base_data_climbmix -> $CORPUS"
fi

# ---------------------------------------------------------------- verification
hdr "VERIFY"

if [[ -f "$TOKDIR/tokenizer.pkl" && -f "$TOKDIR/token_bytes.pt" ]]; then
  ok "tokenizer complete (tokenizer.pkl + token_bytes.pt)"
else
  # RustBPETokenizer.save() writes ONLY tokenizer.pkl. token_bytes.pt comes
  # from separate code in scripts/tok_train.py, which Gate 1 bypassed -- that
  # was Bug 11, and base_train hard-asserts the file.
  bad "tokenizer INCOMPLETE -- base_train will refuse to start. Regenerate with tokenizer_gate/make_token_bytes.py"
fi

nshard=$(ls -1 "$CORPUS"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [[ "$nshard" -gt 0 ]]; then
  ok "corpus: $nshard parquet shards"
  last=$(ls -1 "$CORPUS"/*.parquet | sort | tail -1 | xargs basename)
  # nanochat takes the LAST sorted shard as its val split. If that is not the
  # pinned zz_val_ shard, validation silently becomes some arbitrary source.
  if [[ "$last" == zz_val_* ]]; then
    ok "last sorted shard is $last (the pinned val split)"
  else
    bad "last sorted shard is $last, NOT a zz_val_* shard -- nanochat would use it as validation"
  fi
  # Source tag is the LAST underscore-separated field only in the interleaved
  # scheme (mix_00042_sgp.parquet). Round 1's names (fw2_00042.parquet) put the
  # INDEX there, so the old one-liner extracted "00042", found every value
  # unique, and reported a perfect run of 1 on a corpus with a run of 100.
  # Now: detect the scheme first, and say plainly when it is not interleaved.
  if ls -1 "$CORPUS"/mix_*.parquet >/dev/null 2>&1; then
    runs=$(ls -1 "$CORPUS"/mix_*.parquet | sort | sed 's/.*_//; s/\.parquet//' \
           | uniq -c | awk '{print $1}' | sort -rn | head -1)
    ok "interleaved; longest same-source run: ${runs:-?} shards (Round 1 was 100)"
  else
    bad "corpus is NOT interleaved (no mix_*.parquet). Round 1's source ordering cost a +42% train-loss spike once per epoch. Run: python3 deploy/run_interleave.py --corpus-dir $CORPUS"
  fi
else
  bad "corpus EMPTY at $CORPUS"
fi

if compgen -G "$CKPT/model_*.pt" >/dev/null; then
  step=$(ls -1 "$CKPT"/model_*.pt | sort | tail -1 | sed 's/.*model_0*//; s/\.pt//')
  ok "checkpoint present, highest step: $step"
  # All three files matter. A missing optim_ shard restores weights but
  # restarts Muon COLD, which looks like it worked and silently discards
  # optimisation progress.
  for f in "model_0${step}.pt" "optim_0${step}_rank0.pt" "meta_0${step}.json"; do
    if [[ -f "$CKPT/$f" ]]; then ok "  $f"
    else bad "  $f MISSING -- resume would restart the optimizer cold"; fi
  done
else
  bad "no checkpoint in $CKPT"
fi

[[ -f "$ENVSH" ]] && ok "hindi_env.sh present" || bad "hindi_env.sh missing"
if [[ -f "$ENVSH" ]]; then
  cp_line=$(grep -m1 '^export CPATH=' "$ENVSH" | cut -d= -f2-)
  if [[ -f "$cp_line/Python.h" ]]; then ok "CPATH resolves to real headers: $cp_line"
  else bad "CPATH points at $cp_line which has no Python.h -- torch.compile will fail"; fi
fi

for l in tokenizer base_data_climbmix; do
  if [[ -L "$TRAIN/$l" && -e "$TRAIN/$l" ]]; then ok "symlink $l resolves"
  else bad "symlink $TRAIN/$l broken or missing"; fi
done

hdr "RESULT"
if [[ $fail -eq 0 ]]; then
  echo "  ${GRN}bring-up clean.${RST}"
  echo
  echo "  Next, in order:"
  echo "    source $ENVSH"
  echo "    python3 -u run_resume_test.py     # Req 7.7 -- NEVER been executed"
  echo "    # then finish stable to 24,400 with:"
  echo "    #   --resume-from-step=21200 --num-iterations=30500 --warmdown-ratio=0.20"
  echo "    # num-iterations and warmdown-ratio MUST NOT CHANGE -- the LR"
  echo "    # schedule is a pure function of them and is unfixable after the fact."
  exit 0
else
  echo "  ${RED}$fail check(s) failed -- fix before spending GPU time.${RST}"
  exit 1
fi
