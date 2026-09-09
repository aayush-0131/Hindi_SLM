"""Resumable, md5-verified download of the final artifacts from the GPU host.

WHY THIS EXISTS
---------------
`ops/transfer.py` solves the upload direction. Download is now the dangerous
one, and it is the direction that already cost this project a checkpoint:
ROUND2_HANDOFF.md sec 6.1 records `optim_022000_rank0.pt` never arriving because
the link degraded from 1.7 MB/s to ~22 KB/s at window close and `scp` has no
resume. `/workspace` is purged after the window, so anything not on the laptop
by then is gone permanently.

So this script never restarts a file from zero. It asks the remote for the size
and md5, compares against whatever is already on disk, and appends only the
missing tail bytes with `tail -c +N`. Re-running after a drop resumes at the
byte it stopped on; a file that already matches is skipped for free.

Artifacts live OUTSIDE `/workspace` (sec 2) and are therefore not scp-reachable,
so they have to be hard-linked into `backup/` first -- same filesystem, instant,
no disk cost. `--stage-only` does just that.

The identity split matters here (sec 2): the ssh *login* is `team02` but the
interactive shell is `usr1-iairo`, and `/data1/users/usr1-iairo` is `drwx------`.
A presence check that runs as the wrong identity reports "does not exist" for
files that are present -- which on 2026-09-06 nearly triggered a pointless
21 GB re-upload. `--probe` reports what is actually readable, and no check here
suppresses stderr.

USAGE
-----
    # what is on the server, what identity are we, what is readable
    python3 ops/download.py --probe

    # hard-link the newest checkpoint trio into backup/, then stop
    python3 ops/download.py --stage-only

    # download everything staged (resumable -- just re-run if it drops)
    python3 ops/download.py

    # only the checkpoint, and pick the step explicitly
    python3 ops/download.py --stage ckpt --step 33750
"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time

DEFAULT_HOST = "team02@global.prd.ga.run.brev.nvidia.com"
DEFAULT_PORT = "21510"
DEFAULT_LOCAL_ROOT = os.path.expanduser("~/hindi_lm_backup")

# Everything below is relative to the remote login cwd (== /workspace).
REMOTE_BACKUP = "backup"
# Artifact roots, which live outside /workspace and must be hard-linked in.
ARTIFACT_ROOT = "/data1/users/usr1-iairo"
CKPT_DIRS = {
    "stable": f"{ARTIFACT_ROOT}/hindi_lm_train/base_checkpoints/stable",
    "decay_a": f"{ARTIFACT_ROOT}/hindi_lm_train/base_checkpoints/decay_a",
    "decay_b": f"{ARTIFACT_ROOT}/hindi_lm_train/base_checkpoints/decay_b",
}
LOG_FILES = [
    f"{ARTIFACT_ROOT}/stable_train.log",
    f"{ARTIFACT_ROOT}/hindi_env.sh",
]

# The artifact root cannot be listed, so a log can only be found by guessing its
# exact name. Round 2 may not have reused Round 1's filename, and if training ran
# in tmux with no redirect there may be no file at all -- in which case the val
# series has to come from the checkpoint metadata instead (--val-series).
LOG_CANDIDATES = [
    f"{ARTIFACT_ROOT}/stable_train.log",
    f"{ARTIFACT_ROOT}/stable_train_r2.log",
    f"{ARTIFACT_ROOT}/stable_train_round2.log",
    f"{ARTIFACT_ROOT}/round2_train.log",
    f"{ARTIFACT_ROOT}/train.log",
    f"{ARTIFACT_ROOT}/train_r2.log",
    f"{ARTIFACT_ROOT}/hindi_lm_train.log",
    f"{ARTIFACT_ROOT}/nohup.out",
]

STAGE_NAMES = ["ckpt", "logs"]


def md5_local(path, blocksize=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(blocksize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human(n):
    """Byte counts, readable at every scale.

    Reporting `meta_*.json` and `hindi_env.sh` as "0.0 MB" made a present file
    indistinguishable from an empty one, which is exactly the ambiguity this
    script exists to remove.
    """
    if n is None:
        return "?"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} KB"
    return f"{n} B"


# From ROUND2_HANDOFF.md sec 6: every checkpoint of this architecture is exactly
# these sizes. A file that is present but a different size is still being
# written, or is truncated -- both look plausible and fail confusingly at load.
EXPECTED_SIZES = {
    "model": 2_661_366_058,
    "optim": 3_795_102_997,
}


class Remote:
    """ssh wrapper sharing ONE multiplexed connection.

    ControlPath is deliberately the same socket `ops/transfer.py` uses, so an
    already-open master is reused and the whole run costs zero extra password
    prompts. ControlPersist keeps it alive across separate invocations, which is
    what makes "just re-run it" cheap after a drop.
    """

    def __init__(self, host, port, control_path, dry_run=False):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self._base = [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path}",
            "-o", "ControlPersist=8h",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
        ]

    def ssh(self, command, check=True):
        argv = ["ssh"] + self._base + ["-p", self.port, self.host, command]
        if self.dry_run:
            print(f"    [dry-run] ssh: {command}")
            return ""
        proc = subprocess.run(argv, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"ssh failed ({proc.returncode}): {command}\n"
                f"{(proc.stderr or '').strip()}")
        # stderr is intentionally surfaced, never swallowed: "permission denied"
        # and "does not exist" must stay distinguishable (sec 2).
        if not check and proc.returncode != 0 and proc.stderr:
            print(f"    remote stderr: {proc.stderr.strip()}")
        return proc.stdout or ""

    def open_session(self):
        """Force the master up, so any password prompt happens here rather than
        mid-transfer. If the socket is already live this is free."""
        if self.dry_run:
            print("  [dry-run] would open multiplexed ssh session")
            return
        argv = ["ssh"] + self._base + ["-p", self.port, self.host,
                                       "echo connected; pwd; whoami"]
        proc = subprocess.run(argv, text=True, capture_output=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"cannot reach {self.host}:{self.port}\n"
                f"{(proc.stderr or '').strip()}\n\n"
                f"If this needs a password, open the shared session once from a "
                f"terminal that can prompt:\n"
                f"  ssh -o ControlMaster=auto "
                f"-o ControlPath=~/.ssh/'cm-hindi-%r@%h:%p' "
                f"-o ControlPersist=8h -p {self.port} {self.host} "
                f"'echo connected'\n"
                f"then re-run this script.")
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
        print(f"  connected. remote cwd: {lines[1] if len(lines) > 1 else '?'}"
              f"  identity: {lines[2] if len(lines) > 2 else '?'}")

    def stat_remote(self, path):
        """(size, md5) for one remote file, or (None, None) if unreadable.

        Size comes first and separately: it is cheap, whereas md5 on a 3.8 GB
        file costs real seconds, and resume only needs the size until the final
        verification.
        """
        out = self.ssh(f"stat -c %s {shlex.quote(path)}", check=False).strip()
        if not out.isdigit():
            return None, None
        return int(out), None

    def stat_full(self, path):
        """mode/owner/size/mtime for one path, via a single stat.

        `stat` only needs execute permission on the parent directories, so this
        still works on files that cannot be opened -- which is exactly the state
        the artifact tree is in. Owner and mode are what distinguish "not ours"
        from "revoked".
        """
        out = self.ssh(f"stat -c '%A|%U:%G|%s|%y' {shlex.quote(path)}",
                       check=False).strip()
        return out or None

    def md5_remote(self, path):
        out = self.ssh(f"md5sum {shlex.quote(path)}", check=False).strip()
        return out.split()[0] if out else None

    def fetch_tail(self, path, local_path, offset, total):
        """Append remote bytes from `offset` onward to `local_path`.

        This is the whole point of the script. `tail -c +N` is 1-indexed, so the
        byte at offset N is `+{N+1}`. Output is streamed straight into the local
        file in append mode, so an interrupted run leaves a shorter-but-valid
        prefix that the next run continues from -- unlike scp, which leaves a
        truncated file that looks plausible and fails much later.
        """
        argv = ["ssh"] + self._base + [
            "-p", self.port, self.host,
            f"tail -c +{offset + 1} {shlex.quote(path)}"]
        started = time.time()
        with open(local_path, "ab") as out:
            proc = subprocess.run(argv, stdout=out, stderr=subprocess.PIPE)
        elapsed = max(time.time() - started, 1e-9)
        got = os.path.getsize(local_path)
        moved = got - offset
        rate = moved / elapsed / 1e6
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode(errors="replace").strip()
            return False, f"transfer failed after {human(moved)}: {err}"
        if got != total:
            return False, (f"short read: {got} of {total} bytes "
                           f"({human(moved)} at {rate:.2f} MB/s) -- re-run to "
                           f"resume from here")
        return True, f"{human(moved)} at {rate:.2f} MB/s"


def newest_step(remote, ckpt_dir):
    """Highest step number with a complete model/optim/meta trio present.

    Reads the directory rather than trusting a step passed in by hand: the run
    may have saved another checkpoint since anyone last looked.
    """
    out = remote.ssh(
        f"ls {shlex.quote(ckpt_dir)}", check=False)
    steps = {}
    for name in out.split():
        for prefix, key in (("model_", "model"), ("optim_", "optim"),
                            ("meta_", "meta")):
            if name.startswith(prefix):
                digits = name[len(prefix):len(prefix) + 6]
                if digits.isdigit():
                    steps.setdefault(int(digits), set()).add(key)
    if not steps:
        return None, {}
    complete = [s for s, parts in steps.items()
                if {"model", "optim", "meta"} <= parts]
    return (max(complete) if complete else max(steps)), steps


def stage_checkpoint(remote, ckpt_dir, step, tag):
    """Hard-link one checkpoint trio into backup/ so scp can reach it.

    Hard links, not copies: same filesystem, instant, and 6.5 GB of checkpoint
    costs no extra disk. The `ln` runs as a single remote command per file with
    NO glob -- sec 6.1 records that `ssh "... *_NNNNNN* ..."` fails with a
    misleading "No such file or directory" because the glob does not expand
    through that quoting.
    """
    dest = f"{REMOTE_BACKUP}/ckpt_{step:06d}" if tag == "stable" else \
           f"{REMOTE_BACKUP}/ckpt_{tag}_{step:06d}"
    remote.ssh(f"mkdir -p {shlex.quote(dest)}")
    names = [f"model_{step:06d}.pt", f"optim_{step:06d}_rank0.pt",
             f"meta_{step:06d}.json"]
    linked = []
    for name in names:
        src = f"{ckpt_dir}/{name}"
        out = remote.ssh(
            f"ln -f {shlex.quote(src)} {shlex.quote(dest)}/{shlex.quote(name)} "
            f"&& stat -c %s {shlex.quote(dest)}/{shlex.quote(name)}",
            check=False).strip()
        if out.isdigit():
            size = int(out)
            linked.append((name, size))
            kind = name.split("_")[0]
            note = ""
            if kind in EXPECTED_SIZES and size != EXPECTED_SIZES[kind]:
                note = (f"  <-- WARNING: expected {EXPECTED_SIZES[kind]:,} B, "
                        f"got {size:,} B (still being written, or truncated)")
            print(f"    linked {name} ({human(size)}){note}")
        elif not remote.dry_run:
            print(f"    COULD NOT LINK {name} -- see stderr above",
                  file=sys.stderr)
    return dest, linked


def stage_logs(remote):
    """Hard-link the loose logs into backup/logs.

    `stable_train.log` is the only full record of the run; only 2 KB of it made
    it to the laptop in Round 1 (sec 6.1). `hindi_env.sh` carries the
    load-bearing CPATH line (sec 7.1).
    """
    dest = f"{REMOTE_BACKUP}/logs"
    remote.ssh(f"mkdir -p {shlex.quote(dest)}")
    linked = []
    for src in LOG_FILES:
        name = os.path.basename(src)
        out = remote.ssh(
            f"ln -f {shlex.quote(src)} {shlex.quote(dest)}/{shlex.quote(name)} "
            f"&& stat -c %s {shlex.quote(dest)}/{shlex.quote(name)}",
            check=False).strip()
        if out.isdigit():
            linked.append((name, int(out)))
            print(f"    linked {name} ({human(int(out))})")
    return dest, linked


def download_file(remote, remote_path, local_path, verify=True):
    """Download one file with resume and md5 verification.

    Priority is correctness over speed: a file is only reported ok once its md5
    matches the remote's. An md5 mismatch on a fully-sized file means the local
    copy is corrupt rather than short, so it is removed -- leaving it would make
    the next run skip it as "already complete".
    """
    size, _ = remote.stat_remote(remote_path)
    if size is None:
        return False, "not readable on remote"
    have = os.path.getsize(local_path) if os.path.exists(local_path) else 0

    if have == size:
        if not verify:
            return True, "already complete (size match, md5 not checked)"
        if md5_local(local_path) == remote.md5_remote(remote_path):
            return True, "already verified"
        os.remove(local_path)
        have = 0
        print("    local copy was corrupt, restarting from zero")
    elif have > size:
        os.remove(local_path)
        have = 0
        print("    local copy larger than remote, restarting from zero")

    if have:
        print(f"    resuming at {human(have)} of {human(size)} "
              f"({100 * have / size:.1f}% present)")
    ok, why = remote.fetch_tail(remote_path, local_path, have, size)
    if not ok:
        return False, why
    if verify:
        if md5_local(local_path) != remote.md5_remote(remote_path):
            return False, "md5 mismatch after transfer -- re-run to retry"
    return True, why + (", md5 verified" if verify else "")


def harvest_val_series(remote, ckpt_dir, from_step, to_step, every, out_csv):
    """Rebuild the validation curve from checkpoint metadata.

    Preferred source is the training log, but the log may not exist: if training
    ran in tmux without a redirect, the only record is scrollback. Every
    `meta_NNNNNN.json` carries `val_bpb` and `loop_state`, so the curve can be
    reconstructed from the checkpoints themselves at `save_every` granularity.

    Steps are enumerated arithmetically and fetched BY EXACT NAME in one round
    trip. No glob: globbing needs read permission on the directory, which is the
    permission we do not have -- the same trap that made `ls .../stable` fail
    while `stat` on a known filename succeeded.

    Note the value is the last eval performed at save time, not an eval at that
    exact step (eval_every=500 vs save_every=400), so consecutive rows repeat
    where no eval fell between two saves. That is a sampling artifact of the
    metadata, not a plateau in training.
    """
    steps = list(range(from_step, to_step + 1, every))
    if to_step not in steps:
        steps.append(to_step)
    names = [f"meta_{s:06d}.json" for s in steps]
    print(f"  probing {len(names)} metadata files by exact name "
          f"({steps[0]}..{steps[-1]} every {every})")

    quoted = " ".join(shlex.quote(n) for n in names)
    script = (f"cd {shlex.quote(ckpt_dir)} || exit 1; "
              f"for f in {quoted}; do "
              f"if [ -r \"$f\" ]; then echo \"@@@ $f\"; cat \"$f\"; fi; done")
    out = remote.ssh(script, check=False)
    if not out.strip():
        print("  nothing readable -- no metadata retrieved", file=sys.stderr)
        return []

    import json
    rows = []
    for block in out.split("@@@ ")[1:]:
        newline = block.find("\n")
        if newline < 0:
            continue
        try:
            meta = json.loads(block[newline + 1:])
        except ValueError:
            continue
        loop = meta.get("loop_state") or {}
        loader = meta.get("dataloader_state_dict") or {}
        rows.append({
            "step": meta.get("step"),
            "val_bpb": meta.get("val_bpb"),
            "min_val_bpb": loop.get("min_val_bpb"),
            "smooth_train_loss": loop.get("smooth_train_loss"),
            "total_training_time_s": loop.get("total_training_time"),
            "epoch": loader.get("epoch"),
            "pq_idx": loader.get("pq_idx"),
        })
    rows = [r for r in rows if r["step"] is not None]
    rows.sort(key=lambda r: r["step"])

    if rows:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        cols = ["step", "val_bpb", "min_val_bpb", "smooth_train_loss",
                "total_training_time_s", "epoch", "pq_idx"]
        with open(out_csv, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join("" if r[c] is None else str(r[c])
                                  for c in cols) + "\n")
        print(f"  recovered {len(rows)} point(s), "
              f"steps {rows[0]['step']}..{rows[-1]['step']}")
        print(f"  wrote {out_csv}")
        print(f"  best val_bpb seen: {min(r['val_bpb'] for r in rows):.6f}")
        # The highest step with readable metadata is the newest checkpoint that
        # exists -- useful because the directory itself cannot be listed.
        print(f"  newest checkpoint present: step {rows[-1]['step']}")
    return rows


def probe(remote, probe_steps=()):
    print("\n[probe] identity and readability")
    for label, cmd in (
            ("login cwd", "pwd"),
            ("identity", "whoami"),
            ("groups", "id -Gn"),
    ):
        print(f"  {label}: {remote.ssh(cmd, check=False).strip()}")

    # Mode/owner of every component on the way down. `ls` on a directory needs
    # READ permission, but reaching a file inside it only needs EXECUTE. So a
    # directory that cannot be listed can still hand over a file whose exact
    # name is known -- which is why this reports the chain instead of giving up
    # on the first "Permission denied".
    print("\n[probe] permission chain")
    chain = ARTIFACT_ROOT.split("/")
    paths = ["/".join(chain[:i + 1]) for i in range(1, len(chain))]
    paths += [f"{ARTIFACT_ROOT}/hindi_lm_train",
              f"{ARTIFACT_ROOT}/hindi_lm_train/base_checkpoints",
              f"{ARTIFACT_ROOT}/hindi_lm_train/base_checkpoints/stable"]
    for path in paths:
        out = remote.ssh(f"stat -c '%A %U:%G' {shlex.quote(path)}",
                         check=False).strip()
        print(f"  {out or 'unreadable'}  {path}")

    print("\n[probe] checkpoint directories")
    for tag, path in CKPT_DIRS.items():
        step, steps = newest_step(remote, path)
        if step is None:
            print(f"  {tag}: cannot LIST {path}")
            continue
        parts = sorted(steps)
        print(f"  {tag}: {len(parts)} step(s) present, newest complete = "
              f"{step}  (range {parts[0]}..{parts[-1]})")

    # Direct-name access, which works even when listing does not.
    if probe_steps:
        print("\n[probe] direct access by exact filename  "
              "(mode | owner:group | bytes | mtime)")
        for step in probe_steps:
            for name in (f"model_{step:06d}.pt", f"optim_{step:06d}_rank0.pt",
                         f"meta_{step:06d}.json"):
                path = f"{CKPT_DIRS['stable']}/{name}"
                info = remote.stat_full(path)
                if not info:
                    print(f"  {name}: NOT REACHABLE")
                    continue
                parts = info.split("|")
                flag = ""
                kind = name.split("_")[0]
                if kind in EXPECTED_SIZES and len(parts) > 2 and \
                        parts[2].isdigit():
                    flag = (" size as expected"
                            if int(parts[2]) == EXPECTED_SIZES[kind]
                            else f" SIZE UNEXPECTED "
                                 f"(want {EXPECTED_SIZES[kind]:,})")
                # Whether the bytes can actually be READ is a separate question
                # from whether stat works, so test it directly on one byte.
                readable = remote.ssh(
                    f"head -c 1 {shlex.quote(path)} > /dev/null && echo yes",
                    check=False).strip() == "yes"
                print(f"  {name}: {info}{flag}   "
                      f"readable: {'YES' if readable else 'NO'}")

    print("\n[probe] loose logs")
    for src in LOG_FILES:
        size, _ = remote.stat_remote(src)
        print(f"  {os.path.basename(src)}: "
              f"{f'{size:,} B ({human(size)})' if size is not None else 'NOT READABLE'}")

    print("\n[probe] candidate log names (exact paths, no glob)")
    for path in LOG_CANDIDATES:
        size, _ = remote.stat_remote(path)
        if size is not None:
            print(f"  {path}: {size:,} B ({human(size)})")
    print("  (only readable candidates are listed)")

    # The artifact root may not be listable, but /workspace is -- and a log
    # written with a relative redirect would have landed there.
    print(f"\n[probe] listing {ARTIFACT_ROOT}")
    print(remote.ssh(f"ls -la {shlex.quote(ARTIFACT_ROOT)}",
                     check=False).rstrip() or "  (no output)")
    print("\n[probe] listing the login cwd (/workspace)")
    print(remote.ssh("ls -la .", check=False).rstrip() or "  (no output)")

    print("\n[probe] already staged in backup/")
    print(remote.ssh(f"du -sh {REMOTE_BACKUP}/* 2>&1 || true",
                     check=False).rstrip() or "  (nothing)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    ap.add_argument("--stage", action="append", choices=STAGE_NAMES,
                    help="only these stages (repeatable); default is all")
    ap.add_argument("--tag", action="append",
                    choices=sorted(CKPT_DIRS),
                    help="which checkpoint dirs to pull (default: all present)")
    ap.add_argument("--step", type=int,
                    help="checkpoint step (default: newest complete trio)")
    ap.add_argument("--probe", action="store_true",
                    help="report identity, readability and what exists; "
                         "transfer nothing")
    ap.add_argument("--val-series", action="store_true",
                    help="rebuild the validation curve from checkpoint "
                         "metadata (works without listing the directory)")
    ap.add_argument("--from-step", type=int, default=22400,
                    help="first metadata step to probe (default 22400: the "
                         "first save_every=400 boundary after the resume)")
    ap.add_argument("--to-step", type=int, default=33750)
    ap.add_argument("--every", type=int, default=400,
                    help="save_every used by the run (default 400)")
    ap.add_argument("--stage-only", action="store_true",
                    help="hard-link into backup/ but do not download")
    ap.add_argument("--no-optim", action="store_true",
                    help="skip optim_*.pt. Pretraining is finished, so the "
                         "optimizer state is only needed to resume pretraining "
                         "-- not for eval, SFT or the demo. It is also the "
                         "largest file (3.8 GB), so skipping it roughly halves "
                         "the transfer when the window is closing")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip md5 (NOT recommended: a truncated checkpoint "
                         "looks plausible and fails confusingly later)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = args.stage or STAGE_NAMES
    control_path = os.path.join(
        os.path.expanduser("~"), ".ssh", "cm-hindi-%r@%h:%p")
    os.makedirs(os.path.dirname(control_path), exist_ok=True)
    remote = Remote(args.host, args.port, control_path, dry_run=args.dry_run)

    print(f"source: {args.host}:{args.port}   local root: {args.local_root}")
    if not args.dry_run:
        remote.open_session()

    if args.probe:
        probe(remote, probe_steps=[args.step] if args.step else ())
        return 0

    if args.val_series:
        print(f"\n[val-series] from checkpoint metadata in "
              f"{CKPT_DIRS['stable']}")
        rows = harvest_val_series(
            remote, CKPT_DIRS["stable"], args.from_step, args.to_step,
            args.every, os.path.join(args.local_root, "val_loss_round2.csv"))
        return 0 if rows else 1

    jobs = []  # (remote_dir, local_dir, [(name, size)])

    if "ckpt" in stages:
        tags = args.tag or sorted(CKPT_DIRS)
        for tag in tags:
            path = CKPT_DIRS[tag]
            step = args.step
            if step is None:
                step, _ = newest_step(remote, path)
                if step is None:
                    print(f"\n[ckpt:{tag}] nothing readable at {path}, skipping")
                    continue
            print(f"\n[ckpt:{tag}] step {step}")
            dest, linked = stage_checkpoint(remote, path, step, tag)
            if args.no_optim:
                linked = [(n, s) for n, s in linked if not n.startswith("optim")]
            local_dir = os.path.join(args.local_root, os.path.basename(dest))
            jobs.append((dest, local_dir, linked))

    if "logs" in stages:
        print("\n[logs]")
        dest, linked = stage_logs(remote)
        jobs.append((dest, os.path.join(args.local_root, "logs"), linked))

    if args.stage_only:
        total = sum(s for _, _, files in jobs for _, s in files)
        print(f"\nstaged {human(total)} into {REMOTE_BACKUP}/. "
              f"Re-run without --stage-only to download.")
        return 0

    # Smallest first inside each job: meta_*.json is ~1.3 KB and is what makes
    # a checkpoint resumable at the right dataloader position, so it should
    # never be the file left behind by a collapsing link.
    failures = []
    moved = 0
    for remote_dir, local_dir, files in jobs:
        if not files:
            continue
        os.makedirs(local_dir, exist_ok=True)
        print(f"\n{remote_dir} -> {local_dir}")
        for name, size in sorted(files, key=lambda f: f[1]):
            print(f"  {name} ({human(size)})", flush=True)
            ok, why = download_file(
                remote, f"{remote_dir}/{name}",
                os.path.join(local_dir, name), verify=not args.no_verify)
            if ok:
                moved += size
                print(f"    {why}")
            else:
                print(f"    FAILED: {why}", file=sys.stderr)
                failures.append((name, why))

    print(f"\ntotal accounted for: {human(moved)}")
    if failures:
        print(f"\n{len(failures)} failure(s) -- RE-RUN THIS SCRIPT to resume "
              f"from the last byte received:", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("all files verified on the laptop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
