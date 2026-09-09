"""Resumable, md5-verified upload of the Round 2 artifacts to the GPU host.

WHY THIS EXISTS
---------------
Round 1's download failed in exactly the way this script is built to survive.
From ROUND2_HANDOFF.md sec 6.1:

  - `scp` has NO resume. On a degrading link a 6.5 GB file is a coin flip --
    and the link did degrade, from 1.7 MB/s to ~22 KB/s, which is why
    `optim_022000_rank0.pt` never arrived and Round 2 has to resume from step
    21,200 instead of 22,000.
  - Absolute `/data1/...` scp destinations fail with a confusing "No such file
    or directory". Only the relative `:./` form works. Two stale uploads went
    undetected because of this.
  - Every retry cost a password prompt.
  - A silently truncated file looks plausible and fails confusingly much later.

So: chunk large files so a drop costs one chunk rather than the whole file,
verify every byte with md5 on both ends, skip anything already correct so
re-running is cheap, multiplex one SSH connection so the whole run needs one
password, and only ever use the `:./` relative form.

Round 2 direction is UPLOAD (~20.5 GB: 14 GB corpus + 6.5 GB checkpoint),
against a 15-hour GPU-off window. At the good-case 1.7 MB/s that is ~3.4h, so
starting early matters more than going fast.

PRIORITY ORDER IS DELIBERATE
----------------------------
Stages run smallest-and-most-irreplaceable first, so that if the link collapses
partway the surviving transfer is the one that matters most. The tokenizer is
773 KB and irreplaceable in practice; the corpus is 14 GB and could in
principle be rebuilt (at the cost of val-bpb comparability).

USAGE
-----
    # see what would move, contact nothing
    python3 ops/transfer.py --plan-only

    # go (one password prompt, resumable -- just re-run it if it drops)
    python3 ops/transfer.py

    # only the checkpoint
    python3 ops/transfer.py --stage ckpt
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

# Files at or above this size are split, so an interrupted transfer costs one
# chunk instead of restarting the whole file.
CHUNK_THRESHOLD = 600 * 1024 * 1024
CHUNK_SIZE = 400 * 1024 * 1024

# Stage order is the priority order -- see build_stages().
STAGE_NAMES = ["code", "tokenizer", "logs", "ckpt", "corpus"]


def md5_local(path, blocksize=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(blocksize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class Remote:
    """Thin ssh/scp wrapper with connection multiplexing.

    ControlMaster means the whole run authenticates once. Without it, every
    chunk of every retry is another password prompt -- which is what made
    Round 1's retries so expensive.
    """

    def __init__(self, host, port, control_path, dry_run=False):
        self.host = host
        self.port = port
        self.control_path = control_path
        self.dry_run = dry_run
        self._base = [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path}",
            "-o", "ControlPersist=8h",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
        ]

    def ssh(self, command, check=True, capture=True):
        argv = ["ssh"] + self._base + ["-p", self.port, self.host, command]
        if self.dry_run:
            print(f"    [dry-run] ssh: {command}")
            return ""
        proc = subprocess.run(argv, capture_output=capture, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"ssh failed ({proc.returncode}): {command}\n"
                f"{(proc.stderr or '').strip()}")
        return (proc.stdout or "") if capture else ""

    def scp_up(self, local_path, remote_rel):
        """Upload one file. remote_rel MUST be relative -- the `:./` form is the
        only one that works on this host (ROUND2_HANDOFF.md sec 2)."""
        assert not remote_rel.startswith("/"), (
            f"refusing absolute remote path {remote_rel!r}: absolute scp "
            f"destinations fail confusingly on this host; use the :./ form")
        target = f"{self.host}:./{remote_rel}"
        argv = ["scp"] + self._base + ["-P", self.port, local_path, target]
        if self.dry_run:
            print(f"    [dry-run] scp: {os.path.basename(local_path)} -> {target}")
            return True
        return subprocess.run(argv).returncode == 0

    def open_session(self):
        """Force the master connection up once, so the password prompt happens
        here rather than in the middle of a transfer loop."""
        if self.dry_run:
            print("  [dry-run] would open multiplexed ssh session")
            return
        print("  opening multiplexed ssh session (one auth for the whole run)")
        argv = ["ssh"] + self._base + ["-p", self.port, self.host,
                                       "echo connected; pwd"]
        proc = subprocess.run(argv, text=True, capture_output=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"cannot reach {self.host}:{self.port}\n"
                f"{(proc.stderr or '').strip()}\n\n"
                f"If the node is not provisioned yet this is expected -- "
                f"re-run when the window opens.")
        print(f"  connected. remote cwd: {proc.stdout.strip().splitlines()[-1]}")

    def remote_md5s(self, remote_dir):
        """{filename: md5} for a remote directory; {} if it does not exist."""
        out = self.ssh(
            f"if [ -d {shlex.quote(remote_dir)} ]; then "
            f"cd {shlex.quote(remote_dir)} && md5sum * 2>/dev/null; fi",
            check=False)
        result = {}
        for line in (out or "").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[1].strip()] = parts[0].strip()
        return result


def split_file(path, chunk_dir, chunk_size=CHUNK_SIZE):
    """Split into chunk_dir/<name>.part-NNN, reusing any already-correct parts."""
    os.makedirs(chunk_dir, exist_ok=True)
    name = os.path.basename(path)
    total = os.path.getsize(path)
    chunks = []
    with open(path, "rb") as fh:
        index = 0
        while True:
            offset = fh.tell()
            if offset >= total:
                break
            data = fh.read(chunk_size)
            if not data:
                break
            part = os.path.join(chunk_dir, f"{name}.part-{index:03d}")
            if not (os.path.exists(part) and os.path.getsize(part) == len(data)):
                with open(part, "wb") as out:
                    out.write(data)
            chunks.append(part)
            index += 1
    return chunks


def plan_stage(local_dir, remote_md5s):
    """Return (to_send, skipped) for a directory, comparing md5s.

    A file whose remote md5 already matches is skipped -- that is what makes
    re-running after a dropped link cheap instead of a full restart.
    """
    to_send, skipped = [], []
    if not os.path.isdir(local_dir):
        return to_send, skipped
    for name in sorted(os.listdir(local_dir)):
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        size = os.path.getsize(path)
        if name in remote_md5s and remote_md5s[name] == md5_local(path):
            skipped.append((name, size))
        else:
            to_send.append((name, size))
    return to_send, skipped


def send_file(remote, local_path, remote_dir, chunk_dir, verbose=True):
    """Send one file, chunking if large, then verify md5 on the remote."""
    name = os.path.basename(local_path)
    size = os.path.getsize(local_path)
    expected = md5_local(local_path)
    started = time.time()

    if size < CHUNK_THRESHOLD:
        ok = remote.scp_up(local_path, f"{remote_dir}/{name}")
        if not ok and not remote.dry_run:
            return False, "scp failed"
    else:
        parts = split_file(local_path, chunk_dir)
        if verbose:
            print(f"    {name}: {size / 1e9:.2f} GB in {len(parts)} chunks")
        remote_parts = remote.remote_md5s(f"{remote_dir}/.parts")
        remote.ssh(f"mkdir -p {shlex.quote(remote_dir)}/.parts")
        for idx, part in enumerate(parts):
            pname = os.path.basename(part)
            if remote_parts.get(pname) == md5_local(part):
                if verbose:
                    print(f"      [{idx + 1}/{len(parts)}] {pname} already "
                          f"present, skipping")
                continue
            if verbose:
                print(f"      [{idx + 1}/{len(parts)}] {pname}", flush=True)
            if not remote.scp_up(part, f"{remote_dir}/.parts/{pname}"):
                return False, f"chunk {pname} failed -- re-run to resume"
        # Reassemble in index order. The glob is sorted so part-000 comes first;
        # zero-padding to 3 digits is what makes lexical order correct.
        remote.ssh(
            f"cd {shlex.quote(remote_dir)} && "
            f"cat .parts/{shlex.quote(name)}.part-* > {shlex.quote(name)}")

    if remote.dry_run:
        return True, "dry-run"

    got = remote.ssh(
        f"md5sum {shlex.quote(remote_dir)}/{shlex.quote(name)} 2>/dev/null "
        f"| awk '{{print $1}}'", check=False).strip()
    if got != expected:
        return False, f"md5 mismatch (local {expected}, remote {got or 'missing'})"

    elapsed = max(time.time() - started, 1e-9)
    if verbose:
        print(f"    {name}: verified ({size / 1e6:.0f} MB, "
              f"{size / elapsed / 1e6:.2f} MB/s)")
    # Chunks are only removed once the reassembled file is md5-verified.
    if size >= CHUNK_THRESHOLD:
        remote.ssh(f"rm -f {shlex.quote(remote_dir)}/.parts/"
                   f"{shlex.quote(name)}.part-*", check=False)
    return True, "ok"


def send_tree_as_tar(remote, local_dir, remote_dir, chunk_dir, verbose=True):
    """Send a whole directory tree as one tarball.

    `deploy/` is ~40 small files across 5 package subdirectories. Per-file scp
    would be 40 round trips, and a flat file listing would silently miss the
    subdirectories entirely -- so the tree goes as a single tarball, which is
    also what ROUND2_HANDOFF.md sec 8 recommends for many-file transfers.
    Verified by the tarball's own md5, then unpacked.
    """
    os.makedirs(chunk_dir, exist_ok=True)
    base = os.path.basename(local_dir.rstrip(os.sep))
    tarball = os.path.join(chunk_dir, f"{base}.tar")
    argv = [
        "tar", "cf", tarball,
        "-C", os.path.dirname(os.path.abspath(local_dir.rstrip(os.sep))),
        "--exclude", "__pycache__", "--exclude", "*.pyc",
        "--exclude", ".DS_Store",
        base,
    ]
    if remote.dry_run:
        print(f"    [dry-run] would tar {local_dir} -> {tarball} and unpack "
              f"into :./ (contains the package subdirectories)")
        return True, "dry-run"
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"tar failed: {(proc.stderr or '').strip()}"

    expected = md5_local(tarball)
    remote.ssh("mkdir -p .transfer-tmp")
    if not remote.scp_up(tarball, f".transfer-tmp/{base}.tar"):
        return False, "tarball scp failed"
    got = remote.ssh(
        f"md5sum .transfer-tmp/{base}.tar 2>/dev/null | awk '{{print $1}}'",
        check=False).strip()
    if got != expected:
        return False, f"tarball md5 mismatch (local {expected}, remote {got or 'missing'})"
    # Unpack over the target. Extracting into the parent of remote_dir means
    # the tarball's own top-level directory name lands as remote_dir.
    parent = os.path.dirname(remote_dir) or "."
    remote.ssh(f"mkdir -p {shlex.quote(parent)} && "
               f"tar xf .transfer-tmp/{base}.tar -C {shlex.quote(parent)}")
    count = remote.ssh(
        f"find {shlex.quote(remote_dir)} -name '*.py' | wc -l",
        check=False).strip()
    if verbose:
        print(f"    {base}.tar verified and unpacked "
              f"({os.path.getsize(tarball) / 1e6:.1f} MB, "
              f"{count} .py files on remote)")
    remote.ssh(f"rm -f .transfer-tmp/{base}.tar", check=False)
    return True, "ok"


def build_stages(local_root, repo_root):
    """(stage name, local dir, remote dir) in priority order.

    Smallest-and-most-irreplaceable first, so a link that collapses partway
    leaves the things that cannot be rebuilt already on the far side. `code`
    and `tokenizer` are together ~1 MB; the corpus is 14 GB and is last.
    """
    return [
        ("code", os.path.join(repo_root, "deploy") if repo_root else "",
         "deploy"),
        ("tokenizer", os.path.join(local_root, "tokenizer"), "backup/tokenizer"),
        ("logs", os.path.join(local_root, "logs"), "backup/logs"),
        ("ckpt", os.path.join(local_root, "ckpt"), "backup/ckpt"),
        ("corpus", os.path.join(local_root, "corpus"), "backup/corpus"),
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT,
                    help=f"laptop backup root (default {DEFAULT_LOCAL_ROOT})")
    ap.add_argument("--repo-root",
                    default=os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    help="project root, for uploading deploy/")
    ap.add_argument("--stage", action="append", choices=STAGE_NAMES,
                    help="only these stages (repeatable); default is all, in "
                         "priority order")
    ap.add_argument("--chunk-dir",
                    default=os.path.join(DEFAULT_LOCAL_ROOT, ".chunks"),
                    help="scratch space for split chunks")
    ap.add_argument("--plan-only", action="store_true",
                    help="show what would transfer; still needs to reach the "
                         "host to read remote md5s")
    ap.add_argument("--dry-run", action="store_true",
                    help="contact nothing at all")
    args = ap.parse_args()

    stages = build_stages(args.local_root, args.repo_root)
    if args.stage:
        stages = [s for s in stages if s[0] in args.stage]

    control_path = os.path.join(
        os.path.expanduser("~"), ".ssh", "cm-hindi-%r@%h:%p")
    os.makedirs(os.path.dirname(control_path), exist_ok=True)
    remote = Remote(args.host, args.port, control_path, dry_run=args.dry_run)

    print(f"target: {args.host}:{args.port}  (relative :./ form only)")
    if not args.dry_run:
        remote.open_session()

    total_send = total_skip = 0
    failures = []
    for name, local_dir, remote_dir in stages:
        print(f"\n[{name}] {local_dir} -> :./{remote_dir}")
        if not local_dir or not os.path.isdir(local_dir):
            print("  local directory missing, skipping")
            continue
        # The code tree has subdirectories, so it goes as one tarball rather
        # than a flat per-file listing (which would silently omit them).
        if name == "code":
            if args.plan_only or args.dry_run:
                send_tree_as_tar(remote, local_dir, remote_dir, args.chunk_dir)
                continue
            ok, why = send_tree_as_tar(remote, local_dir, remote_dir,
                                       args.chunk_dir)
            if not ok:
                print(f"    FAILED {name}: {why}", file=sys.stderr)
                failures.append((name, "tree", why))
            continue

        existing = {} if args.dry_run else remote.remote_md5s(remote_dir)
        to_send, skipped = plan_stage(local_dir, existing)
        send_bytes = sum(s for _, s in to_send)
        skip_bytes = sum(s for _, s in skipped)
        total_send += send_bytes
        total_skip += skip_bytes
        print(f"  {len(to_send)} file(s) to send ({send_bytes / 1e9:.2f} GB); "
              f"{len(skipped)} already verified ({skip_bytes / 1e9:.2f} GB)")
        if args.plan_only or args.dry_run:
            for fname, size in to_send[:10]:
                print(f"    would send {fname} ({size / 1e6:.0f} MB)")
            if len(to_send) > 10:
                print(f"    ... {len(to_send) - 10} more")
            continue
        if to_send:
            remote.ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        for fname, _ in to_send:
            ok, why = send_file(remote, os.path.join(local_dir, fname),
                                remote_dir, args.chunk_dir)
            if not ok:
                print(f"    FAILED {fname}: {why}", file=sys.stderr)
                failures.append((name, fname, why))

    print(f"\ntotal: {total_send / 1e9:.2f} GB to send, "
          f"{total_skip / 1e9:.2f} GB already verified")
    if failures:
        print(f"\n{len(failures)} failure(s) -- re-run this script to resume; "
              f"verified files and chunks are skipped:", file=sys.stderr)
        for stage, fname, why in failures:
            print(f"  {stage}/{fname}: {why}", file=sys.stderr)
        return 1
    if not (args.plan_only or args.dry_run):
        print("\nall stages verified. Next: run ops/bringup.sh on the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
