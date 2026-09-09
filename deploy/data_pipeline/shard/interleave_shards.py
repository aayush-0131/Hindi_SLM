"""Interleave final parquet shard FILENAMES so sources mix in the training stream.

WHY THIS EXISTS
---------------
`nanochat`'s dataloader calls `list_parquet_files()`, which **sorts filenames**
and streams them in that order (`[:-1]` train, `[-1:]` val). Round 1's shards
sorted as all 100 `fw2_*` then all 26 `sg_*`, so every epoch trained on
FineWeb-2 for ~4,900 steps and only then on Sangraha. Measured cost
(ROUND2_HANDOFF.md sec 9.1):

  - train loss spiked 2.571 -> 3.642 (+42%) at each FineWeb-2 -> Sangraha
    boundary, recurring EVERY epoch (steps 5,500 / 11,500 / 17,500 -- exactly
    6,000 apart, which is the epoch length)
  - ~1,100 of every 6,000 steps spent re-acquiring a distribution the model
    had already seen

This module fixes the ordering by renaming shards, which costs nothing: the
bytes are untouched, so there is no re-upload and no re-tokenization.

THE VALIDATION SPLIT IS DELIBERATELY NOT TOUCHED
------------------------------------------------
`zz_val_00000.parquet` sorts last and is therefore nanochat's val split. It is
left byte-identical and name-identical so that **val bpb stays comparable
across the Round 1 / Round 2 window boundary**. Every emitted train name is
asserted to sort strictly before the first val name.

This is the distinction ROUND2_HANDOFF.md sec 8 Phase A was drawing when it
warned against "regenerating" the corpus: regenerating changes which documents
exist and which shard is val, making val bpb incomparable. Renaming does
neither. What renaming *does* change is that a resumed run's stored
`pq_idx` (a positional index into the sorted list) lands on a different train
shard -- so the run re-sees some documents and skips others within the current
epoch. That is harmless: it is the same corpus, and the epoch boundary
accounting is unchanged.

WHY 7-WAY AND NOT 2-WAY
-----------------------
The `sg_*` block is itself internally source-ordered -- 8 Sangraha-web shards,
then 8 Sangraha-PDF, then 8 unverified, then finepdfs/indiccorp -- so a
two-way fw2-vs-sg interleave would leave the PDF-only and finepdfs-only runs
fully intact. Shards are therefore grouped by DOMINANT SOURCE into 7 groups
and interleaved against each other (see `_interleave_order`).

Round 1's shards are also wildly non-uniform in size (0.7 MB to 340 MB),
because documents-per-shard was held constant while tokens-per-document varies
several fold across sources (OCR'd PDF pages are long, speech transcripts are
short). That made byte-weighting an obvious thing to try -- but it measured
*worse* than count-weighting on both metrics, so it was rejected. The numbers
are in `_interleave_order`'s docstring.

IDEMPOTENT
----------
Running this twice is a no-op. Run it again after adding shards (e.g. via
`add_fineweb2.py`) and it re-interleaves the whole directory, including the new
ones.
"""

import collections
import glob
import json
import os
import uuid

import pyarrow.parquet as pq

# Short, stable filename tags per source. Kept explicit rather than derived so
# a renamed source upstream fails loudly here instead of silently producing a
# new tag and a different sort order.
SOURCE_TAGS = {
    "fineweb2": "fw2",
    "sangraha_verified_web": "sgw",
    "sangraha_verified_pdf": "sgp",
    "sangraha_verified_speech": "sgs",
    "sangraha_unverified": "sgu",
    "finepdfs": "fpd",
    "indiccorpv2": "icv",
    # Decay-phase-only sources (Requirement 3.7) -- distinct row-level
    # `source` values from the decay acquisition scripts, not the stable
    # corpus's plain "fineweb2".
    "fineweb2_top": "fw2t",
    "wikipedia_hi": "wik",
    "fineweb_edu_hindi": "edu",
}

# Emitted train shards use this prefix. It must sort BEFORE the val glob's
# prefix ("zz_"): "mix_" < "zz_" holds, and _assert_val_sorts_last re-checks it
# against the actual val filenames rather than trusting this comment.
TRAIN_PREFIX = "mix"

DEFAULT_VAL_GLOB = "zz_val_*.parquet"


def _shard_stats(path):
    """Read only the `source` column -- parquet is columnar, so this does not
    read the (much larger) text column."""
    table = pq.read_table(path, columns=["source"])
    counts = collections.Counter(table.column("source").to_pylist())
    return {
        "rows": table.num_rows,
        "bytes": os.path.getsize(path),
        "sources": dict(counts),
    }


def survey(corpus_dir, val_glob=DEFAULT_VAL_GLOB):
    """Return (train_shards, val_names). train_shards is a list of dicts with
    name/rows/bytes/sources/dominant, in current sorted order."""
    all_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.parquet")))
    if not all_paths:
        raise ValueError(f"no *.parquet files in {corpus_dir}")

    val_names = sorted(os.path.basename(p)
                       for p in glob.glob(os.path.join(corpus_dir, val_glob)))
    if not val_names:
        raise ValueError(
            f"no files matched val glob {val_glob!r} in {corpus_dir}. Refusing to "
            f"proceed: without a pinned val shard this rename would change which "
            f"shard nanochat treats as validation, breaking val bpb comparability."
        )

    train = []
    for path in all_paths:
        name = os.path.basename(path)
        if name in val_names:
            continue
        stats = _shard_stats(path)
        if not stats["sources"]:
            raise ValueError(f"{name} has no rows / no source values")
        unknown = set(stats["sources"]) - set(SOURCE_TAGS)
        if unknown:
            raise ValueError(
                f"{name} contains unrecognised source(s) {sorted(unknown)}. Add "
                f"them to SOURCE_TAGS deliberately -- an unmapped source would "
                f"otherwise get an arbitrary tag and change the sort order."
            )
        stats["name"] = name
        stats["dominant"] = max(stats["sources"], key=stats["sources"].get)
        train.append(stats)
    return train, val_names


def _interleave_order(train_shards):
    """Order shards so no source occupies a long consecutive run.

    Error diffusion (a.k.a. smooth weighted round robin): every output slot
    credits each group with its share of the remaining slots, then the group
    holding the largest credit is emitted and charged 1. This spreads all
    groups against *each other*, which is the property we need.

    Weighting is by shard COUNT, not bytes. Both were measured on the Round 1
    corpus:

        weighting          longest run    worst 8-shard byte-share deviation
        by count                     6                                 0.254
        by bytes                    17                                 0.272

    Byte weighting exhausts the small-byte groups early and leaves a 17-shard
    FineWeb-2 tail, which is the exact failure mode being fixed. Count
    weighting wins on both metrics, so it is what ships. (Two other algorithms
    were tried and rejected: byte-weighted fair queueing gave a run of 13, and
    per-group stratified placement gave 11 because the three similarly-sized
    Sangraha groups land on near-identical positions and bunch together.)

    For reference, Round 1's ordering had a longest run of 100.

    Ties break on group name, so the ordering is deterministic.
    """
    groups = collections.OrderedDict()
    for shard in train_shards:
        groups.setdefault(shard["dominant"], []).append(shard)
    # Within a group, preserve the existing sorted order.
    for name in groups:
        groups[name].sort(key=lambda s: s["name"])

    total = len(train_shards)
    increment = {name: len(shards) / total for name, shards in groups.items()}
    credit = {name: 0.0 for name in groups}
    queues = {name: list(shards) for name, shards in groups.items()}

    order = []
    for _ in range(total):
        for name in credit:
            credit[name] += increment[name]
        candidates = [name for name in sorted(queues) if queues[name]]
        best = max(candidates, key=lambda name: (credit[name], name))
        order.append(queues[best].pop(0))
        credit[best] -= 1.0
    return order


def _assert_val_sorts_last(new_names, val_names):
    first_val = min(val_names)
    offenders = [n for n in new_names if n >= first_val]
    if offenders:
        raise AssertionError(
            f"{len(offenders)} emitted name(s) sort at or after the val shard "
            f"{first_val!r} (e.g. {offenders[:3]}). nanochat takes the LAST "
            f"sorted shard as val, so this would silently replace the "
            f"validation set."
        )


def plan(corpus_dir, val_glob=DEFAULT_VAL_GLOB):
    """Compute the rename plan without touching anything.

    Returns a dict with `mapping` (old name -> new name, only for shards that
    actually move), `order` (the full new ordering), and `val_names`.
    """
    train, val_names = survey(corpus_dir, val_glob)
    order = _interleave_order(train)

    width = max(5, len(str(len(order))))
    new_names = []
    for idx, shard in enumerate(order):
        tag = SOURCE_TAGS[shard["dominant"]]
        new_names.append(f"{TRAIN_PREFIX}_{idx:0{width}d}_{tag}.parquet")

    if len(set(new_names)) != len(new_names):
        raise AssertionError("computed duplicate target names")
    _assert_val_sorts_last(new_names, val_names)

    for shard, new_name in zip(order, new_names):
        shard["new_name"] = new_name

    mapping = {s["name"]: s["new_name"] for s in order
               if s["name"] != s["new_name"]}
    return {
        "corpus_dir": os.path.abspath(corpus_dir),
        "val_names": val_names,
        "order": order,
        "mapping": mapping,
        "already_interleaved": not mapping,
        "total_train_shards": len(order),
        "total_train_bytes": sum(s["bytes"] for s in order),
        "total_train_rows": sum(s["rows"] for s in order),
    }


def mixing_report(order, window=8):
    """Quantify how well-mixed the ordering is, so the result is measured rather
    than assumed. Returns max same-source run length and the worst
    sliding-window deviation from each source's global byte share."""
    total = sum(s["bytes"] for s in order)
    global_share = collections.defaultdict(float)
    for shard in order:
        global_share[shard["dominant"]] += shard["bytes"] / total

    longest_run = 1
    current_run = 1
    for prev, curr in zip(order, order[1:]):
        if curr["dominant"] == prev["dominant"]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    worst_dev = 0.0
    worst_at = None
    if len(order) >= window:
        for start in range(len(order) - window + 1):
            chunk = order[start:start + window]
            chunk_bytes = sum(s["bytes"] for s in chunk)
            local = collections.defaultdict(float)
            for shard in chunk:
                local[shard["dominant"]] += shard["bytes"] / chunk_bytes
            for source in global_share:
                dev = abs(local[source] - global_share[source])
                if dev > worst_dev:
                    worst_dev = dev
                    worst_at = (start, source)
    return {
        "longest_same_source_run": longest_run,
        "worst_window_share_deviation": worst_dev,
        "worst_window_at": worst_at,
        "window": window,
        "global_byte_share": dict(global_share),
    }


def apply(corpus_dir, val_glob=DEFAULT_VAL_GLOB, manifest_path=None,
          dry_run=False, verbose=True):
    """Compute and (unless dry_run) perform the rename.

    Renames go through unique temporary names first, because the old and new
    name sets overlap -- a direct rename could clobber a shard that has not been
    moved yet.
    """
    result = plan(corpus_dir, val_glob)
    order = result["order"]
    report = mixing_report(order)
    result["mixing_report"] = report

    if verbose:
        before = [s["name"] for s in sorted(order, key=lambda s: s["name"])]
        before_doms = [s["dominant"] for s in
                       sorted(order, key=lambda s: s["name"])]
        before_run = 1
        run = 1
        for prev, curr in zip(before_doms, before_doms[1:]):
            run = run + 1 if curr == prev else 1
            before_run = max(before_run, run)
        print(f"corpus:  {result['corpus_dir']}")
        print(f"  train shards: {result['total_train_shards']}  "
              f"rows: {result['total_train_rows']:,}  "
              f"bytes: {result['total_train_bytes'] / 1e9:.2f} GB")
        print(f"  val (untouched): {', '.join(result['val_names'])}")
        print(f"  longest same-source run: {before_run} -> "
              f"{report['longest_same_source_run']}")
        print(f"  worst {report['window']}-shard window share deviation: "
              f"{report['worst_window_share_deviation']:.3f}")
        print(f"  renames: {len(result['mapping'])}")

    if result["already_interleaved"]:
        if verbose:
            print("  already interleaved -- nothing to do")
        return result

    if dry_run:
        if verbose:
            print("  DRY RUN -- no files touched")
            for shard in order[:12]:
                print(f"    {shard['name']:24s} -> {shard['new_name']}")
            if len(order) > 12:
                print(f"    ... {len(order) - 12} more")
        return result

    # Manifest is written BEFORE renaming so the mapping survives a crash
    # mid-rename and the operation stays reversible.
    if manifest_path:
        with open(manifest_path, "w") as fh:
            json.dump(
                {k: v for k, v in result.items() if k != "order"} |
                {"order": [{kk: vv for kk, vv in s.items()} for s in order]},
                fh, indent=1, default=str)
        if verbose:
            print(f"  manifest: {manifest_path}")

    token = uuid.uuid4().hex[:8]
    staged = []
    for shard in order:
        if shard["name"] == shard["new_name"]:
            continue
        src = os.path.join(corpus_dir, shard["name"])
        tmp = os.path.join(corpus_dir, f".interleave-{token}-{shard['new_name']}")
        os.rename(src, tmp)
        staged.append((tmp, os.path.join(corpus_dir, shard["new_name"])))
    for tmp, dst in staged:
        os.rename(tmp, dst)

    # Post-verify: every expected file present, and the val shard still sorts
    # last in the directory as nanochat will see it.
    present = sorted(os.path.basename(p) for p in
                     glob.glob(os.path.join(corpus_dir, "*.parquet")))
    expected = sorted([s["new_name"] for s in order] + result["val_names"])
    if present != expected:
        raise AssertionError(
            f"post-rename mismatch: {len(present)} present vs {len(expected)} "
            f"expected; missing={sorted(set(expected) - set(present))[:5]} "
            f"unexpected={sorted(set(present) - set(expected))[:5]}"
        )
    if present[-1] not in result["val_names"]:
        raise AssertionError(
            f"post-rename the last sorted shard is {present[-1]!r}, not a val "
            f"shard -- nanochat would use it as the validation set."
        )
    if verbose:
        print(f"  renamed {len(staged)} shards; last sorted shard is "
              f"{present[-1]} (val) -- OK")
    return result
