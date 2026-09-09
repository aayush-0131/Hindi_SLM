"""
Build the "FineWeb-2 top 2% by quality score" decay component (Req 3.7)
from FineWeb-2's OWN SHIPPED METADATA, because the classifier scores that
requirement presupposes do not exist.

THIS IS A PROXY, NOT THE THING REQUIREMENT 3.7 ASKS FOR
-------------------------------------------------------
Requirement 4.1's classifier (`finepdfs_edu_classifier_hin_Deva`) was only ever
run over 5,000 documents. Measured throughput puts a full 22.1M-document pass
at 157-393 GPU-hours against a 28-hour total allocation (ROUND2_HANDOFF.md
sec 4), so per-document quality scores for FineWeb-2 do not exist and cannot be
produced within this project's budget. ROUND2_HANDOFF.md sec 8 Phase B
explicitly offers the choice: "substitute a metadata-based proxy or drop that
component."

This module substitutes. It ranks on metadata FineWeb-2 computed itself at full
corpus scale:

  - `language_score`  -- confidence the document is really Hindi/Devanagari.
                         The primary ranking signal here.
  - `minhash_cluster_size` -- FineWeb-2's own near-duplicate cluster size. 1
                         means the document is not mirrored boilerplate. This
                         is a full-corpus dedup signal, strictly better
                         coverage than anything we could build in memory.
  - length            -- a hard floor, since very short web documents carry
                         little of the long-form character the decay phase is
                         meant to emphasise.

What this does NOT capture is educational or informational quality, which is
what "top 2% by quality score" was actually reaching for. A high
`language_score` means "confidently Hindi", not "worth learning from". Treat
the resulting slice as "clean, unique, long-form Hindi web text" and record it
as a deviation -- `decay_planner.SUBSLICE_SUBSTITUTIONS` already does.

CALIBRATE THEN APPLY
--------------------
"Top 2%" is a quantile, so the threshold is measured on a sample first and then
applied deterministically -- the same split design.md specifies for the
FineWeb-2 classifier, and the same one whose absence was Bug 12.
"""

from data_pipeline.sources.common import normalize_records, write_corpus_docs

DEFAULT_MIN_CHARS = 2000          # long-form floor, 10x add_fineweb2's 200
DEFAULT_MAX_CLUSTER_SIZE = 1      # unique only, vs add_fineweb2's <=20
DEFAULT_KEEP_FRACTION = 0.02      # Requirement 3.7's "top 2%"


def _passes_hard_gates(row, min_chars, max_cluster_size):
    text = row.get("text") or ""
    if len(text) < min_chars:
        return False
    cluster = row.get("minhash_cluster_size")
    # A missing field is treated as "no opinion" and passes, so absent metadata
    # cannot silently reject the entire corpus (the same convention
    # add_fineweb2.py uses).
    if cluster is not None and cluster > max_cluster_size:
        return False
    return True


def calibrate_threshold(sample_size=200_000, keep_fraction=DEFAULT_KEEP_FRACTION,
                        min_chars=DEFAULT_MIN_CHARS,
                        max_cluster_size=DEFAULT_MAX_CLUSTER_SIZE,
                        progress_every=50_000):
    """Find the `language_score` cutoff that keeps `keep_fraction` of the corpus.

    The fraction is of ALL scanned documents, not of those passing the hard
    gates -- otherwise "top 2%" would silently mean "2% of an
    already-filtered pool" and the slice would be far larger than intended.

    Returns (threshold, stats).
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-2", "hin_Deva",
                      split="train", streaming=True)  # train only (Req 4.4)
    scores = []
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > sample_size:
            break
        if _passes_hard_gates(row, min_chars, max_cluster_size):
            value = row.get("language_score")
            if isinstance(value, (int, float)):
                scores.append(value)
        if progress_every and scanned % progress_every == 0:
            print(f"[fw2-top] calibrating: scanned {scanned:,}, "
                  f"{len(scores):,} passed hard gates", flush=True)

    scanned = min(scanned, sample_size)
    if not scores:
        raise ValueError(
            f"no documents passed the hard gates in {scanned:,} rows "
            f"(min_chars={min_chars}, max_cluster_size={max_cluster_size}). "
            f"Loosen them rather than shipping an empty decay component.")

    target_keep = int(scanned * keep_fraction)
    scores.sort()
    if target_keep >= len(scores):
        # The hard gates alone already keep fewer than the target fraction, so
        # language_score cannot tighten it further. Say so instead of returning
        # a threshold that quietly means "everything".
        threshold = scores[0]
        note = (f"hard gates alone keep {len(scores) / scanned:.2%}, which is "
                f"already at or below the {keep_fraction:.2%} target -- "
                f"language_score threshold set to its minimum ({threshold:.4f}) "
                f"and is not the binding filter")
    else:
        threshold = scores[len(scores) - target_keep]
        note = (f"language_score >= {threshold:.4f} keeps {target_keep:,} of "
                f"{scanned:,} scanned ({target_keep / scanned:.2%})")

    stats = {
        "scanned": scanned,
        "passed_hard_gates": len(scores),
        "hard_gate_rate": len(scores) / scanned,
        "target_keep_fraction": keep_fraction,
        "threshold": threshold,
        "note": note,
    }
    return threshold, stats


def iter_top_slice(min_language_score, max_docs=None,
                   min_chars=DEFAULT_MIN_CHARS,
                   max_cluster_size=DEFAULT_MAX_CLUSTER_SIZE,
                   skip_docs=0, progress_every=100_000):
    """Stream FineWeb-2 and yield documents above the calibrated threshold.

    `skip_docs` skips the first N documents of the stream, so this can draw
    from the ~17M documents Round 1 never touched instead of re-selecting from
    the 5.94M already in the stable corpus -- decay data overlapping the stable
    corpus would waste the decay phase re-showing text the model already saw
    four times.
    """
    from datasets import load_dataset
    import time

    ds = load_dataset("HuggingFaceFW/fineweb-2", "hin_Deva",
                      split="train", streaming=True)
    kept = scanned = 0
    started = time.time()
    for row in ds:
        scanned += 1
        if scanned <= skip_docs:
            continue
        if max_docs is not None and kept >= max_docs:
            break
        if not _passes_hard_gates(row, min_chars, max_cluster_size):
            continue
        value = row.get("language_score")
        if value is not None and value < min_language_score:
            continue
        yield {
            "doc_id": str(row.get("id", f"fw2top-{scanned}")),
            "text": row["text"],
        }
        kept += 1
        if progress_every and scanned % progress_every == 0:
            rate = scanned / max(time.time() - started, 1e-9)
            print(f"[fw2-top] scanned {scanned:,} kept {kept:,} "
                  f"({100 * kept / max(scanned - skip_docs, 1):.2f}%) "
                  f"| {rate:,.0f} rows/s", flush=True)


def acquire(out_dir, min_language_score, max_docs=None,
            min_chars=DEFAULT_MIN_CHARS,
            max_cluster_size=DEFAULT_MAX_CLUSTER_SIZE, skip_docs=0):
    records = normalize_records(
        iter_top_slice(min_language_score, max_docs=max_docs,
                       min_chars=min_chars, max_cluster_size=max_cluster_size,
                       skip_docs=skip_docs),
        source="fineweb2_top")
    return write_corpus_docs(records, out_dir)
