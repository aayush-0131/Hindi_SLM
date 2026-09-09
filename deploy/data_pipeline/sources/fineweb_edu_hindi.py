"""
Acquire `KathirKs/fineweb-edu-hindi` -- DECAY B ONLY (Requirements 3.3, 3.7).

Requirement 3.3 is explicit: this source is restricted to the decay phase and
must never enter the stable-phase base mixture. Requirement 3.7 gives it 15% of
Decay B and 0% of Decay A -- that difference is the entire point of running two
variants. Decay A is native-only; Decay B adds this machine-translation-adjacent
educational slice, and Requirement 8.2 settles empirically which wins rather
than assuming.

`mixture/decay_planner.py` enforces the A-exclusion. Callers should not acquire
this for Decay A at all.

Requirements.md sizes the "top 0.5%" slice at ~1.5B tokens, which makes this the
one decay source with genuinely abundant supply -- unlike Hindi Wikipedia and
Sangraha Speech, which cannot fill their nominal shares.

Quality selection: the dataset ships an educational-quality score per document.
The field name is NOT confirmed against the live repo (research.md never
verified this source's schema -- it was excluded from the base mixture, so it
was never acquired in Round 1). `SCORE_FIELD_CANDIDATES` is therefore tried in
order and the adapter FAILS LOUDLY listing the available columns if none match,
rather than silently degrading to unfiltered acquisition -- an unscored
"top 0.5%" slice would not be the thing Requirement 3.7 asks for.
"""

from data_pipeline.sources.common import normalize_records, write_corpus_docs

REPO = "KathirKs/fineweb-edu-hindi"

# Tried in order. The FineWeb-edu family has used several names across releases.
SCORE_FIELD_CANDIDATES = ("score", "edu_score", "fw_edu_score",
                          "fw_edu_v2_score", "educational_score")


def _pick_score_field(row):
    for name in SCORE_FIELD_CANDIDATES:
        if name in row and isinstance(row[name], (int, float)):
            return name
    raise KeyError(
        f"{REPO}: none of {SCORE_FIELD_CANDIDATES} present as a numeric field. "
        f"Available columns: {sorted(row.keys())}. Requirement 3.7 asks for the "
        f"top-0.5% slice by quality score, so acquiring this unfiltered would "
        f"not satisfy it -- pick the right field name and pass score_field=."
    )


def iter_fineweb_edu_hindi(max_docs=None, min_score=None, score_field=None,
                           min_chars=200):
    """Stream the dataset, optionally keeping only documents at or above
    `min_score`. Pass min_score=None to take documents in stream order."""
    from datasets import load_dataset

    ds = load_dataset(REPO, split="train", streaming=True)
    kept = 0
    for i, row in enumerate(ds):
        if max_docs is not None and kept >= max_docs:
            break
        text = row.get("text") or ""
        if len(text) < min_chars:
            continue
        if min_score is not None:
            field = score_field or _pick_score_field(row)
            score_field = field
            if row.get(field) is None or row[field] < min_score:
                continue
        yield {
            "doc_id": f"fineweb_edu_hindi-{row.get('id', i)}",
            "text": text,
        }
        kept += 1


def calibrate_score_threshold(sample_size=20_000, keep_fraction=0.005,
                              score_field=None):
    """Find the score cutoff that keeps the top `keep_fraction` of a sample.

    Requirement 3.7's "top 0.5%" is a quantile, not an absolute score, and the
    score distribution of this dataset is not documented anywhere in
    research.md -- so it is measured rather than guessed. Mirrors the
    calibrate-then-apply split already used for the FineWeb-2 classifier
    (design.md's Batch/Job Contract), so the threshold is measured once and
    then applied deterministically.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO, split="train", streaming=True)
    scores = []
    field = score_field
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        if field is None:
            field = _pick_score_field(row)
        value = row.get(field)
        if isinstance(value, (int, float)):
            scores.append(value)
    if not scores:
        raise ValueError(f"{REPO}: no numeric scores in the first "
                         f"{sample_size} rows under field {field!r}")
    scores.sort()
    # keep_fraction from the TOP, so index from the end.
    cut_index = max(0, int(len(scores) * (1.0 - keep_fraction)) - 1)
    return scores[cut_index], field, len(scores)


def acquire(out_dir, max_docs=None, min_score=None, score_field=None,
            min_chars=200):
    records = normalize_records(
        iter_fineweb_edu_hindi(max_docs=max_docs, min_score=min_score,
                               score_field=score_field, min_chars=min_chars),
        source="fineweb_edu_hindi")
    return write_corpus_docs(records, out_dir)
