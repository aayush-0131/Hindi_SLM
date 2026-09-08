"""
Acquire Sangraha verified/hin (split into web/pdf/speech by the `type` field)
and unverified/hin.

The `type` field's literal values are NOT confirmed (research.md flags this
explicitly) -- `inspect_sangraha_types()` validates them at runtime and RAISES
if they don't match `EXPECTED_TYPE_VALUES`, rather than silently mis-partitioning.
Run it first. If it raises, update EXPECTED_TYPE_VALUES and the mapping in
`_iter_verified` before trusting anything downstream of this file.
"""

from datasets import load_dataset
from data_pipeline.sources.common import normalize_records, write_corpus_docs

# Best-guess literal values -- CONFIRM via inspect_sangraha_types() before trusting this.
EXPECTED_TYPE_VALUES = {"web", "pdf", "speech"}


def inspect_sangraha_types(sample_size=2000):
    """Run this first. Prints the actual `type` values so EXPECTED_TYPE_VALUES /
    the split mapping below can be corrected if the guess is wrong."""
    ds = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)
    seen = set()
    for i, row in enumerate(ds):
        seen.add(row["type"])
        if i >= sample_size:
            break
    print("Observed `type` values in verified/hin:", seen)
    unexpected = seen - EXPECTED_TYPE_VALUES
    if unexpected:
        raise ValueError(
            f"Sangraha `type` field contains unexpected values {unexpected}. "
            f"Update EXPECTED_TYPE_VALUES and the split mapping in sangraha.py "
            f"before proceeding -- do not silently guess the mapping."
        )
    return seen


def _iter_verified(type_value, max_docs=None, max_scan=None, progress_every=20_000):
    """
    Scans the verified/hin stream for rows matching `type_value`, up to
    `max_docs` matches. Types are mixed within one continuous stream (not
    separate splits), so finding N matches can require scanning far more than
    N rows if that type is rare or clustered later in the stream -- max_scan
    bounds the scan itself so this can never hang unbounded. If the scan cap
    is hit before max_docs matches are found, this returns however many were
    actually found (fewer than requested) rather than continuing forever.
    """
    if max_scan is None:
        max_scan = (max_docs * 30) if max_docs is not None else 2_000_000
    ds = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)
    n = 0
    scanned = 0
    for row in ds:
        scanned += 1
        if row["type"] == type_value:
            yield {"doc_id": f"sangraha-verified-{type_value}-{n}", "text": row["text"]}
            n += 1
            if max_docs is not None and n >= max_docs:
                break
        if scanned % progress_every == 0:
            print(f"[sangraha:{type_value}] scanned {scanned} rows, found {n} matches so far...")
        if scanned >= max_scan:
            print(f"[sangraha:{type_value}] scan cap ({max_scan} rows) hit -- found {n}"
                  f"/{max_docs if max_docs is not None else '?'} before stopping. "
                  f"This type may be rare or clustered later in the stream than the cap reached.")
            break


def acquire_all_verified(out_dirs, max_docs=None, max_scan=None, progress_every=20_000):
    """
    Single pass over verified/hin, bucketing rows into web/pdf/speech
    simultaneously -- replaces calling the per-type acquire functions
    separately, which each independently re-scanned the same stream from
    scratch (found on a real run: web+pdf+speech each doing their own up-to-
    max_scan scan of the same underlying data, a real 3x redundancy).

    out_dirs: {"web": path, "pdf": path, "speech": path}
    max_docs: None (unlimited each), a single int (applied to all three), or
              {"web": N, "pdf": N, "speech": N}.
    Returns {"web": count, "pdf": count, "speech": count} -- the number of
    documents actually written per type.
    """
    types = ["web", "pdf", "speech"]
    if max_docs is None:
        max_docs = {t: None for t in types}
    elif isinstance(max_docs, int):
        max_docs = {t: max_docs for t in types}

    if max_scan is None:
        finite_targets = [v for v in max_docs.values() if v is not None]
        max_scan = (max(finite_targets) * 30) if finite_targets else 2_000_000

    buffers = {t: [] for t in types}
    counts = {t: 0 for t in types}
    done = {t: (max_docs[t] == 0) for t in types}

    ds = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)
    scanned = 0
    for row in ds:
        scanned += 1
        t = row["type"]
        if t in types and not done.get(t, True):
            buffers[t].append({"doc_id": f"sangraha-verified-{t}-{counts[t]}", "text": row["text"]})
            counts[t] += 1
            if max_docs[t] is not None and counts[t] >= max_docs[t]:
                done[t] = True
        if scanned % progress_every == 0:
            print(f"[sangraha:verified] scanned {scanned} rows, found so far: "
                  f"web={counts['web']} pdf={counts['pdf']} speech={counts['speech']}...")
        if all(done.values()) or scanned >= max_scan:
            if scanned >= max_scan and not all(done.values()):
                print(f"[sangraha:verified] scan cap ({max_scan} rows) hit -- "
                      f"final counts: web={counts['web']} pdf={counts['pdf']} speech={counts['speech']}")
            break

    results = {}
    for t in types:
        records = normalize_records(iter(buffers[t]), source=f"sangraha_verified_{t}")
        results[t] = write_corpus_docs(records, out_dirs[t])
    return results


def iter_unverified(max_docs=None):
    ds = load_dataset("ai4bharat/sangraha", data_dir="unverified/hin", split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        yield {"doc_id": f"sangraha-unverified-{i}", "text": row["text"]}


def acquire_verified_web(out_dir, max_docs=None):
    records = normalize_records(_iter_verified("web", max_docs), source="sangraha_verified_web")
    return write_corpus_docs(records, out_dir)


def acquire_verified_pdf(out_dir, max_docs=None):
    records = normalize_records(_iter_verified("pdf", max_docs), source="sangraha_verified_pdf")
    return write_corpus_docs(records, out_dir)


def acquire_verified_speech(out_dir, max_docs=None):
    records = normalize_records(_iter_verified("speech", max_docs), source="sangraha_verified_speech")
    return write_corpus_docs(records, out_dir)


def acquire_unverified(out_dir, max_docs=None):
    records = normalize_records(iter_unverified(max_docs), source="sangraha_unverified")
    return write_corpus_docs(records, out_dir)
