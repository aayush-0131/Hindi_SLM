"""Detect eval-set contamination in the training corpus (handoff sec 9.6).

WHY THIS EXISTS
---------------
ROUND2_HANDOFF.md sec 9.6: "Add eval-set decontamination if the corpus is
rebuilt. Never done. The corpus has never been checked for n-gram overlap
against ARC-hi, BoolQ-hi, HellaSwag-hi, or Belebele hin_Deva -- which makes any
'we beat Goldfish' claim attackable."

That is the whole point. FineWeb-2, Sangraha Unverified and IndicCorp v2 are
web crawls, and Hindi benchmark sets are themselves largely translations of
English sets that have been on the public web for years. Overlap is plausible,
and if it exists and is unmeasured, the headline comparison against
Goldfish-Hindi is not defensible.

WHAT IT REPORTS, AND WHICH NUMBER MATTERS
-----------------------------------------
Two different things, easily confused:

  1. the fraction of CORPUS DOCUMENTS containing eval text -- nearly
     irrelevant; a handful of documents in 6.25M changes no weights
  2. the fraction of EVAL ITEMS that appear somewhere in the corpus -- the
     number that decides whether a score is trustworthy

(2) is the actionable output. A contaminated eval item should be excluded from
scoring, so this emits the offending item indices per task, ready to feed into
an exclusion list.

METHOD
------
Word-level n-gram overlap at n=13, the GPT-3 convention: an exact 13-word
Devanagari run shared between an eval item and a web document is not
coincidence. Text is normalised (punctuation stripped, whitespace collapsed,
casefolded) so formatting differences do not hide a real match.

Eval items shorter than n words cannot form an n-gram at all -- a short BoolQ
question is a real case. Those fall back to matching their ENTIRE normalised
text, and the count of such items is reported rather than silently dropped.
Both directions of error are therefore visible: this does not claim to detect
paraphrase or translation-level contamination, only verbatim overlap.
"""

import collections
import glob
import os
import re
import unicodedata

DEFAULT_N = 13

# Underlying datasets behind the lighteval task names in eval_harness/config.py.
#
# UNVERIFIED, and deliberately overridable. design.md lists these repo ids under
# Allowed Dependencies, but research.md never confirmed them and lighteval's
# own registry has been shown to disagree with published task names for other
# languages (issues #606, #509). A repo that does not resolve is reported as a
# LOAD FAILURE, never silently skipped -- a task quietly missing from a
# decontamination report is worse than no report, because it reads as "clean".
# CONFIG NAMES VERIFIED 2026-09-06 by probing the Hub. All four repos exist,
# but three REQUIRE a config that design.md does not record, and loading them
# without one raises "Config name is missing":
#   ai4bharat/ai2_arc-hi   -> 'ARC-Challenge' | 'ARC-Easy'
#   ai4bharat/boolq-hi     -> 'en' | 'gu' | 'hi' | 'ml' | 'mr' | 'ta'
#   ai4bharat/hellaswag-hi -> 'hi'
# boolq-hi's 'en' config is the trap: defaulting wrong there would silently
# decontaminate against ENGLISH BoolQ and report a clean Hindi corpus.
# ARC ships Easy and Challenge separately, matching the two distinct lighteval
# tasks in eval_harness/config.py.
EVAL_SOURCES = [
    # (label, hf_repo, config, splits)
    ("arc_hi_easy",      "ai4bharat/ai2_arc-hi",   "ARC-Easy",
     ("test", "validation", "train")),
    ("arc_hi_challenge", "ai4bharat/ai2_arc-hi",   "ARC-Challenge",
     ("test", "validation", "train")),
    ("boolq_hi",         "ai4bharat/boolq-hi",     "hi",
     ("validation", "test", "train")),
    ("hellaswag_hi",     "ai4bharat/hellaswag-hi", "hi",
     ("validation", "test")),
    ("belebele_hi",      "facebook/belebele",      "hin_Deva",
     ("test",)),
]

# Fields that are labels/ids rather than text, so they neither identify
# contamination nor should be searched for.
_SKIP_FIELDS = {"id", "idx", "index", "label", "answer", "answerkey",
                "answer_key", "gold", "link", "url", "split", "language",
                "dialect", "question_number", "correct_answer_num"}

# Devanagari PUNCTUATION lives inside the Devanagari block, so a naive
# "keep everything in the Devanagari range" rule preserves it -- which would
# leave danda attached to words and block n-gram matches at sentence
# boundaries ("है।" vs "है ."). Stripped explicitly, before the general pass.
#   U+0964 danda, U+0965 double danda, U+0970 abbreviation sign,
#   U+0971 high spacing dot, U+A8CE/U+A8CF Devanagari Extended dandas
_DEVA_PUNCT = dict.fromkeys(
    (0x0964, 0x0965, 0x0970, 0x0971, 0xA8CE, 0xA8CF), " ")

_PUNCT_RE = re.compile(
    r"[^\wऀ-ॿ꣠-ꣿ᳐-᳿]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise(text):
    """Casefold, strip punctuation, collapse whitespace, NFC-normalise.

    NFC matters for Devanagari specifically: the same visual text can be
    encoded with composed or decomposed forms, and an un-normalised comparison
    would miss a real match between two differently-encoded copies.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    text = text.translate(_DEVA_PUNCT)      # danda etc. -- see _DEVA_PUNCT
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def ngrams(words, n):
    for i in range(len(words) - n + 1):
        yield hash(tuple(words[i:i + n]))


def _item_text(row):
    """Concatenate a row's text-bearing fields, schema-agnostically.

    Field names differ across these four datasets and none of the schemas were
    verified, so every string (and list-of-string) field that is not obviously
    a label or id is used. That is deliberately over-inclusive: a missed field
    would make a contaminated item look clean.
    """
    parts = []
    for key, value in row.items():
        if key.lower() in _SKIP_FIELDS:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(v for v in value if isinstance(v, str))
        elif isinstance(value, dict):
            parts.extend(v for v in value.values() if isinstance(v, str))
    return " ".join(parts)


def build_eval_index(sources=None, n=DEFAULT_N, verbose=True):
    """{ngram_hash: {(task, item_index), ...}} plus per-task metadata."""
    from datasets import load_dataset

    sources = sources or EVAL_SOURCES
    index = collections.defaultdict(set)
    meta = {}
    failures = []

    for label, repo, config, splits in sources:
        loaded = 0
        short = 0
        try:
            rows = []
            for split in splits:
                try:
                    ds = (load_dataset(repo, config, split=split)
                          if config else load_dataset(repo, split=split))
                except Exception:
                    continue          # a missing split is normal; a missing repo is not
                rows.extend(list(ds))
            if not rows:
                raise ValueError(f"no rows loaded from splits {splits}")
        except Exception as exc:
            failures.append((label, repo, repr(exc)))
            if verbose:
                print(f"  [{label}] LOAD FAILED: {exc}")
            continue

        for i, row in enumerate(rows):
            words = normalise(_item_text(row)).split()
            if len(words) >= n:
                for h in ngrams(words, n):
                    index[h].add((label, i))
            elif words:
                # Too short for an n-gram: match the whole item instead.
                short += 1
                index[hash(tuple(words))].add((label, i))
            loaded += 1

        meta[label] = {"repo": repo, "items": loaded,
                       "items_too_short_for_ngram": short}
        if verbose:
            print(f"  [{label}] {loaded:,} items "
                  f"({short:,} shorter than {n} words -> whole-text match)")

    return dict(index), meta, failures


_INDEX = None      # per-worker global, set by _init


def _init(index):
    global _INDEX
    _INDEX = index


def scan_shard(args):
    """Worker: return (shard, docs_scanned, {source: hit_docs}, hit_items)."""
    import pyarrow.parquet as pq

    path, n, text_col, source_col = args
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cols = [text_col] + ([source_col] if source_col in names else [])
    table = pq.read_table(path, columns=cols)
    texts = table.column(text_col).to_pylist()
    srcs = (table.column(source_col).to_pylist()
            if source_col in names else [None] * len(texts))

    hit_docs = collections.Counter()
    hit_items = collections.defaultdict(set)
    for text, src in zip(texts, srcs):
        words = normalise(text).split()
        if len(words) < n:
            continue
        matched = set()
        for h in ngrams(words, n):
            found = _INDEX.get(h)
            if found:
                matched |= found
        if matched:
            hit_docs[src or "unknown"] += 1
            for task, item in matched:
                hit_items[task].add(item)
    return os.path.basename(path), len(texts), hit_docs, hit_items


def scan_corpus(corpus_glob, index, n=DEFAULT_N, workers=None,
                text_col="text", source_col="source", limit_shards=None,
                verbose=True):
    import multiprocessing as mp

    paths = sorted(glob.glob(corpus_glob))
    if not paths:
        raise ValueError(f"no parquet files matched {corpus_glob!r}")
    if limit_shards:
        paths = paths[:limit_shards]

    jobs = [(p, n, text_col, source_col) for p in paths]
    total_docs = 0
    hit_docs = collections.Counter()
    hit_items = collections.defaultdict(set)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1:
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(workers, initializer=_init, initargs=(index,)) as pool:
            for i, (name, ndocs, hd, hi) in enumerate(
                    pool.imap_unordered(scan_shard, jobs), 1):
                total_docs += ndocs
                hit_docs.update(hd)
                for task, items in hi.items():
                    hit_items[task] |= items
                if verbose and i % 10 == 0:
                    print(f"    {i}/{len(jobs)} shards, {total_docs:,} docs, "
                          f"{sum(hit_docs.values()):,} hits", flush=True)
    else:
        _init(index)
        for i, job in enumerate(jobs, 1):
            name, ndocs, hd, hi = scan_shard(job)
            total_docs += ndocs
            hit_docs.update(hd)
            for task, items in hi.items():
                hit_items[task] |= items
    return total_docs, hit_docs, {k: sorted(v) for k, v in hit_items.items()}
