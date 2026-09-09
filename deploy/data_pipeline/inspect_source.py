"""Quantitative Devanagari text-health checks, plus samples to actually read.

WHY THIS EXISTS
---------------
ROUND2_HANDOFF.md sec 9.7 has said, since Round 1, "read 100 random documents
per source -- never done". It matters most for Sangraha Verified PDF, which is
OCR'd, ships no OCR-confidence column (research.md: the upstream Setu pipeline
computes such signals but does not expose them), and under Plan C is ~40-49% of
the decay mixture -- the phase that finalizes the shipped model. Devanagari OCR
mojibake would be completely invisible in row counts and obvious in 30 seconds
of reading.

Rather than only dumping text for a human, this computes signals that catch the
specific ways Devanagari extraction fails, so the result is a number you can
put in a report and not just an impression:

  - U+FFFD replacement characters -- unambiguous decode failure
  - ORPHANED COMBINING MARKS: a matra, nukta or virama with no base consonant
    before it. In correct Devanagari every Mn/Mc character follows a base
    letter; one that starts a word (or follows a space/punctuation) means the
    base was lost in extraction. This is the highest-signal check here and the
    one most specific to Indic OCR.
  - DOUBLED VIRAMA -- a sequence Devanagari orthography does not permit, so it
    indicates corruption rather than unusual text
  - `stacked_marks` is REPORTED BUT NOT USED AS A FLAG. Measured on the real
    corpus it runs 7-95 per document across sources that are otherwise clean,
    because mark-after-mark is legitimate and common in Devanagari (a matra
    followed by anusvara or chandrabindu). It is kept for eyeballing only;
    treating it as a corruption signal would flag the entire corpus.
  - script mix at the CHARACTER level (Devanagari / Latin / digit / other)
  - single-character word fraction -- OCR over-segmentation shatters words
  - long identical-character runs -- OCR "stutter"
  - control and private-use characters

None of these individually proves corruption, so the tool reports rates and
lets a threshold decide, and always prints real samples alongside. A clean
Hindi corpus should show orphaned marks in well under 1% of documents.
"""

import collections
import glob
import os
import random
import re
import unicodedata

# Devanagari block plus the extended/Vedic ranges that legitimately appear.
_DEVA_RANGES = ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF))
VIRAMA = "्"
NUKTA = "़"
REPLACEMENT = "�"

_RUN_RE = re.compile(r"(.)\1{4,}")          # same char 5+ times
_WS_RE = re.compile(r"\s+")


def _is_devanagari(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DEVA_RANGES)


def _is_mark(ch):
    return unicodedata.category(ch) in ("Mn", "Mc")


def _is_base_letter(ch):
    """A character a combining mark may legitimately attach to."""
    return unicodedata.category(ch) in ("Lo", "Ll", "Lu", "Lt", "Lm")


def analyse_text(text):
    """Per-document signals. Pure function -- unit-testable without any corpus."""
    n = len(text)
    if n == 0:
        return {"chars": 0, "empty": True}

    deva = latin = digit = other = ctrl = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if _is_devanagari(ch):
            deva += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
        elif ch.isdigit():
            digit += 1
        elif cat.startswith("C") and ch not in "\n\r\t":
            ctrl += 1
        elif not ch.isspace() and not cat.startswith("P"):
            other += 1

    # Orphaned combining marks: a mark whose preceding character is not a base
    # letter (or which begins the text). This is THE Indic-OCR signal.
    orphaned = 0
    doubled_virama = 0
    stacked_marks = 0
    prev = ""
    prev_was_mark = False
    for ch in text:
        if _is_mark(ch):
            if prev == "" or not _is_base_letter(prev):
                # A mark following another mark is "stacked", counted
                # separately -- some sequences (matra + nasal sign) are legal,
                # so this is a weaker signal than a true orphan.
                if prev_was_mark:
                    stacked_marks += 1
                else:
                    orphaned += 1
            prev_was_mark = True
        else:
            prev_was_mark = False
        if ch == VIRAMA and prev == VIRAMA:
            doubled_virama += 1
        prev = ch

    words = _WS_RE.split(text.strip())
    words = [w for w in words if w]
    single_char_words = sum(1 for w in words if len(w) == 1)

    return {
        "chars": n,
        "empty": False,
        "words": len(words),
        "deva_frac": deva / n,
        "latin_frac": latin / n,
        "digit_frac": digit / n,
        "other_frac": other / n,
        "control_chars": ctrl,
        "replacement_chars": text.count(REPLACEMENT),
        "orphaned_marks": orphaned,
        "stacked_marks": stacked_marks,
        "doubled_virama": doubled_virama,
        "char_runs": len(_RUN_RE.findall(text)),
        "single_char_word_frac": (single_char_words / len(words)) if words else 0.0,
        "mean_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
    }


# A document tripping any of these is flagged for human attention. Thresholds
# are deliberately loose -- the goal is to surface real corruption, not to
# fail documents for being unusual.
def flags_for(stats):
    if stats.get("empty"):
        return ["empty"]
    flags = []
    if stats["replacement_chars"] > 0:
        flags.append("replacement_chars")           # decode failure, unambiguous
    if stats["orphaned_marks"] > 2:
        flags.append("orphaned_marks")              # lost base consonants
    if stats["doubled_virama"] > 0:
        flags.append("doubled_virama")              # invalid orthography
    if stats["control_chars"] > 0:
        flags.append("control_chars")
    if stats["deva_frac"] < 0.50:
        flags.append("low_devanagari")              # is this even Hindi?
    if stats["single_char_word_frac"] > 0.30:
        flags.append("over_segmented")              # OCR shattered the words
    if stats["char_runs"] > 3:
        flags.append("char_runs")                   # OCR stutter
    return flags


def sample_sources(shard_glob, sources, sample_size=100, seed=0,
                   text_col="text", source_col="source"):
    """Reservoir-sample every requested source in ONE pass over the shards.

    Reservoir sampling gives a uniform sample in a single pass without holding
    a whole source in memory -- and, unlike taking the first N rows, it does
    not land entirely on whichever shard sorts first. That mistake is what
    produced Round 1's biased tokens/doc estimates (research.md: "the shuffled
    sample landed on the PDF shards").

    Sampling all sources together matters for cost, not just tidiness: the
    corpus is ~15 GB, so a pass per source meant re-reading it seven times.

    `sources` may contain None to mean "everything, ignoring the source
    column". Returns ({source: [(shard, text), ...]}, {source: seen_count}).
    """
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(shard_glob))
    if not paths:
        raise ValueError(f"no parquet files matched {shard_glob!r}")

    wanted = list(sources)
    catch_all = None in wanted
    rngs = {s: random.Random(seed) for s in wanted}
    reservoirs = {s: [] for s in wanted}
    seen = {s: 0 for s in wanted}

    for path in paths:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        has_source = source_col in available
        cols = [text_col] + ([source_col] if has_source else [])
        table = pq.read_table(path, columns=cols)
        texts = table.column(text_col).to_pylist()
        srcs = (table.column(source_col).to_pylist()
                if has_source else [None] * len(texts))
        shard_name = os.path.basename(path)

        for text, src in zip(texts, srcs):
            targets = [None] if catch_all else []
            if not catch_all and src in reservoirs:
                targets = [src]
            for key in targets:
                seen[key] += 1
                res = reservoirs[key]
                if len(res) < sample_size:
                    res.append((shard_name, text))
                else:
                    j = rngs[key].randrange(seen[key])
                    if j < sample_size:
                        res[j] = (shard_name, text)
    return reservoirs, seen


def sample_source(shard_glob, source=None, sample_size=100, seed=0,
                  text_col="text", source_col="source"):
    """Single-source convenience wrapper over `sample_sources`."""
    reservoirs, seen = sample_sources(
        shard_glob, [source], sample_size=sample_size, seed=seed,
        text_col=text_col, source_col=source_col)
    return reservoirs[source], seen[source]


def report(reservoir, seen, source_label, show=5, snippet_chars=600):
    """Aggregate report plus real samples to read."""
    if not reservoir:
        print(f"no documents found for source {source_label!r}")
        return {}

    all_stats = [analyse_text(t) for _, t in reservoir]
    all_flags = [flags_for(s) for s in all_stats]
    flagged = [i for i, f in enumerate(all_flags) if f]
    counter = collections.Counter(f for fl in all_flags for f in fl)
    nonempty = [s for s in all_stats if not s.get("empty")]

    def mean(key):
        return sum(s[key] for s in nonempty) / len(nonempty) if nonempty else 0.0

    print(f"\n{'=' * 72}")
    print(f"SOURCE: {source_label}   sampled {len(reservoir)} of {seen:,} docs")
    print(f"{'=' * 72}")
    print(f"  mean chars/doc            {mean('chars'):10,.0f}")
    print(f"  mean words/doc            {mean('words'):10,.0f}")
    print(f"  mean word length          {mean('mean_word_len'):10.2f}")
    print(f"  Devanagari char fraction  {mean('deva_frac'):10.3f}")
    print(f"  Latin char fraction       {mean('latin_frac'):10.3f}")
    print(f"  digit fraction            {mean('digit_frac'):10.3f}")
    print(f"  single-char word fraction {mean('single_char_word_frac'):10.3f}")
    print(f"  orphaned combining marks  {mean('orphaned_marks'):10.2f} per doc")
    print(f"  stacked marks             {mean('stacked_marks'):10.2f} per doc")
    print(f"  U+FFFD replacement chars  {mean('replacement_chars'):10.2f} per doc")
    print(f"  doubled virama            {mean('doubled_virama'):10.2f} per doc")
    print(f"\n  DOCUMENTS FLAGGED: {len(flagged)}/{len(reservoir)} "
          f"({100 * len(flagged) / len(reservoir):.1f}%)")
    if counter:
        for flag, count in counter.most_common():
            print(f"    {flag:22s} {count:4d} doc(s)")
    else:
        print("    (none)")

    print(f"\n  --- {min(show, len(reservoir))} sample(s) to read ---")
    for i in list(range(len(reservoir)))[:show]:
        shard, text = reservoir[i]
        snippet = _WS_RE.sub(" ", text[:snippet_chars]).strip()
        print(f"\n  [{i}] {shard}  ({all_stats[i].get('chars', 0):,} chars)"
              f"{'  FLAGS: ' + ','.join(all_flags[i]) if all_flags[i] else ''}")
        print(f"      {snippet}")

    if flagged:
        print(f"\n  --- flagged sample(s), read these closely ---")
        for i in flagged[:show]:
            shard, text = reservoir[i]
            snippet = _WS_RE.sub(" ", text[:snippet_chars]).strip()
            print(f"\n  [{i}] {shard}  FLAGS: {','.join(all_flags[i])}")
            print(f"      {snippet}")

    return {
        "source": source_label,
        "sampled": len(reservoir),
        "population": seen,
        "flagged": len(flagged),
        "flag_rate": len(flagged) / len(reservoir),
        "flag_counts": dict(counter),
        "means": {k: mean(k) for k in
                  ("chars", "words", "mean_word_len", "deva_frac", "latin_frac",
                   "digit_frac", "single_char_word_frac", "orphaned_marks",
                   "stacked_marks", "replacement_chars", "doubled_virama")},
    }
