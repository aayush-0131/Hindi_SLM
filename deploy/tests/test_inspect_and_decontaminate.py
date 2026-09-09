"""Tests for the two checks handoff sec 9.6/9.7 flagged as never done.

Both tools can fail SILENTLY in the direction that matters -- an inspection
that reports clean text because its corruption check never fires, or a
decontamination pass that reports a clean corpus because its normaliser never
matches. So the tests here plant known-bad input and require it to be caught.
"""

import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.inspect_source import (  # noqa: E402
    analyse_text, flags_for, sample_sources,
)
from eval_harness.decontaminate import (  # noqa: E402
    EVAL_SOURCES, _item_text, ngrams, normalise,
)

SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string()),
                    ("doc_id", pa.string())])

CLEAN_HINDI = "यह एक सामान्य हिन्दी वाक्य है जिसमें कोई त्रुटि नहीं है।"


# --------------------------------------------------------------- inspection
def test_clean_devanagari_is_not_flagged():
    stats = analyse_text(CLEAN_HINDI * 5)
    assert flags_for(stats) == [], (stats, flags_for(stats))
    assert stats["deva_frac"] > 0.8, stats["deva_frac"]
    assert stats["orphaned_marks"] == 0, stats
    print("  clean Hindi produces no flags: OK")


def test_orphaned_combining_mark_is_caught():
    """THE Indic-OCR signal: a matra with no base consonant before it. In
    correct Devanagari every mark follows a base letter, so one at a word start
    means extraction lost the consonant."""
    # U+093E DEVANAGARI VOWEL SIGN AA at the start of words, no base letter.
    broken = "ा ा ा " + CLEAN_HINDI
    stats = analyse_text(broken)
    assert stats["orphaned_marks"] >= 3, stats
    assert "orphaned_marks" in flags_for(stats), flags_for(stats)
    print(f"  orphaned matras caught ({stats['orphaned_marks']} found): OK")


def test_replacement_char_is_caught():
    stats = analyse_text(CLEAN_HINDI + " �� ")
    assert stats["replacement_chars"] == 2, stats
    assert "replacement_chars" in flags_for(stats)
    print("  U+FFFD decode failure caught: OK")


def test_doubled_virama_is_caught():
    stats = analyse_text("क्््ष " + CLEAN_HINDI)
    assert stats["doubled_virama"] >= 1, stats
    assert "doubled_virama" in flags_for(stats)
    print("  doubled virama caught: OK")


def test_low_devanagari_and_over_segmentation_are_caught():
    assert "low_devanagari" in flags_for(analyse_text("hello world " * 40))
    shattered = analyse_text(" ".join("अ आ इ ई उ ऊ ए ऐ ओ औ".split() * 20))
    assert "over_segmented" in flags_for(shattered), shattered
    print("  low-Devanagari and over-segmentation caught: OK")


def test_empty_text_is_handled():
    assert flags_for(analyse_text("")) == ["empty"]
    print("  empty document handled without dividing by zero: OK")


def test_sample_sources_is_one_pass_and_per_source(tmp_path=None):
    """Regression: sampling used to make one full pass PER source, re-reading
    ~15 GB seven times. Also pins that each reservoir only holds its own
    source's rows."""
    import tempfile
    d = tempfile.mkdtemp()
    try:
        for shard in range(3):
            rows = []
            for i in range(30):
                src = "sangraha_verified_pdf" if i % 3 == 0 else "fineweb2"
                rows.append({"text": f"{src} doc {shard}-{i} " + CLEAN_HINDI,
                             "source": src, "doc_id": f"{shard}-{i}"})
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA),
                           os.path.join(d, f"mix_{shard:05d}.parquet"))

        reservoirs, seen = sample_sources(
            os.path.join(d, "*.parquet"),
            ["sangraha_verified_pdf", "fineweb2"], sample_size=5, seed=1)

        assert seen["sangraha_verified_pdf"] == 30, seen   # 10 per shard x 3
        assert seen["fineweb2"] == 60, seen
        assert len(reservoirs["sangraha_verified_pdf"]) == 5
        assert len(reservoirs["fineweb2"]) == 5
        for _, text in reservoirs["sangraha_verified_pdf"]:
            assert text.startswith("sangraha_verified_pdf"), text[:40]
        for _, text in reservoirs["fineweb2"]:
            assert text.startswith("fineweb2"), text[:40]
        # Samples must not all come from one shard.
        shards = {s for s, _ in reservoirs["fineweb2"]}
        assert len(shards) > 1, shards
        print("  one-pass multi-source sampling, correctly partitioned: OK")
    finally:
        import shutil
        shutil.rmtree(d)


# ----------------------------------------------------------- decontamination
def test_normalise_makes_formatting_differences_match():
    a = normalise("यह  एक, हिन्दी!  वाक्य   है।")
    b = normalise("यह एक हिन्दी वाक्य है")
    assert a == b, (a, b)
    print("  punctuation/whitespace normalised to a common form: OK")


def test_normalise_is_nfc_so_encoding_differences_match():
    """Devanagari can be composed or decomposed; an un-normalised comparison
    would miss a real match between two differently-encoded copies."""
    composed = "नि"          # na + vowel sign i
    decomposed = "नि"
    assert normalise(composed) == normalise(decomposed)
    # क + nukta vs the precomposed क़
    assert normalise("क़") == normalise("क़")
    print("  NFC normalisation unifies composed/decomposed Devanagari: OK")


def test_ngram_overlap_detects_planted_contamination():
    words = normalise(
        "एक दो तीन चार पांच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह").split()
    assert len(words) >= 13
    index = {h: {("task", 0)} for h in ngrams(words, 13)}

    # A document embedding the eval text must hit.
    doc = normalise("कुछ भूमिका " + " ".join(words) + " कुछ अंत")
    assert any(h in index for h in ngrams(doc.split(), 13))

    # Unrelated Hindi of similar length must not.
    other = normalise(
        "सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस "
        "सत्ताईस अट्ठाईस उनतीस तीस").split()
    assert not any(h in index for h in ngrams(other, 13))
    print("  13-gram overlap: planted match found, unrelated text clean: OK")


def test_ngrams_shorter_than_n_produce_nothing():
    assert list(ngrams(["a", "b", "c"], 13)) == []
    print("  documents shorter than n produce no n-grams: OK")


def test_item_text_skips_labels_but_keeps_choices():
    row = {"question": "प्रश्न यहाँ", "choices": ["पहला", "दूसरा"],
           "answerKey": "A", "id": "q-1", "label": 0}
    text = _item_text(row)
    assert "प्रश्न यहाँ" in text and "पहला" in text and "दूसरा" in text
    assert "q-1" not in text and "A" not in text.split()
    print("  eval item text includes choices, excludes ids/labels: OK")


def test_eval_sources_all_declare_a_config_where_required():
    """Regression: three of these repos REQUIRE a config and design.md records
    none, so loading raised "Config name is missing" and two whole tasks were
    silently absent from the report. boolq-hi also has an 'en' config -- the
    wrong pick would decontaminate against ENGLISH BoolQ."""
    by_repo = {repo: cfg for _, repo, cfg, _ in EVAL_SOURCES}
    assert by_repo["ai4bharat/boolq-hi"] == "hi", by_repo
    assert by_repo["facebook/belebele"] == "hin_Deva", by_repo
    assert by_repo["ai4bharat/hellaswag-hi"] == "hi", by_repo
    arc = [cfg for _, repo, cfg, _ in EVAL_SOURCES
           if repo == "ai4bharat/ai2_arc-hi"]
    assert sorted(arc) == ["ARC-Challenge", "ARC-Easy"], arc
    labels = [label for label, *_ in EVAL_SOURCES]
    assert len(labels) == len(set(labels)), labels
    print(f"  all {len(EVAL_SOURCES)} eval sources declare correct configs: OK")


if __name__ == "__main__":
    for fn in [test_clean_devanagari_is_not_flagged,
               test_orphaned_combining_mark_is_caught,
               test_replacement_char_is_caught,
               test_doubled_virama_is_caught,
               test_low_devanagari_and_over_segmentation_are_caught,
               test_empty_text_is_handled,
               test_sample_sources_is_one_pass_and_per_source,
               test_normalise_makes_formatting_differences_match,
               test_normalise_is_nfc_so_encoding_differences_match,
               test_ngram_overlap_detects_planted_contamination,
               test_ngrams_shorter_than_n_produce_nothing,
               test_item_text_skips_labels_but_keeps_choices,
               test_eval_sources_all_declare_a_config_where_required]:
        print(fn.__name__)
        fn()
    print("\nall inspection + decontamination tests passed")
