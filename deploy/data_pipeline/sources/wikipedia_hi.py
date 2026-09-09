"""
Acquire Hindi Wikipedia -- a DECAY-PHASE-ONLY source (Requirement 3.7).

It is deliberately absent from the stable mixture (Requirement 3.6 lists seven
sources and this is not one of them). Requirement 3.7 gives it 24% of Decay A
and 20% of Decay B.

Size reality, and why the 24%/20% targets are not directly achievable:
requirements.md records 169,067 articles. Hindi Wikipedia is small in absolute
terms -- on the order of 100M tokens total -- so at any decay budget above
~400M tokens its nominal share cannot be filled without repeating documents
several times over. That is a supply constraint, not a bug in this adapter:
this module acquires whatever exists and reports it, and
`mixture/decay_planner.py` recomputes the realized shares from measured
availability with a bounded upsampling factor. Do not "fix" a shortfall here.

Source repo: `wikimedia/wikipedia`, the maintained dump-derived parquet release
(plain `text` per article, already wikitext-stripped), config `20231101.hi`.
"""

from data_pipeline.sources.common import normalize_records, write_corpus_docs

# The wikimedia/wikipedia repo is published per (dump date, language). A date
# that no longer exists raises rather than silently falling back, so the
# configured value is explicit and checkable.
DEFAULT_CONFIG = "20231101.hi"


def iter_wikipedia_hi(max_docs=None, config=DEFAULT_CONFIG, min_chars=200):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train",
                      streaming=True)
    kept = 0
    for i, row in enumerate(ds):
        if max_docs is not None and kept >= max_docs:
            break
        text = row.get("text") or ""
        # Stubs are a large fraction of any Wikipedia and carry almost no
        # signal; the same >=200-char floor add_fineweb2.py uses is applied so
        # the two are comparable.
        if len(text) < min_chars:
            continue
        title = row.get("title") or ""
        yield {
            "doc_id": f"wikipedia_hi-{row.get('id', i)}",
            "text": (f"{title}\n\n{text}" if title else text),
        }
        kept += 1


def acquire(out_dir, max_docs=None, config=DEFAULT_CONFIG, min_chars=200):
    records = normalize_records(
        iter_wikipedia_hi(max_docs=max_docs, config=config,
                          min_chars=min_chars),
        source="wikipedia_hi")
    return write_corpus_docs(records, out_dir)
