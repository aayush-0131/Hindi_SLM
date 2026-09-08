"""Acquire FineWeb-2 hin_Deva -- TRAIN SPLIT ONLY. The official `test` split is
never referenced here or anywhere else in this pipeline (Requirement 4.4)."""

from datasets import load_dataset
from data_pipeline.sources.common import normalize_records, write_corpus_docs


def iter_fineweb2(streaming=True, max_docs=None):
    ds = load_dataset("HuggingFaceFW/fineweb-2", "hin_Deva", split="train", streaming=streaming)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        yield {"doc_id": row.get("id", str(i)), "text": row["text"]}


def acquire(out_dir, max_docs=None):
    records = normalize_records(iter_fineweb2(max_docs=max_docs), source="fineweb2")
    return write_corpus_docs(records, out_dir)
