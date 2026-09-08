"""Acquire FinePDFs hin_Deva (train split), carrying fw_edu_v2_scores through
as quality_score -- this source arrives pre-filtered (Requirement 4.2)."""

from datasets import load_dataset
from data_pipeline.sources.common import normalize_records, write_corpus_docs


def iter_finepdfs(max_docs=None):
    ds = load_dataset("HuggingFaceFW/finepdfs", "hin_Deva", split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        scores = row.get("fw_edu_v2_scores")
        yield {
            "doc_id": row.get("id", str(i)),
            "text": row["text"],
            "quality_score": max(scores) if scores else None,
        }


def acquire(out_dir, max_docs=None):
    records = normalize_records(iter_finepdfs(max_docs=max_docs), source="finepdfs")
    return write_corpus_docs(records, out_dir)
