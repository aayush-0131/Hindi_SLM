"""
Acquire IndicCorp v2 Hindi from the OFFICIAL ai4bharat/IndicCorpV2 repo --
NOT the satpalsr/indicCorpv2 mirror. The mirror's custom loading script has the
HF dataset-viewer/datasets-server disabled; the official repo's per-language
parquet split is directly queryable. Confirmed via research.md: hin_Deva split
= 13,228,283 rows / 5.02GB (~620M estimated native tokens).

Capped at <=2% of the stable mixture by mixture_planner.py -- this adapter just
acquires what's available; it does not enforce the cap itself.
"""

from datasets import load_dataset
from data_pipeline.sources.common import normalize_records, write_corpus_docs


def iter_indiccorpv2(max_docs=None):
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        yield {"doc_id": f"indiccorpv2-{i}", "text": row["text"]}


def acquire(out_dir, max_docs=None):
    records = normalize_records(iter_indiccorpv2(max_docs=max_docs), source="indiccorpv2")
    return write_corpus_docs(records, out_dir)
