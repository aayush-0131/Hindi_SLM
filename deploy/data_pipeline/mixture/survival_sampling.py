"""Measures dedup survival rate on a cross-corpus sample and classifier keep
rate on a calibration sample (Requirement 4.5) -- both already produced as a
side effect of running dedup_sources.py / calibrate_threshold() above on a
bounded sample. This module exists as the explicit, named home for that
measurement step so mixture_planner.py has a single place to read it from.

It also owns the tokens-per-document measurement, because Requirements 3.6 and
3.8 both specify mixture shares BY TOKEN, while every count the pipeline
produces upstream is a DOCUMENT count. Those units are not interchangeable
here: tokens/doc varies several-fold across these sources (OCR'd PDF documents
versus speech transcripts), so a mixture planned on document counts realizes
materially different token shares than the ones the requirements specify --
and in particular under-enforces the PDF ceiling, since PDF documents are the
long ones.
"""

import glob
import os
import pyarrow.parquet as pq

from data_pipeline.shard.shard_writer import is_survivor

DEFAULT_SAMPLE_SIZE = 500


def measure_tokens_per_doc(parquet_dir, tokenizer, sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Mean tokens per SURVIVOR document in a deduped/filtered source directory.

    Only survivors are sampled, because only survivors get written to the final
    shards -- averaging over rejected documents too would bias the estimate
    (the classifier rejects short, low-quality documents disproportionately).

    Returns None if the directory holds no survivors, so the caller can
    distinguish "empty source" from "0 tokens per document".
    """
    texts = []
    for shard_path in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        table = pq.read_table(shard_path)
        for row in table.to_pylist():
            if not is_survivor(row):
                continue
            texts.append(row["text"] or "")
            if len(texts) >= sample_size:
                break
        if len(texts) >= sample_size:
            break
    if not texts:
        return None
    # One batched call, not sample_size individual ones: passing a list routes
    # to tiktoken's parallel encode_ordinary_batch(num_threads=8) rather than
    # the single-threaded per-string path.
    return sum(len(ids) for ids in tokenizer.encode(texts)) / len(texts)


def measure_available_tokens(deduped_dirs, survivor_doc_counts, tokenizer,
                             sample_size=DEFAULT_SAMPLE_SIZE):
    """
    Converts per-source survivor DOCUMENT counts into per-source available
    TOKEN counts, which is the unit mixture_planner.plan_stable_mixture()
    actually needs.

    deduped_dirs:        {source: deduped parquet dir}
    survivor_doc_counts: {source: survivor document count}
    Returns (available_tokens, tokens_per_doc) -- both keyed by source, with
    tokens_per_doc retained so the caller can convert the resulting token plan
    back into the document limits the shard writer consumes.
    """
    available_tokens = {}
    tokens_per_doc = {}
    for source, docs in survivor_doc_counts.items():
        if docs == 0:
            available_tokens[source] = 0
            tokens_per_doc[source] = None
            continue
        mean = measure_tokens_per_doc(deduped_dirs[source], tokenizer, sample_size)
        tokens_per_doc[source] = mean
        available_tokens[source] = int(docs * mean) if mean else 0
    return available_tokens, tokens_per_doc


def summarize_survival(survival_rates: dict, classifier_threshold: float, classifier_keep_rate: float):
    return {
        "survival_rates": survival_rates,
        "classifier_threshold": classifier_threshold,
        "classifier_keep_rate": classifier_keep_rate,
    }
