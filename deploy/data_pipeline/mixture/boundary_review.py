"""Surfaces the 200 documents nearest the calibrated classifier threshold for
hand-scoring (Requirement 4.6) -- confirms the filter isn't systematically
dropping poetry, dialogue, or code-mixed text."""

import glob
import pyarrow.parquet as pq


def nearest_to_threshold(scored_dir, threshold, n=200):
    candidates = []
    for shard_path in sorted(glob.glob(f"{scored_dir}/*.parquet")):
        table = pq.read_table(shard_path, columns=["doc_id", "text", "quality_score"])
        for row in table.to_pylist():
            if row["quality_score"] is None:
                continue
            candidates.append((abs(row["quality_score"] - threshold), row))
    candidates.sort(key=lambda x: x[0])
    return [row for _, row in candidates[:n]]
