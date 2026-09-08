"""
Applies HuggingFaceFW/finepdfs_edu_classifier_hin_Deva to FineWeb-2 CorpusDocs
ONLY (Requirement 4.1). Every other source arrives pre-filtered and is never
passed through this module (Requirement 4.2).

Batches multiple documents per forward pass (score_documents) -- a real
throughput requirement at FineWeb-2 scale, not just an optimization, since
unbatched per-document inference is dominated by Python/kernel-launch
overhead rather than actual GPU compute.

NOTE: each document is truncated to <=2048 tokens rather than chunked into
multiple windows with a max-across-chunks score -- a deliberate simplification
of the "chunk and take the max" contract described in design.md, accepted for
implementation speed. Revisit if long-document quality signal turns out to
matter in the boundary-hand-scoring pass (Requirement 4.6).
"""

import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "HuggingFaceFW/finepdfs_edu_classifier_hin_Deva"
MAX_CHUNK_TOKENS = 2048
DEFAULT_BATCH_SIZE = 32

_model = None
_tokenizer = None


def _load_model(device=None):
    global _model, _tokenizer
    if _model is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    return _model, _tokenizer


def score_documents(texts, device=None, batch_size=DEFAULT_BATCH_SIZE, log_every_batches=50):
    """Scores a list of documents in batches -- the throughput-critical path.
    Returns a list of float scores, same order as `texts`."""
    if not texts:
        return []
    model, tokenizer = _load_model(device)
    dev = next(model.parameters()).device
    scores = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = texts[b * batch_size:(b + 1) * batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_CHUNK_TOKENS,
        ).to(dev)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        if logits.dim() == 0:
            scores.append(float(logits.item()))
        else:
            scores.extend(float(x) for x in logits.tolist())
        if (b + 1) % log_every_batches == 0 or (b + 1) == n_batches:
            print(f"  scored {len(scores)}/{len(texts)} documents ({batch_size}/batch)...")
    return scores


def score_document(text, device=None):
    """Single-document convenience wrapper -- prefer score_documents() for throughput."""
    return score_documents([text], device=device)[0]


def score_shards(in_dir, out_dir, threshold=None, batch_size=DEFAULT_BATCH_SIZE):
    """Reads CorpusDoc parquet shards from in_dir, scores every row (batched),
    writes the same rows + populated quality_score (+ keep, if threshold is
    given) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for shard_path in sorted(glob.glob(os.path.join(in_dir, "*.parquet"))):
        table = pq.read_table(shard_path)
        rows = table.to_pylist()
        texts = [r["text"] for r in rows]
        scores = score_documents(texts, batch_size=batch_size)
        for row, score in zip(rows, scores):
            row["quality_score"] = score
            if threshold is not None:
                row["keep"] = score >= threshold
        total += len(rows)
        out_table = pa.Table.from_pylist(rows)
        pq.write_table(out_table, os.path.join(out_dir, os.path.basename(shard_path)))
    print(f"Scored {total} documents total -> {out_dir}")


def calibrate_threshold(in_dir, sample_size=50_000, target_keep_rate=(0.20, 0.25), batch_size=DEFAULT_BATCH_SIZE):
    """Scores up to `sample_size` documents (batched) and returns the score
    threshold that lands the keep rate inside target_keep_rate (Requirement
    4.1 / 4.5)."""
    texts = []
    for shard_path in sorted(glob.glob(os.path.join(in_dir, "*.parquet"))):
        table = pq.read_table(shard_path, columns=["text"])
        for row in table.to_pylist():
            texts.append(row["text"])
            if len(texts) >= sample_size:
                break
        if len(texts) >= sample_size:
            break

    scores = score_documents(texts, batch_size=batch_size)
    scores.sort(reverse=True)
    lo, hi = target_keep_rate
    mid_rate = (lo + hi) / 2
    idx = min(int(len(scores) * mid_rate), max(len(scores) - 1, 0))
    threshold = scores[idx] if scores else 0.0
    print(f"Calibrated threshold={threshold:.4f} on {len(scores)} docs "
          f"(targeting {mid_rate:.0%} keep rate)")
    return threshold, scores

THRESHOLD_FILENAME = "calibration.json"


def save_threshold(out_dir, threshold, n_scored, target_keep_rate):
    """Persists the calibrated threshold next to the classified output.

    design.md's Batch/Job Contract for this component already specifies this
    ("calibrate_threshold's result is written to a small JSON artifact
    alongside the output so apply_filter can be rerun deterministically
    against the same threshold without recalibrating") but it was never
    implemented -- so every rerun recalibrated, scoring the calibration sample
    on the GPU a second time on top of the full scoring pass. On a 50K-document
    FineWeb-2 pool that is the classifier running twice over the same corpus,
    and the GPU it consumes is GPU the pretraining gates need.
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, THRESHOLD_FILENAME)
    with open(path, "w") as f:
        json.dump({
            "threshold": float(threshold),
            "n_scored": int(n_scored),
            "target_keep_rate": list(target_keep_rate),
        }, f, indent=2)
    print(f"Saved calibration to {path} (threshold={threshold:.4f})")
    return path


def load_threshold(out_dir):
    """Returns the persisted threshold, or None if there isn't one."""
    import json
    path = os.path.join(out_dir, THRESHOLD_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"Reusing calibrated threshold={data['threshold']:.4f} from {path} "
          f"(originally calibrated on {data['n_scored']:,} docs)")
    return data["threshold"]
