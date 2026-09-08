"""Verifies both tokenization passes hand LISTS to tokenizer.encode(), not strings.

Why this matters (nanochat/tokenizer.py:96-120): RustBPETokenizer.encode()
dispatches on input type.
  - str  -> self.enc.encode_ordinary(text)                      single-threaded
  - list -> self.enc.encode_ordinary_batch(text, num_threads=8) parallel Rust
Passing documents one at a time therefore leaves 7 of 8 threads idle AND pays
Python->Rust call overhead per document. The fix is purely about how the input
is shaped, so a fake tokenizer that records call shapes is the right test.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.mixture.survival_sampling import measure_tokens_per_doc
from data_pipeline.shard import token_count_verify


class FakeTokenizer:
    """Mimics RustBPETokenizer.encode()'s type dispatch and records call shapes."""
    def __init__(self):
        self.calls = []          # "str" or "list"
        self.batch_sizes = []

    def encode(self, text, **kw):
        if isinstance(text, str):
            self.calls.append("str")
            return [0] * max(1, len(text.split()))
        elif isinstance(text, list):
            self.calls.append("list")
            self.batch_sizes.append(len(text))
            return [[0] * max(1, len(t.split())) for t in text]
        raise ValueError(type(text))


def _write(dir_path, rows, name="shard_00000.parquet"):
    os.makedirs(dir_path, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(dir_path, name))


def test_measure_tokens_per_doc_batches():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "deduped")
        rows = [{"text": f"doc number {i} with words", "source": "s",
                 "doc_id": str(i), "duplicate": False, "keep": True}
                for i in range(50)]
        _write(src, rows)
        tok = FakeTokenizer()
        mean = measure_tokens_per_doc(src, tok, sample_size=50)
        assert tok.calls == ["list"], tok.calls
        assert tok.batch_sizes == [50], tok.batch_sizes
        assert mean == 5.0, mean          # "doc number i with words" = 5 words
        print(f"  measure_tokens_per_doc: 1 batched call of 50 (was 50 calls), mean={mean}")


def test_measure_tokens_per_doc_excludes_non_survivors():
    """Only survivors reach the final shards, so only survivors may be averaged."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "deduped")
        rows = [
            {"text": "a b c d e f g h", "source": "s", "doc_id": "1",
             "duplicate": False, "keep": True},                       # survivor, 8
            {"text": "x", "source": "s", "doc_id": "2",
             "duplicate": True, "keep": True},                        # duplicate
            {"text": "y", "source": "s", "doc_id": "3",
             "duplicate": False, "keep": False},                      # rejected
            {"text": "p q r s", "source": "s", "doc_id": "4",
             "duplicate": False, "keep": True},                       # survivor, 4
        ]
        _write(src, rows)
        tok = FakeTokenizer()
        mean = measure_tokens_per_doc(src, tok, sample_size=100)
        assert tok.batch_sizes == [2], tok.batch_sizes
        assert mean == 6.0, mean          # (8 + 4) / 2
        print(f"  non-survivors excluded: batch of 2, mean={mean}")


def test_token_count_verify_batches_and_sums():
    with tempfile.TemporaryDirectory() as d:
        phase = os.path.join(d, "stable")
        # Batching is per shard: shard 0 (2,000 rows) -> [1000, 1000],
        # shard 1 (500 rows) -> [500]. Three calls, not 2,500.
        _write(phase, [{"text": "one two three"} for _ in range(2000)],
               "shard_00000.parquet")
        _write(phase, [{"text": "one two three"} for _ in range(500)],
               "shard_00001.parquet")

        tok = FakeTokenizer()
        import unittest.mock as mock
        fake_mod = mock.MagicMock()
        fake_mod.RustBPETokenizer.from_directory.return_value = tok
        with mock.patch.dict(sys.modules, {"nanochat": mock.MagicMock(),
                                           "nanochat.tokenizer": fake_mod}):
            total, passed = token_count_verify.count_unique_tokens(phase, "/unused")

        assert tok.calls == ["list"] * 3, tok.calls
        assert tok.batch_sizes == [1000, 1000, 500], tok.batch_sizes
        assert total == 2500 * 3, total
        assert passed is False, "2,500 docs must not pass the 12B floor"
        print(f"  token_count_verify: 3 batched calls (was 2,500), total={total:,}, "
              f"floor correctly not met")


def test_empty_dir_returns_none():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "empty")
        os.makedirs(src)
        assert measure_tokens_per_doc(src, FakeTokenizer()) is None
        print("  empty source -> None (not a divide-by-zero): OK")


if __name__ == "__main__":
    for fn in [test_measure_tokens_per_doc_batches,
               test_measure_tokens_per_doc_excludes_non_survivors,
               test_token_count_verify_batches_and_sums,
               test_empty_dir_returns_none]:
        fn()
    print("\nall encode-batching tests passed")
