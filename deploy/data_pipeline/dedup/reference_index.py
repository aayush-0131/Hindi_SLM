"""
Builds a MinHash reference index over FineWeb-2 (train split only) using
datasketch.MinHashLSH. threshold=0.70 is the Jaccard similarity threshold
directly (Requirement 4.3 -- matches Sangraha's own dedup parameters).
"""

import glob
import pickle
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.70


def shingles(text, n=5):
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def make_minhash(text, num_perm=DEFAULT_NUM_PERM, n_grams=5):
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, n=n_grams):
        m.update(sh.encode("utf8"))
    return m


def build_reference_index(fineweb2_dir, index_out_path, num_perm=DEFAULT_NUM_PERM,
                           threshold=DEFAULT_THRESHOLD, n_grams=5, max_docs=None):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    count = 0
    for shard_path in sorted(glob.glob(f"{fineweb2_dir}/*.parquet")):
        table = pq.read_table(shard_path, columns=["doc_id", "text"])
        for row in table.to_pylist():
            mh = make_minhash(row["text"], num_perm=num_perm, n_grams=n_grams)
            lsh.insert(str(row["doc_id"]), mh)
            count += 1
            if max_docs is not None and count >= max_docs:
                break
        if max_docs is not None and count >= max_docs:
            break
    with open(index_out_path, "wb") as f:
        pickle.dump({"lsh": lsh, "num_perm": num_perm, "threshold": threshold, "n_grams": n_grams}, f)
    print(f"Reference index built over {count} FineWeb-2 documents -> {index_out_path}")
    return lsh
