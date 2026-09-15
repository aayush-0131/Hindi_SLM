#!/usr/bin/env python3
"""Audit History-HI SFT text against frozen HistoryBench without exposing benchmark text.

Exact lexical overlap is a contamination *screen*, not proof of semantic independence.
The script is read-only with respect to source data and refuses to overwrite its report
unless --force is explicitly supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

DEFAULT_BENCH_SHA = "21dc840c8d4fc07288ddbeaf61c4b8cf41e930b0504131e3252ab45d4e224fbe"
DEFAULT_SFT_SHA = "9d78b9a9fce9c3a8d7343195ea1c49a848b1c63214c445fb8cfdadef09587f0f"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    return " ".join(re.findall(r"[\w\u0900-\u097F]+", text.lower(), flags=re.UNICODE))


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = norm(text).split()
    return {tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))}


def flatten_json(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_json(v)


def load_sft(path: Path) -> list[str]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {e}")
        rows.append(" ".join(flatten_json(obj)))
    return rows


def load_benchmark(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [" ".join(str(v) for v in row.values() if v is not None) for row in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True, help="Frozen all-100 HistoryBench CSV")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=Path("results/contamination/historybench_all100_audit.json"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--expected-benchmark-sha256", default=DEFAULT_BENCH_SHA)
    ap.add_argument("--expected-sft-package-sha256", default=DEFAULT_SFT_SHA,
                    help="Provenance only: canonical seed-package hash; source JSONL hashes are recorded separately")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.n < 2:
        raise SystemExit("--n must be >=2")
    if args.report.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.report}; pass --force only intentionally")
    for p in (args.benchmark, args.train, args.val):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")

    bench_sha = sha256(args.benchmark)
    if bench_sha != args.expected_benchmark_sha256:
        raise SystemExit(f"Frozen benchmark hash mismatch: got {bench_sha}")

    bench = load_benchmark(args.benchmark)
    if len(bench) != 100:
        raise SystemExit(f"Expected frozen all-100 benchmark, found {len(bench)} rows")
    bench_sets = [ngrams(x, args.n) for x in bench]

    split_results = {}
    total_overlap_rows = 0
    for split, path in (("train", args.train), ("val", args.val)):
        rows = load_sft(path)
        hits = []
        for i, text in enumerate(rows):
            s = ngrams(text, args.n)
            matched = [j + 1 for j, b in enumerate(bench_sets) if s & b]
            if matched:
                # Record only indices/counts, never benchmark or SFT text.
                hits.append({"sft_row": i + 1, "benchmark_rows": matched})
        total_overlap_rows += len(hits)
        split_results[split] = {
            "path": str(path), "sha256": sha256(path), "rows": len(rows),
            "rows_with_overlap": len(hits), "matches": hits,
        }

    report = {
        "schema_version": "1.0",
        "audit": "exact_token_ngram_overlap",
        "ngram_n": args.n,
        "benchmark": {"path": str(args.benchmark), "sha256": bench_sha, "rows": len(bench)},
        "sft_seed_package_canonical_sha256": args.expected_sft_package_sha256,
        "splits": split_results,
        "total_sft_rows_with_overlap": total_overlap_rows,
        "status": "PASS_NO_EXACT_NGRAM_OVERLAP" if total_overlap_rows == 0 else "REVIEW_EXACT_NGRAM_OVERLAP",
        "claim_boundary": "Exact n-gram screening does not establish absence of semantic contamination.",
        "privacy": "Report contains row indices/counts only; benchmark text is never emitted.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_suffix(args.report.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.report)
    print(json.dumps({"report": str(args.report), "status": report["status"],
                      "total_sft_rows_with_overlap": total_overlap_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
