#!/usr/bin/env python3
"""
Hindi LM Data Pipeline -- orchestration script for the SSH-accessible GPU box.

Implements the Data Pipeline component from .sdd/specs/hindi-language-model/design.md
(Requirements 3 & 4): acquire 7 public Hindi sources, filter FineWeb-2 for quality,
deduplicate every source against a FineWeb-2 reference set, plan the stable-phase
mixture, and write final nanochat-compatible parquet shards.

Run under tmux (this can take a long time at full scale):

    tmux new -s data-pipeline
    python3 run_pipeline.py --sample          # quick end-to-end sanity check
    python3 run_pipeline.py                   # full-scale run (after reviewing sample output)

Before running for real:
1. Run with --inspect-sangraha-only first and confirm the printed `type` values
   match EXPECTED_TYPE_VALUES in data_pipeline/sources/sangraha.py.
2. Review research.md's dedup timing risk -- measure the 1% sample throughput
   before committing to the full FineWeb-2-scale dedup pass.
3. The token-count verification step needs the Gate 1 tokenizer artifact
   (tokenizer.pkl + token_bytes.pt) at --tokenizer-dir -- run that component
   first, or pass --skip-token-verify.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.sources import fineweb2, sangraha, finepdfs, indiccorpv2
from data_pipeline.classify import finepdfs_edu_classifier as classifier
from data_pipeline.dedup import reference_index, dedup_sources
from data_pipeline.mixture import survival_sampling, boundary_review, mixture_planner
from data_pipeline.shard import shard_writer, token_count_verify


import contextlib
import time

_STAGE_TIMINGS = []


@contextlib.contextmanager
def stage(name):
    """Times a pipeline stage and prints it, so the next run reports where the
    wall-clock actually went instead of leaving it to be inferred. The first
    full-scale run took 53m32s with no per-stage breakdown, which made the
    bottleneck a guess (real 53m32 vs user 82m25 implied only ~1.5x CPU
    parallelism, but not which stage was serial)."""
    print(f"\n>>> [stage] {name} ...", flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        _STAGE_TIMINGS.append((name, dt))
        print(f">>> [stage] {name} finished in {dt / 60:.1f} min", flush=True)


def print_stage_summary():
    if not _STAGE_TIMINGS:
        return
    total = sum(dt for _, dt in _STAGE_TIMINGS)
    print("\n" + "=" * 60)
    print("Stage timings (longest first)")
    print("=" * 60)
    for name, dt in sorted(_STAGE_TIMINGS, key=lambda x: -x[1]):
        pct = 100 * dt / total if total else 0
        print(f"  {dt / 60:7.1f} min  {pct:5.1f}%  {name}")
    print(f"  {total / 60:7.1f} min  100.0%  TOTAL")


def _load_tokenizer(tokenizer_dir):
    """Loads the Gate 1 tokenizer artifact read-only for tokens/doc measurement.

    Returns None (rather than raising) if it isn't available, so a pipeline run
    without the artifact still completes -- with a loud warning and a
    document-unit mixture plan -- instead of failing at the last stage.
    """
    if not tokenizer_dir or not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
        return None
    try:
        from nanochat.tokenizer import RustBPETokenizer
    except ImportError:
        print("[run_pipeline] nanochat not importable; cannot measure tokens/doc.")
        return None
    return RustBPETokenizer.from_directory(tokenizer_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("HINDI_LM_DATA_DIR", os.path.expanduser("~/hindi_lm_data_pipeline")),
        help="Root directory for all pipeline artifacts (persistent local disk on the leased server).",
    )
    parser.add_argument("--sample", action="store_true", help="Shorthand for --sample-docs-per-source 2000 if that's not otherwise set.")
    parser.add_argument("--sample-docs-per-source", type=int, default=None,
                         help="Cap applied to all non-fineweb2 sources. Takes effect whenever set, "
                              "regardless of --sample -- does NOT require --sample to be passed too.")
    parser.add_argument(
        "--fineweb2-max-docs", type=int, default=None,
        help="Independent cap on FineWeb-2 acquisition, since it's the only source that needs "
             "the slow classifier pass. Overrides --sample-docs-per-source for fineweb2 "
             "specifically -- lets other sources run uncapped/full-scale (cheap, dedup-only) "
             "while bounding the one genuinely slow source to fit available time.",
    )
    parser.add_argument("--total-stable-tokens", type=int, default=11_750_000_000)  # reduced from 17B: see research.md's windowed-GPU-access recompute
    parser.add_argument(
        "--tokenizer-dir", default=None,
        help="Directory containing the Gate 1 tokenizer artifact (tokenizer.pkl + token_bytes.pt).",
    )
    parser.add_argument("--skip-token-verify", action="store_true")
    parser.add_argument(
        "--reuse-fineweb2", action="store_true",
        help="Skip FineWeb-2 acquisition AND classification if a classified "
             "output with a persisted calibration.json already exists, reusing "
             "both. FineWeb-2 is the only GPU-bound stage in this pipeline, and "
             "re-running it competes directly with the pretraining gates for "
             "the GPU. Use when growing the OTHER six sources' caps -- "
             "FineWeb-2 cannot carry volume at any affordable classifier "
             "budget (a full 22M-document pass is 157-393 GPU-hours).")
    parser.add_argument(
        "--dedup-fineweb2-only", action="store_true",
        help="Run MinHash dedup on FineWeb-2 only; pass the other six sources "
             "through as non-duplicates. The first full-scale run measured "
             "99.80-100%% survival on all six (four returned exactly zero "
             "duplicates) while MinHash dominated the CPU wall-clock. NOTE: "
             "those rates came from a 50K-document reference index (0.2%% of "
             "FineWeb-2), so the true duplicate rate is higher than measured -- "
             "this is a deliberate time-vs-completeness trade, not a finding "
             "that dedup is unnecessary.")
    parser.add_argument(
        "--inspect-sangraha-only", action="store_true",
        help="Just print Sangraha's `type` field values and exit -- run this first.",
    )
    args = parser.parse_args()

    if args.inspect_sangraha_only:
        sangraha.inspect_sangraha_types()
        return

    if args.sample_docs_per_source is not None:
        docs_limit = args.sample_docs_per_source
    elif args.sample:
        docs_limit = 2000
    else:
        docs_limit = None
    fineweb2_docs_limit = args.fineweb2_max_docs if args.fineweb2_max_docs is not None else docs_limit

    artifacts_dir = os.path.join(args.base_dir, "data_pipeline_artifacts")
    for sub in [
        "acquired/fineweb2", "acquired/sangraha_verified_web", "acquired/sangraha_verified_pdf",
        "acquired/sangraha_verified_speech", "acquired/sangraha_unverified", "acquired/finepdfs",
        "acquired/indiccorpv2",
        "classified/fineweb2",
        "deduped/fineweb2", "deduped/sangraha_verified_web", "deduped/sangraha_verified_pdf",
        "deduped/sangraha_verified_speech", "deduped/sangraha_unverified", "deduped/finepdfs",
        "deduped/indiccorpv2",
        "reference_index",
        "final/stable", "final/decay_a", "final/decay_b",
    ]:
        os.makedirs(os.path.join(artifacts_dir, sub), exist_ok=True)
    print(f"Artifacts dir: {artifacts_dir}")

    # Step 1: validate Sangraha's `type` field before trusting the web/pdf/speech split.
    sangraha.inspect_sangraha_types()

    # Step 2: acquisition
    dirs = {
        "fineweb2": os.path.join(artifacts_dir, "acquired/fineweb2"),
        "sangraha_verified_web": os.path.join(artifacts_dir, "acquired/sangraha_verified_web"),
        "sangraha_verified_pdf": os.path.join(artifacts_dir, "acquired/sangraha_verified_pdf"),
        "sangraha_verified_speech": os.path.join(artifacts_dir, "acquired/sangraha_verified_speech"),
        "sangraha_unverified": os.path.join(artifacts_dir, "acquired/sangraha_unverified"),
        "finepdfs": os.path.join(artifacts_dir, "acquired/finepdfs"),
        "indiccorpv2": os.path.join(artifacts_dir, "acquired/indiccorpv2"),
    }
    # Timed per source, not as one block: the sources have very different
    # cost profiles (Sangraha scans one mixed stream and discards non-matching
    # rows, FineWeb-2 is the largest, IndicCorp v2 is a plain parquet read),
    # and knowing which one dominates is what tells us where to raise caps.
    classified_dir = os.path.join(artifacts_dir, "classified/fineweb2")
    reusing_fineweb2 = (
        args.reuse_fineweb2
        and glob.glob(os.path.join(classified_dir, "*.parquet"))
        and classifier.load_threshold(classified_dir) is not None
    )
    if reusing_fineweb2:
        print("\n>>> [stage] acquire/fineweb2 + classify SKIPPED (--reuse-fineweb2): "
              "reusing the existing classified output and its persisted threshold, "
              "leaving the GPU free for the pretraining gates.")
    else:
        with stage("acquire/fineweb2"):
            print("fineweb2:", fineweb2.acquire(dirs["fineweb2"], max_docs=fineweb2_docs_limit))
    # Single pass over verified/hin buckets web/pdf/speech simultaneously,
    # instead of scanning the same stream 3 separate times (a real
    # inefficiency found on a real run -- see research.md).
    with stage("acquire/sangraha_verified (single pass, 3 buckets)"):
        verified_counts = sangraha.acquire_all_verified(
            {
                "web": dirs["sangraha_verified_web"],
                "pdf": dirs["sangraha_verified_pdf"],
                "speech": dirs["sangraha_verified_speech"],
            },
            max_docs=docs_limit,
        )
        print("sangraha_verified_{web,pdf,speech}:", verified_counts)
    with stage("acquire/sangraha_unverified"):
        print("sangraha_unverified:", sangraha.acquire_unverified(dirs["sangraha_unverified"], max_docs=docs_limit))
    with stage("acquire/finepdfs"):
        print("finepdfs:", finepdfs.acquire(dirs["finepdfs"], max_docs=docs_limit))
    with stage("acquire/indiccorpv2"):
        print("indiccorpv2:", indiccorpv2.acquire(dirs["indiccorpv2"], max_docs=docs_limit))

    # Step 3: classify FineWeb-2 only
    calib_sample_size = min(fineweb2_docs_limit, 50_000) if fineweb2_docs_limit else 50_000
    if reusing_fineweb2:
        threshold = classifier.load_threshold(classified_dir)
    else:
        with stage("classify/calibrate_threshold (GPU)"):
            threshold, _scores = classifier.calibrate_threshold(
                dirs["fineweb2"], sample_size=calib_sample_size)
            # Persist it so the next run can reuse it instead of scoring the
            # calibration sample on the GPU all over again.
            classifier.save_threshold(classified_dir, threshold, len(_scores),
                                      (0.20, 0.25))
        with stage("classify/score_shards (GPU)"):
            classifier.score_shards(dirs["fineweb2"], classified_dir, threshold=threshold)

    # Step 4: dedup -- build the FineWeb-2 reference index (from the full raw
    # pool, for maximum reference coverage), then dedup every source against
    # it. FineWeb-2's own entry is deduped from classified_dir, not the raw
    # acquired dir, so the classifier's `keep` column survives into the
    # deduped output alongside the new `duplicate` column -- otherwise
    # shard_writer's quality filter would silently never fire for FineWeb-2.
    index_path = os.path.join(artifacts_dir, "reference_index/fineweb2_reference.pkl")
    with stage("dedup/build_reference_index"):
        lsh = reference_index.build_reference_index(dirs["fineweb2"], index_path)
    num_perm = reference_index.DEFAULT_NUM_PERM

    dedup_inputs = dict(dirs)
    dedup_inputs["fineweb2"] = classified_dir

    survival_rates = {}
    for name, src_dir in dedup_inputs.items():
        out_dir = os.path.join(artifacts_dir, f"deduped/{name}")
        if args.dedup_fineweb2_only and name != "fineweb2":
            with stage(f"dedup/{name} (SKIPPED)"):
                survival_rates[name] = dedup_sources.passthrough_no_dedup(
                    src_dir, out_dir, reason="--dedup-fineweb2-only")
        else:
            with stage(f"dedup/{name}"):
                survival_rates[name] = dedup_sources.dedup_against_reference(
                    src_dir, out_dir, lsh, num_perm)
    print("survival_rates:", survival_rates)

    # Step 5: boundary hand-scoring sample (Requirement 4.6) -- inspect this list yourself
    with stage("mixture/boundary_review"):
        boundary_docs = boundary_review.nearest_to_threshold(classified_dir, threshold, n=min(200, fineweb2_docs_limit or 200))
    print(f"{len(boundary_docs)} boundary documents surfaced for hand-scoring")

    # Step 6: mixture planning.
    #
    # Two different units are in play and conflating them is a real bug that
    # shipped once already: count_survivors() returns DOCUMENT counts, while
    # Requirements 3.6/3.8 specify mixture shares BY TOKEN and
    # plan_stable_mixture() expects tokens. Because tokens/doc varies
    # several-fold across these sources (OCR'd PDFs are long, speech
    # transcripts are short), planning on document counts realizes materially
    # different token shares than the requirements specify -- and specifically
    # under-enforces the PDF ceiling, since PDF documents are the long ones.
    #
    # So: measure survivor documents, convert to tokens with a per-source
    # measured tokens/doc, plan in TOKENS, then convert the plan back to
    # DOCUMENT limits for the shard writer.
    deduped_dirs = {name: os.path.join(artifacts_dir, f"deduped/{name}") for name in dirs}
    with stage("mixture/count_survivors"):
        survivor_doc_counts = {
            name: shard_writer.count_survivors(d) for name, d in deduped_dirs.items()
        }
    print("survivor doc counts:", survivor_doc_counts)

    tokenizer = _load_tokenizer(args.tokenizer_dir)
    if tokenizer is None:
        print(
            "\n[run_pipeline] WARNING: no usable tokenizer at --tokenizer-dir, so "
            "tokens/doc cannot be measured. Falling back to planning the mixture in "
            "DOCUMENT counts. The resulting token shares will NOT match Requirement "
            "3.6, and the Requirement 3.8 PDF ceiling will be enforced in the wrong "
            "unit (PDF documents are longer than average, so its real token share "
            "will exceed 25%). Pass --tokenizer-dir with the Gate 1 artifact to get "
            "a token-correct mixture.\n"
        )
        measured_available = survivor_doc_counts
        tokens_per_doc = {}
    else:
        with stage("mixture/measure_tokens_per_doc"):
            measured_available, tokens_per_doc = survival_sampling.measure_available_tokens(
                deduped_dirs, survivor_doc_counts, tokenizer
            )
        print("measured tokens/doc:", {
            k: (round(v, 1) if v else v) for k, v in tokens_per_doc.items()
        })
        print("measured available TOKENS:", measured_available)
        print(f"total available: {sum(measured_available.values()):,} tokens "
              f"across {sum(survivor_doc_counts.values()):,} documents")

    plan = mixture_planner.plan_stable_mixture(measured_available, args.total_stable_tokens)
    print("stable mixture plan (tokens):", plan)
    planned_total = sum(plan.values())
    if planned_total:
        print("stable mixture plan (realized token share):", {
            k: f"{v / planned_total:.1%}" for k, v in plan.items()
        })
        pdf_share = sum(plan.get(s, 0) for s in mixture_planner.PDF_SOURCES) / planned_total
        print(f"PDF-derived token share: {pdf_share:.1%} (Requirement 3.8 ceiling: 25%)")

    # Step 7: write final shards. write_phase_shards() caps per source by
    # DOCUMENT count, so the token plan has to be converted back first.
    stable_out_dir = os.path.join(artifacts_dir, "final/stable")
    if tokenizer is None:
        doc_limits = plan  # already document counts in the fallback path
    else:
        doc_limits = shard_writer.doc_limits_from_token_plan(
            plan, tokens_per_doc, survivor_doc_counts
        )
        print("stable mixture plan (doc limits for shard writer):", doc_limits)
    source_dirs_and_limits = [
        (deduped_dirs[name], limit) for name, limit in doc_limits.items()
    ]
    with stage("shard/write_phase_shards"):
        shard_writer.write_phase_shards(source_dirs_and_limits, stable_out_dir)

    # Step 8: token-count verification against the Gate 1 tokenizer artifact
    if args.skip_token_verify:
        print("Skipping token-count verification (--skip-token-verify).")
    elif args.tokenizer_dir and os.path.exists(os.path.join(args.tokenizer_dir, "tokenizer.pkl")):
        with stage("shard/token_count_verify"):
            token_count_verify.count_unique_tokens(stable_out_dir, args.tokenizer_dir)
    else:
        print(
            "Skipping token-count verification: no --tokenizer-dir with a tokenizer.pkl given. "
            "Run the Gate 1 tokenizer component first, then re-run with --tokenizer-dir."
        )

    print_stage_summary()
    print("\nData pipeline run complete.")


if __name__ == "__main__":
    main()
