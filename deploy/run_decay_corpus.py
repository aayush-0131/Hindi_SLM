#!/usr/bin/env python3
"""Build the Decay A / Decay B corpora (Requirement 3.7). They do not exist.

Round 1 built only `final/stable/`. Both decay variants branch from the
step-27,000 checkpoint and need their own mixture, so this runs during the
GPU-off upload window -- it is CPU/network work and must not compete with
training.

PLAN C ORDER OF OPERATIONS (whole upload window)
------------------------------------------------
    # A. grow the STABLE corpus first -- it gates resuming at all.
    #    Round 1's corpus is 3.13B tokens = 4.09 epochs by step 24,400, i.e.
    #    repetition-limited. Plan C extends stable to 27,000 steps, which is
    #    only worth doing on fresh data. FineWeb-2 alone has ~12.7B unused.
    python3 add_fineweb2.py --out-dir <final/stable> --max-docs 8000000 \
        --start-shard-idx <n>
    python3 run_interleave.py --corpus-dir <final/stable>   # MANDATORY after

    # B. ONE long Sangraha pass (~4.4h) -- grows BOTH pdf and speech, which
    #    Round 1 left at ~300M and ~30M tokens against 2,436M and 869M
    #    available. Start early; it is the long pole.
    python3 run_decay_corpus.py grow-sangraha

    # C. acquire the sources Round 1 never touched, plus the fw2 top slice
    python3 run_decay_corpus.py acquire --variant a
    python3 run_decay_corpus.py acquire --variant b

    # D. measure real token supply per component (uses the Gate 1 tokenizer)
    python3 run_decay_corpus.py measure --tokenizer-dir <gate1 tokenizer>

    # E. plan, review the deviation list, then write shards
    python3 run_decay_corpus.py build --variant a
    python3 run_decay_corpus.py build --variant b

Step A's `run_interleave.py` is not optional. `add_fineweb2.py` writes one
contiguous block of shards, and nanochat streams shards in sorted filename
order -- so skipping the re-interleave recreates the source-ordering problem
that cost Round 1 a +42% train-loss spike once per epoch.

WHAT WILL NOT MATCH REQUIREMENT 3.7, AND WHY
--------------------------------------------
Only **Hindi Wikipedia** is a genuine supply ceiling: ~85M tokens total
(169k articles), against a 24% share that would need ~566M at the 2.359B decay
budget. It reaches ~9% at the default 2x upsampling cap.

Sangraha PDF and speech are NOT ceilings -- they are scan-wall-clock bound, and
step B above is what lifts them (2,436M and 869M exist; Round 1 took ~300M and
~30M). An earlier version of this docstring wrongly listed speech as capped at
~30M.

`decay_planner` water-fills whatever still cannot be met onto the components
with spare supply and records every deviation. Read the deviation list; do not
ignore it.

Three sub-slices Requirement 3.7 names are also not identifiable in the
released data (Sangraha ships no sub-source column, and FineWeb-2 quality
scores were never computed at scale). Substitutions are recorded per component.
"""

import argparse
import json
import os
import sys

from data_pipeline.mixture.decay_planner import (
    COMPONENT_SOURCES, DEFAULT_MAX_UPSAMPLE, decay_targets, format_plan,
    plan_decay_mixture,
)

DEFAULT_ARTIFACTS = "data_pipeline_artifacts"
# Plan C: 4,500 steps per variant. Single-sourced from training/config.py so
# the corpus target and the training schedule cannot drift apart.
try:
    from training.config import DECAY_STEPS_PER_VARIANT, TOTAL_BATCH_TOKENS
    DEFAULT_DECAY_TOKENS = DECAY_STEPS_PER_VARIANT * TOTAL_BATCH_TOKENS
except Exception:      # keep the tool usable without the training package
    DEFAULT_DECAY_TOKENS = 4500 * 524_288


def _acquired_dir(artifacts, source):
    return os.path.join(artifacts, "acquired", source)


def cmd_grow_sangraha(args):
    """One long single-pass scan of Sangraha verified/hin for the decay phase.

    WHY ONE LONG SCAN, AND WHY IT IS WORTH HOURS
    --------------------------------------------
    Round 1 took only ~400K documents per type, which left the decay-relevant
    types badly under-supplied:

        type    Round 1 got   requirements.md sec 2 says exists
        pdf     ~300M tok     2,435.7M tok  (NCERT/state textbooks, eGyanKosh)
        speech  ~30M tok        869.0M tok  (NPTEL, OpenSubtitles, Mann Ki Baat)

    ROUND2_HANDOFF.md sec 7.14 notes speech is only ~0.18% of the stream and
    concludes a bigger scan is "pure waste". That is true for a *short* scan
    with a small cap, but it does NOT mean speech is exhausted -- 14M rows was
    a LOWER BOUND (that scan was still running when it was recorded), and the
    published measured figure is 869M tokens. So speech is bounded by scan
    wall-clock, not by supply.

    At ~0.18% density and ~1,710 tokens/document, collecting the ~191K speech
    documents that Requirement 3.7's 18% share needs means scanning ~106M rows,
    about 4.2 h at the measured ~420K rows/min. That is affordable inside the
    15-hour GPU-off window, and it is the difference between speech at 3% of
    the decay mixture and speech at its nominal ~18%.

    Critically this is ONE pass, not three: `acquire_all_verified` (the Bug 5
    fix) buckets web/pdf/speech from a single stream simultaneously, so the
    same 4.2 h that finds the rare speech documents also fills the abundant
    PDF cap many times over.
    """
    from data_pipeline.sources.sangraha import acquire_all_verified

    out_dirs = {
        "web": _acquired_dir(args.artifacts, "sangraha_verified_web"),
        "pdf": _acquired_dir(args.artifacts, "sangraha_verified_pdf"),
        "speech": _acquired_dir(args.artifacts, "sangraha_verified_speech"),
    }
    for path in out_dirs.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    caps = {"web": args.web_docs, "pdf": args.pdf_docs,
            "speech": args.speech_docs}
    print(f"single-pass Sangraha verified/hin scan")
    print(f"  caps      : {caps}")
    print(f"  max_scan  : {args.max_scan:,} rows "
          f"(~{args.max_scan / 420_000 / 60:.1f} h at the measured "
          f"~420K rows/min)")
    print(f"  NOTE: speech is ~0.18% of the stream, so the scan bound -- not "
          f"the speech cap -- is what ends this run. That is expected.\n")

    counts = acquire_all_verified(out_dirs, max_docs=caps,
                                  max_scan=args.max_scan,
                                  flush_every=args.flush_every)
    print(f"\nwritten: {counts}")
    for t, n in counts.items():
        if caps[t] and n < caps[t]:
            print(f"  {t}: got {n:,} of {caps[t]:,} requested -- raise "
                  f"--max-scan to collect more (this type is rarer in the "
                  f"stream than the cap assumes)")
    return 0


def cmd_acquire(args):
    """Acquire only the components this variant actually needs."""
    targets = decay_targets(args.variant)
    os.makedirs(os.path.join(args.artifacts, "acquired"), exist_ok=True)

    for component in targets:
        source = COMPONENT_SOURCES[component]
        out_dir = _acquired_dir(args.artifacts, source)
        if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
            print(f"[{component}] {out_dir} already populated, skipping "
                  f"(--force to redo)")
            continue

        if source in ("sangraha_verified_pdf", "sangraha_verified_speech"):
            # Handled once, by the single-pass grow step below -- not per
            # component, because both types come out of the SAME stream.
            continue

        if source == "wikipedia_hi":
            from data_pipeline.sources import wikipedia_hi
            print(f"[{component}] acquiring Hindi Wikipedia -> {out_dir}")
            wikipedia_hi.acquire(out_dir, max_docs=args.max_docs)

        elif source == "fineweb2_top":
            from data_pipeline.sources import fineweb2_top_slice as top
            print(f"[{component}] calibrating the metadata proxy threshold "
                  f"(Requirement 3.7's classifier scores do not exist)")
            threshold, stats = top.calibrate_threshold(
                sample_size=args.calibration_sample)
            print(f"  {stats['note']}")
            with open(os.path.join(args.artifacts,
                                   "fineweb2_top_threshold.json"), "w") as fh:
                json.dump(stats, fh, indent=1)
            # skip_docs avoids re-selecting the ~5.94M documents already in the
            # stable corpus; decay data overlapping stable would waste the
            # phase re-showing text the model has already seen ~4 times.
            print(f"[{component}] acquiring -> {out_dir} "
                  f"(skipping the first {args.skip_docs:,} stream docs)")
            top.acquire(out_dir, min_language_score=threshold,
                        max_docs=args.max_docs, skip_docs=args.skip_docs)

        elif source == "fineweb_edu_hindi":
            if args.variant.lower() == "a":
                raise AssertionError(
                    "fineweb_edu_hindi must not be acquired for Decay A "
                    "(Requirements 3.3, 7.3)")
            from data_pipeline.sources import fineweb_edu_hindi as edu
            print(f"[{component}] calibrating top-{args.edu_keep:.1%} score "
                  f"threshold")
            try:
                threshold, field, n = edu.calibrate_score_threshold(
                    sample_size=args.calibration_sample,
                    keep_fraction=args.edu_keep)
                print(f"  field={field!r} threshold={threshold} "
                      f"(from {n} scores)")
                edu.acquire(out_dir, max_docs=args.max_docs,
                            min_score=threshold, score_field=field)
            except KeyError as e:
                # Verified 2026-09-07 by direct schema inspection:
                # KathirKs/fineweb-edu-hindi's `meta_data` column is pure
                # CommonCrawl provenance (dump/file_path/id/url), not a
                # quality score -- under ANY of SCORE_FIELD_CANDIDATES,
                # nested or top-level. This is not a wrong field name; the
                # dataset ships no per-document score at all. A dataset
                # literally named "fineweb-edu-hindi" is presumably already
                # an edu-filtered slice upstream, so Requirement 3.7's
                # "top 0.5% by quality score" has no score left to select
                # on for this source. DEVIATION (recorded, not hidden):
                # acquire unfiltered rather than block the component.
                print(f"  DEVIATION: {e}")
                print(f"  no quality-score field exists in this dataset "
                      f"(confirmed by schema inspection: meta_data = "
                      f"dump/file_path/id/url only). Requirement 3.7's "
                      f"top-0.5%% selection is inapplicable to this source "
                      f"as shipped -- acquiring UNFILTERED instead.")
                edu.acquire(out_dir, max_docs=args.max_docs, min_score=None)
        else:
            raise ValueError(f"no acquisition path for source {source!r}")
    return 0


def cmd_measure(args):
    """Measure real tokens available per component, using the Gate 1 tokenizer.

    Planning in DOCUMENT space was Bug 9 -- tokens/doc varies several fold
    across these sources, so a document-space plan realizes materially
    different token shares than Requirement 3.7 specifies.
    """
    from data_pipeline.mixture.survival_sampling import measure_available_tokens

    all_components = set(decay_targets("a")) | set(decay_targets("b"))
    source_dirs = {}
    for component in sorted(all_components):
        source = COMPONENT_SOURCES[component]
        path = _acquired_dir(args.artifacts, source)
        if os.path.isdir(path) and os.listdir(path):
            source_dirs[component] = path
        else:
            print(f"  {component}: NOT ACQUIRED ({path})")

    if not source_dirs:
        print("nothing acquired yet -- run `acquire` first", file=sys.stderr)
        return 1

    from nanochat.tokenizer import RustBPETokenizer
    from data_pipeline.shard.shard_writer import count_survivors

    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    doc_counts = {c: count_survivors(d) for c, d in source_dirs.items()}
    for component, n in sorted(doc_counts.items()):
        print(f"  {component:34s} {n:>12,} survivor docs")

    measured, tokens_per_doc = measure_available_tokens(
        source_dirs, doc_counts, tokenizer)

    out = os.path.join(args.artifacts, "decay_available_tokens.json")
    with open(out, "w") as fh:
        json.dump(measured, fh, indent=1)
    # tokens_per_doc is retained because the shard writer consumes DOCUMENT
    # limits, and doc_limits_from_token_plan() needs this to convert the token
    # plan back without re-measuring (and without re-introducing Bug 9).
    with open(os.path.join(args.artifacts,
                           "decay_tokens_per_doc.json"), "w") as fh:
        json.dump(tokens_per_doc, fh, indent=1)

    print(f"\nmeasured token availability -> {out}")
    for component, tokens in sorted(measured.items(), key=lambda kv: -kv[1]):
        print(f"  {component:34s} {tokens:>15,} tokens "
              f"({tokens_per_doc.get(component) or 0:.0f} tok/doc)")
    return 0


def cmd_build(args):
    measured_path = os.path.join(args.artifacts, "decay_available_tokens.json")
    if not os.path.exists(measured_path):
        print(f"missing {measured_path} -- run `measure` first", file=sys.stderr)
        return 1
    measured = json.load(open(measured_path))

    targets = decay_targets(args.variant)
    available = {c: int(measured.get(c, 0)) for c in targets}

    plan = plan_decay_mixture(args.variant, available, args.decay_tokens,
                              max_upsample=args.max_upsample)
    print(format_plan(plan))

    plan_path = os.path.join(args.artifacts,
                             f"decay_{args.variant.lower()}_plan.json")
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=1)
    print(f"\nplan -> {plan_path}")

    if plan["realized_total"] < args.decay_tokens * 0.9:
        print(f"\nREFUSING TO WRITE SHARDS: the plan realizes only "
              f"{plan['realized_total'] / 1e9:.3f}B of the "
              f"{args.decay_tokens / 1e9:.3f}B target ("
              f"{plan['realized_total'] / args.decay_tokens:.0%}). Acquire more "
              f"of a component with spare supply, or raise --max-upsample "
              f"deliberately. Writing a corpus this short would run the decay "
              f"phase off the end of its data.", file=sys.stderr)
        return 1

    if args.plan_only:
        print("\n--plan-only: no shards written")
        return 0

    print("\nShard writing is not wired up in this command yet -- the mixture "
          "plan above is the artifact to review first, since its deviation "
          "list is what needs a human decision. Once the plan looks right, "
          "write shards with shard_writer.write_phase_shards() using "
          "doc_limits_from_token_plan(), then run:\n"
          f"    python3 run_interleave.py --corpus-dir "
          f"{os.path.join(args.artifacts, 'final', 'decay_' + args.variant)}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("grow-sangraha", "acquire", "measure", "build"):
        p = sub.add_parser(name)
        p.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
        if name not in ("measure", "grow-sangraha"):
            p.add_argument("--variant", required=True, choices=["a", "b"])
        if name == "grow-sangraha":
            # Sized from Requirement 3.7's shares at the 1.810B decay budget,
            # using tokens/doc measured on the real corpus (pdf 951,
            # speech 1,710). Both are set well above the strict need so a
            # single scan serves both decay variants.
            p.add_argument("--pdf-docs", type=int, default=1_500_000,
                           help="~1.43B tokens at 951 tok/doc (default 1.5M)")
            p.add_argument("--speech-docs", type=int, default=400_000,
                           help="~684M tokens at 1,710 tok/doc (default 400K); "
                                "expect the scan bound to stop this first")
            p.add_argument("--web-docs", type=int, default=0,
                           help="web is not a decay component (Req 3.7); "
                                "default 0 so the pass does not waste memory")
            p.add_argument("--flush-every", type=int, default=100_000,
                           help="write each type's buffer to durable parquet "
                                "every N documents (default 100,000). Set 0 to "
                                "use the old all-or-nothing behaviour, which "
                                "loses everything if a long scan dies")
            p.add_argument("--max-scan", type=int, default=110_000_000,
                           help="rows to scan before stopping. ~4.4 h at the "
                                "measured ~420K rows/min. THIS is what bounds "
                                "the speech yield, not --speech-docs")
        if name == "acquire":
            p.add_argument("--max-docs", type=int, default=None)
            p.add_argument("--skip-docs", type=int, default=6_000_000,
                           help="skip this many FineWeb-2 stream documents so "
                                "the decay slice does not overlap the stable "
                                "corpus (Round 1 consumed ~5.94M)")
            p.add_argument("--calibration-sample", type=int, default=200_000)
            p.add_argument("--edu-keep", type=float, default=0.005,
                           help="fineweb-edu-hindi top fraction (Req 3.7: 0.5%%)")
            p.add_argument("--force", action="store_true")
        if name == "measure":
            p.add_argument("--tokenizer-dir", required=True,
                           help="Gate 1 tokenizer directory")
        if name == "build":
            p.add_argument("--decay-tokens", type=int,
                           default=DEFAULT_DECAY_TOKENS)
            p.add_argument("--max-upsample", type=float,
                           default=DEFAULT_MAX_UPSAMPLE)
            p.add_argument("--plan-only", action="store_true")

    args = ap.parse_args()
    return {"grow-sangraha": cmd_grow_sangraha, "acquire": cmd_acquire,
            "measure": cmd_measure, "build": cmd_build}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
