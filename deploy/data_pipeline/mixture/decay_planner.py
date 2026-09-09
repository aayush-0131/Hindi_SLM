"""Plan the Decay A / Decay B mixtures from MEASURED availability (Req 3.7).

WHY THIS IS NOT JUST `plan_stable_mixture` WITH DIFFERENT RATIOS
---------------------------------------------------------------
The stable phase's sources were all large relative to their target shares, so
water-filling a shortfall between them was enough. Two of the four decay
components are small in ABSOLUTE terms and cannot fill their nominal shares at
any plausible decay budget:

    component                          nominal (A)   measured supply
    sangraha_verified_pdf_textbooks            41%   ~300M tok (growable)
    hindi_wikipedia                            24%   ~85M tok  (FIXED)
    sangraha_verified_speech_nptel             18%   ~30M tok  (~FIXED)
    fineweb2_top2pct                           17%   large (17M docs untouched)
    fineweb_edu_hindi  (B only)                15%   ~1.5B tok

Hindi Wikipedia has ~169k articles, full stop; Sangraha `speech` rows are only
~0.18% of the `verified/hin` stream, so a bigger scan barely helps
(ROUND2_HANDOFF.md sec 7.14). At a 1.8B-token decay budget, Requirement 3.7's
24% Wikipedia share is ~430M tokens against ~85M available -- 5x repetition --
and 18% speech is ~324M against ~30M -- 11x.

So this planner does three things the stable planner does not:

 1. Bounded upsampling. Repetition is permitted up to `max_upsample`, because
    upsampling curated data is a normal and intended part of a decay phase
    (Requirement 3.7 itself says "upsampled"). But it is BOUNDED, because the
    decay phase is what shapes the shipped model and 11x repetition of 22k
    speech documents invites memorisation rather than generalisation. Default
    2.0 -- deliberately far below the ~4-epoch ceiling used for the stable
    phase, since these sets are orders of magnitude smaller.
 2. Water-fills whatever still cannot be met onto the components that DO have
    spare supply (fineweb2_top2pct, and fineweb_edu_hindi in B).
 3. Reports realized shares and every deviation explicitly, so the shipped
    mixture is never silently different from Requirement 3.7.

SUB-SLICE LABELS DO NOT EXIST IN THE DATA
-----------------------------------------
Requirement 3.7 names "Sangraha Verified PDF -- textbook / eGyanKosh slice" and
"Sangraha Verified Speech -- NPTEL course transcripts". Sangraha's `verified`
config ships only `doc_id, type, text` (research.md) -- there is no sub-source
column, so neither slice is identifiable. The whole of each type is used
instead, and the substitution is recorded in the plan's `deviations`.

Likewise "FineWeb-2, top 2% by quality score" presupposes classifier scores
that were never computed at scale (only 5,000 documents were ever classified --
a full pass is 157-393 GPU-hours). A metadata proxy is used instead; see
`fineweb2_top_slice.py`.
"""

from data_pipeline.config.mixture_targets import (
    DECAY_A_TARGETS, DECAY_B_TARGETS, DECAY_ONLY_SOURCES,
)
from data_pipeline.mixture.mixture_planner import _water_fill

# Logical decay component -> the acquired source directory that supplies it.
# Kept explicit so a substitution is visible here rather than buried in a
# caller's string formatting.
COMPONENT_SOURCES = {
    "sangraha_verified_pdf_textbooks": "sangraha_verified_pdf",
    "hindi_wikipedia": "wikipedia_hi",
    "sangraha_verified_speech_nptel": "sangraha_verified_speech",
    "fineweb2_top2pct": "fineweb2_top",
    "fineweb_edu_hindi": "fineweb_edu_hindi",
}

# Components whose Requirement 3.7 label names a sub-slice that is not
# identifiable in the released data (see module docstring).
SUBSLICE_SUBSTITUTIONS = {
    "sangraha_verified_pdf_textbooks":
        "Requirement 3.7 asks for the textbook/eGyanKosh slice, but Sangraha's "
        "verified config ships only (doc_id, type, text) with no sub-source "
        "column -- the whole sangraha_verified_pdf type is used instead.",
    "sangraha_verified_speech_nptel":
        "Requirement 3.7 asks for the NPTEL transcript slice, but Sangraha "
        "ships no sub-source column -- the whole sangraha_verified_speech type "
        "is used instead.",
    "fineweb2_top2pct":
        "Requirement 3.7 asks for the top 2% by classifier quality score, but "
        "those scores were never computed at scale (a full pass is 157-393 "
        "GPU-hours). A metadata proxy is used instead -- see "
        "fineweb2_top_slice.py.",
}

DEFAULT_MAX_UPSAMPLE = 2.0


def decay_targets(variant):
    variant = variant.lower()
    if variant == "a":
        return dict(DECAY_A_TARGETS)
    if variant == "b":
        return dict(DECAY_B_TARGETS)
    raise ValueError(f"decay variant must be 'a' or 'b', got {variant!r}")


def plan_decay_mixture(variant, measured_available_tokens, total_decay_tokens,
                       max_upsample=DEFAULT_MAX_UPSAMPLE):
    """Plan one decay variant.

    `measured_available_tokens` is keyed by LOGICAL COMPONENT name (the keys of
    DECAY_A_TARGETS / DECAY_B_TARGETS), in tokens -- not documents. Planning in
    document space was Bug 9; tokens/doc varies several fold across these
    sources (OCR'd PDF pages are long, speech transcripts are long, Wikipedia
    articles are short), so a document-space plan realizes materially different
    token shares than Requirement 3.7 specifies.

    Returns a dict with `allocation` (component -> tokens), `upsample`
    (component -> repetition factor >= 1.0), `realized_shares`, `deviations`,
    and `realized_total`.
    """
    targets = decay_targets(variant)

    # Requirement 3.3: fineweb-edu-hindi is Decay B only. Enforced rather than
    # trusted, because leaking machine-translation-adjacent data into Decay A
    # would destroy the only thing the A/B comparison is there to measure
    # (Requirement 8.2).
    if variant.lower() == "a":
        for source in DECAY_ONLY_SOURCES:
            if source in targets:
                raise AssertionError(
                    f"{source} must not appear in Decay A targets "
                    f"(Requirement 3.3/3.7)")
            if measured_available_tokens.get(source):
                raise AssertionError(
                    f"{source} was supplied as available for Decay A. Decay A "
                    f"is native-only (Requirement 7.3); including it would "
                    f"invalidate the A-vs-B comparison in Requirement 8.2.")

    if max_upsample < 1.0:
        raise ValueError("max_upsample must be >= 1.0 (1.0 means no repetition)")

    deviations = []

    # Effective supply = real supply x the permitted repetition factor.
    unique = {c: max(0, int(measured_available_tokens.get(c, 0)))
              for c in targets}
    effective = {c: int(unique[c] * max_upsample) for c in targets}

    allocation = _water_fill(targets, effective, int(total_decay_tokens))

    # Derive the repetition factor each component actually needs, and record a
    # deviation wherever the nominal share could not be met from unique tokens.
    upsample = {}
    for component, tokens in allocation.items():
        have = unique[component]
        if have <= 0:
            upsample[component] = 1.0
            if targets[component] > 0 and tokens > 0:
                deviations.append(
                    f"{component}: allocated {tokens:,} tokens but measured "
                    f"availability is ZERO -- acquire it or drop the component")
            continue
        factor = tokens / have
        upsample[component] = max(1.0, factor)
        if factor > 1.0:
            deviations.append(
                f"{component}: nominal {targets[component]:.0%} needs "
                f"{tokens:,} tokens against {have:,} unique available -- "
                f"upsampled {factor:.2f}x")

    realized_total = sum(allocation.values())
    realized_shares = {
        c: (allocation[c] / realized_total if realized_total else 0.0)
        for c in allocation
    }

    for component, share in realized_shares.items():
        nominal = targets[component]
        if abs(share - nominal) > 0.02:
            deviations.append(
                f"{component}: realized share {share:.1%} vs nominal "
                f"{nominal:.0%} (Requirement 3.7)")

    # 0.1% tolerance: _water_fill truncates to int per component, so an exactly
    # satisfiable plan can land a few tokens light. Reporting that as a
    # shortfall would bury the real ones in noise.
    shortfall = total_decay_tokens - realized_total
    if shortfall > max(1000, int(total_decay_tokens * 0.001)):
        deviations.append(
            f"TOTAL: {realized_total:,} tokens against a {total_decay_tokens:,} "
            f"target -- short by {shortfall:,} ({shortfall / total_decay_tokens:.1%}). "
            f"Either grow a component with spare supply, raise --max-upsample, "
            f"or shorten the decay phase.")

    for component in allocation:
        if component in SUBSLICE_SUBSTITUTIONS and allocation[component] > 0:
            deviations.append(SUBSLICE_SUBSTITUTIONS[component])

    return {
        "variant": variant.lower(),
        "targets": targets,
        "unique_available": unique,
        "max_upsample": max_upsample,
        "allocation": allocation,
        "upsample": upsample,
        "realized_total": realized_total,
        "target_total": int(total_decay_tokens),
        "realized_shares": realized_shares,
        "deviations": deviations,
        "component_sources": {c: COMPONENT_SOURCES[c] for c in targets},
    }


def format_plan(plan):
    lines = [
        f"Decay {plan['variant'].upper()} plan "
        f"(max upsample {plan['max_upsample']:.2f}x)",
        f"  target total: {plan['target_total'] / 1e9:.3f}B tokens   "
        f"realized: {plan['realized_total'] / 1e9:.3f}B",
        "",
        f"  {'component':34s} {'nominal':>8s} {'realized':>9s} "
        f"{'tokens':>12s} {'unique':>12s} {'xN':>6s}",
    ]
    for component in sorted(plan["allocation"],
                            key=lambda c: -plan["allocation"][c]):
        lines.append(
            f"  {component:34s} {plan['targets'][component]:7.0%} "
            f"{plan['realized_shares'][component]:8.1%} "
            f"{plan['allocation'][component]:12,} "
            f"{plan['unique_available'][component]:12,} "
            f"{plan['upsample'][component]:5.2f}x")
    if plan["deviations"]:
        lines.append("")
        lines.append(f"  DEVIATIONS ({len(plan['deviations'])}) -- record these, "
                     f"do not silently ship them:")
        for note in plan["deviations"]:
            lines.append(f"    - {note}")
    return "\n".join(lines)
