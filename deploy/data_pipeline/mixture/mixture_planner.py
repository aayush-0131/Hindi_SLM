"""
Recomputes stable-phase mixture percentages from MEASURED survival/keep rates
(not the estimates in requirements.md SS2). Caps IndicCorp v2 opportunistically
(Requirement 3.5), enforces the PDF ceiling (Requirement 3.8), and keeps
fineweb-edu-hindi out of the stable phase entirely (Requirement 3.3).
"""

from data_pipeline.config.mixture_targets import (
    STABLE_MIXTURE_TARGETS, PDF_SOURCES, PDF_CEILING,
    INDICCORPV2_CAP, DECAY_ONLY_SOURCES,
)


def _water_fill(target_shares: dict, capped_available: dict, total_budget: int, max_iterations=10):
    """
    Allocates total_budget across sources proportional to target_shares, but
    whenever a source's availability can't meet its proportional target, caps
    it at its real availability and redistributes the shortfall among the
    remaining sources (proportional to their own shares) rather than just
    losing that budget -- this is the actual "recompute from measured rates"
    behavior Requirement 4.5 calls for, which a simple independent per-source
    min() does not provide (a shortfall in one source used to just shrink the
    total, never getting redistributed to sources with spare availability).
    """
    remaining = dict(target_shares)
    allocation = {}
    remaining_budget = total_budget

    for _ in range(max_iterations):
        share_total = sum(remaining.values())
        if share_total <= 0:
            break
        newly_capped = []
        for source, share in remaining.items():
            proportional_target = remaining_budget * (share / share_total)
            if capped_available.get(source, 0) < proportional_target:
                newly_capped.append(source)
        if not newly_capped:
            break
        for source in newly_capped:
            allocation[source] = capped_available.get(source, 0)
            remaining_budget -= allocation[source]
            del remaining[source]

    share_total = sum(remaining.values())
    for source, share in remaining.items():
        allocation[source] = int(remaining_budget * (share / share_total)) if share_total > 0 else 0

    return allocation


def plan_stable_mixture(measured_available_tokens: dict, total_stable_tokens: int):
    """
    measured_available_tokens: {source: post-filter-and-dedup token count actually available}
    total_stable_tokens: target stable-phase token budget (e.g. ~11.75e9)
    Returns {source: planned_token_count}.
    """
    for banned in DECAY_ONLY_SOURCES:
        assert banned not in STABLE_MIXTURE_TARGETS, (
            f"{banned} must never appear in the stable-phase plan (Requirement 3.3)"
        )

    # indiccorpv2 is capped opportunistically (Requirement 3.5) regardless of
    # how much water-filling would otherwise want to give it.
    capped_available = dict(measured_available_tokens)
    indiccorpv2_cap = int(total_stable_tokens * INDICCORPV2_CAP)
    if "indiccorpv2" in capped_available:
        capped_available["indiccorpv2"] = min(capped_available["indiccorpv2"], indiccorpv2_cap)

    plan = _water_fill(STABLE_MIXTURE_TARGETS, capped_available, total_stable_tokens)

    for source, target_share in STABLE_MIXTURE_TARGETS.items():
        target_tokens = int(total_stable_tokens * target_share)
        if plan.get(source, 0) < target_tokens:
            print(f"[mixture_planner] {source} realized {plan.get(source, 0)} tokens, under its "
                  f"{target_tokens} nominal target -- shortfall redistributed to other sources "
                  f"with spare availability (Requirement 4.5).")

    # Enforce the PDF ceiling by capping PDF sources down to exactly 25% of
    # whatever total is actually achievable, rather than erroring out --
    # matches the project's preference for making forward progress over
    # halting on a ratio check when the shortfall is a real availability
    # constraint, not a bug (Requirement 3.8).
    pdf_tokens = sum(plan.get(s, 0) for s in PDF_SOURCES)
    total_planned = sum(plan.values())
    if total_planned and (pdf_tokens / total_planned) > PDF_CEILING:
        non_pdf_tokens = total_planned - pdf_tokens
        pdf_allowed_total = non_pdf_tokens * (PDF_CEILING / (1 - PDF_CEILING))
        scale = pdf_allowed_total / pdf_tokens if pdf_tokens else 0
        for s in PDF_SOURCES:
            if s in plan:
                plan[s] = int(plan[s] * scale)
        print(f"[mixture_planner] PDF-derived share {pdf_tokens / total_planned:.1%} exceeded the "
              f"{PDF_CEILING:.0%} ceiling -- capped PDF sources down to fit (Requirement 3.8), "
              f"accepting a smaller total ({sum(plan.values())} tokens) rather than inventing "
              f"non-PDF tokens that aren't actually available.")

    return plan


def loosen_keep_rate(current_keep_rate: float, target_keep_rate: float = 0.30):
    """Requirement 4.7 fallback: if the corpus is short of 17B unique tokens,
    widen the FineWeb-2 classifier keep rate toward 30% rather than adding
    epochs over the existing pool."""
    if current_keep_rate >= target_keep_rate:
        return current_keep_rate
    print(f"[mixture_planner] loosening FineWeb-2 keep rate {current_keep_rate:.0%} "
          f"-> {target_keep_rate:.0%} (Requirement 4.7)")
    return target_keep_rate
