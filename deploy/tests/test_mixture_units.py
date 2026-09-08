"""Bug 9: the mixture planner was fed DOCUMENT counts where it expects TOKENS.

Uses the real numbers from the 2026-09-02 full-scale run so this is a
regression test against an actual observed failure, not a synthetic one.

The bug is not that the planner is wrong -- plan_stable_mixture() is
unit-agnostic proportional math. It is that Requirements 3.6/3.8 specify
shares BY TOKEN, and planning on document counts realizes different token
shares, because tokens/doc varies several-fold across these sources. The PDF
ceiling is the acceptance criterion that actually breaks: PDF documents are
the long ones, so capping PDF at 25% of DOCUMENTS leaves it well above 25% of
TOKENS.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.mixture.mixture_planner import plan_stable_mixture
from data_pipeline.config.mixture_targets import PDF_SOURCES, PDF_CEILING
from data_pipeline.shard.shard_writer import doc_limits_from_token_plan

# Survivor document counts, verbatim from the run's
# "measured_available_tokens (survivor doc counts)" line.
OBSERVED_DOCS = {
    "fineweb2": 11246,
    "sangraha_verified_web": 19959,
    "sangraha_verified_pdf": 20000,
    "sangraha_verified_speech": 1105,
    "sangraha_unverified": 19976,
    "finepdfs": 20000,
    "indiccorpv2": 20000,
}

# Per-source tokens/doc. The run only reported an aggregate (137,730,789 tokens
# / 96,380 planned docs = 1,429 avg), so these are a plausible spread with the
# PDF sources long and speech short -- which is the whole point. The real run
# now measures these directly via survival_sampling.measure_available_tokens().
TOKENS_PER_DOC = {
    "fineweb2": 900,
    "sangraha_verified_web": 700,
    "sangraha_verified_pdf": 4000,
    "sangraha_verified_speech": 300,
    "sangraha_unverified": 800,
    "finepdfs": 3500,
    "indiccorpv2": 250,
}

BUDGET = 11_750_000_000


def pdf_token_share(doc_allocation):
    """Realized PDF share in TOKENS, given an allocation expressed in documents."""
    tokens = {s: n * TOKENS_PER_DOC[s] for s, n in doc_allocation.items()}
    total = sum(tokens.values())
    return sum(tokens.get(s, 0) for s in PDF_SOURCES) / total if total else 0.0


def test_doc_space_planning_breaches_the_pdf_ceiling():
    """The old behavior: doc counts in, doc counts straight to the shard writer."""
    plan_docs = plan_stable_mixture(OBSERVED_DOCS, BUDGET)
    # The planner reports the ceiling as satisfied -- in DOCUMENT space.
    doc_share = sum(plan_docs.get(s, 0) for s in PDF_SOURCES) / sum(plan_docs.values())
    assert doc_share <= PDF_CEILING + 1e-6, doc_share
    # But the realized TOKEN share, which is what Requirement 3.8 governs,
    # is far above the ceiling.
    tok_share = pdf_token_share(plan_docs)
    assert tok_share > PDF_CEILING, tok_share
    print(f"  doc-space plan:  PDF {doc_share:.1%} of docs "
          f"-> {tok_share:.1%} of TOKENS (ceiling {PDF_CEILING:.0%}) -- BREACH")
    return tok_share


def test_token_space_planning_respects_the_pdf_ceiling():
    """The fix: measure tokens, plan in tokens, convert back to doc limits."""
    available_tokens = {s: n * TOKENS_PER_DOC[s] for s, n in OBSERVED_DOCS.items()}
    plan_tokens = plan_stable_mixture(available_tokens, BUDGET)

    tok_share = sum(plan_tokens.get(s, 0) for s in PDF_SOURCES) / sum(plan_tokens.values())
    assert tok_share <= PDF_CEILING + 1e-6, tok_share

    # And the conversion back to document limits must preserve that share.
    doc_limits = doc_limits_from_token_plan(plan_tokens, TOKENS_PER_DOC, OBSERVED_DOCS)
    realized = pdf_token_share(doc_limits)
    assert realized <= PDF_CEILING + 0.01, realized
    print(f"  token-space plan: PDF {tok_share:.1%} of TOKENS, "
          f"{realized:.1%} after doc conversion -- OK")


def test_doc_limits_never_exceed_availability():
    """Rounding up must never ask the shard writer for documents that don't exist."""
    available_tokens = {s: n * TOKENS_PER_DOC[s] for s, n in OBSERVED_DOCS.items()}
    plan_tokens = plan_stable_mixture(available_tokens, BUDGET)
    doc_limits = doc_limits_from_token_plan(plan_tokens, TOKENS_PER_DOC, OBSERVED_DOCS)
    for source, limit in doc_limits.items():
        assert limit <= OBSERVED_DOCS[source], (source, limit, OBSERVED_DOCS[source])
    print("  doc limits clamped to availability: OK")


def test_missing_tokens_per_doc_yields_zero_not_crash():
    """A source with no survivors has tokens_per_doc=None; must not divide by it."""
    limits = doc_limits_from_token_plan(
        {"a": 1000, "b": 500}, {"a": None, "b": 100}, {"a": 0, "b": 50}
    )
    assert limits["a"] == 0, limits
    assert limits["b"] == 5, limits
    print("  None tokens/doc handled: OK")


if __name__ == "__main__":
    print("--- old behavior (bug) ---")
    breached = test_doc_space_planning_breaches_the_pdf_ceiling()
    print("\n--- fixed behavior ---")
    test_token_space_planning_respects_the_pdf_ceiling()
    test_doc_limits_never_exceed_availability()
    test_missing_tokens_per_doc_yields_zero_not_crash()
    print(f"\nall mixture-unit tests passed "
          f"(bug would have shipped PDF at {breached:.1%} of tokens)")
