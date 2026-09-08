"""
Candidate Hindi task suite and baseline model registry for Gate 2
(Requirement 5). Task name strings are CANDIDATES, not confirmed-correct --
research.md flags two open lighteval issues (#606, #509) showing blog-post-
claimed tasks sometimes don't actually resolve in the registry for other
languages. gate_report.py smoke-tests every candidate with --limit 5 first
and drops anything that doesn't resolve, rather than trusting these strings.
"""

# Each entry: (lighteval task spec "name|n_shots", num_choices for the
# chance-baseline check used by is_non_random()).
CANDIDATE_TASKS = [
    ("community_arc_hi_mcfformulation:easy|0", 4),
    ("community_arc_hi_mcfformulation:challenge|0", 4),
    ("community_boolq_hindi|0", 2),
    ("community_hellaswag_hi_mcfformulation|0", 4),
    ("belebele_hin_Deva_mcfformulation|0", 4),
]

# Primary comparison: same architecture family (causal LM) as the target
# model, so the harness's behavior here is the most direct proxy for how it
# will behave against the actual trained checkpoint.
PRIMARY_BASELINE = {
    "name": "goldfish-hindi",
    "hf_repo": "goldfish-models/hin_deva_1000mb",
    "architecture": "causal_lm",
}

# Encoder references (Requirement 6) -- MuRIL and IndicBERT v2 are masked-LM
# encoders, not causal LMs. lighteval's standard loglikelihood-based
# multiple-choice scoring assumes an autoregressive model; scoring an
# encoder the same way needs a different method (e.g. pseudo-log-likelihood)
# that is NOT implemented here yet. Confirmed not gated (safe to fetch), but
# excluded from the automated pass/fail suite below -- see run_gate2.py's
# --skip-encoder-baselines default and the flagged scope decision in
# research.md.
ENCODER_REFERENCE_BASELINES = [
    {"name": "muril", "hf_repo": "google/muril-base-cased", "architecture": "encoder_mlm"},
    {"name": "indicbert-v2", "hf_repo": "ai4bharat/IndicBERTv2-MLM-only", "architecture": "encoder_mlm"},
]

SMOKE_TEST_LIMIT = 5
