"""
Candidate Hindi task suite and baseline model registry for Gate 2
(Requirement 5). Task name strings were originally CANDIDATES (blog-post
guesses), not confirmed-correct -- research.md flags two open lighteval
issues (#606, #509) showing such guesses sometimes don't actually resolve
in the registry for other languages. That happened here: the guesses used
"hi" + "mcfformulation"; the installed lighteval's TASKS_TABLE actually
registers "hin" (Language.HINDI.value, ISO 639-3) + "mcf"
(MCFFormulation().name.lower()). Verified 2026-09-07 on the training server
by importing each multilingual task module directly and reading
`t.name for t in mod.TASKS_TABLE` -- see research.md Bug 15, not by guessing
again. `community_boolq_hin` also required installing the
`lighteval[multilingual]` extra; without it the whole module fails to
import (a silent-looking failure that presents identically to "task not
found"). gate_report.py still smoke-tests every candidate with --limit 5
first and drops anything that doesn't resolve, as defense in depth -- these
strings are now verified, not guessed, but the drop-on-failure behavior
stays.
"""

# Each entry: (lighteval task spec "name|n_shots", num_choices for the
# chance-baseline check used by is_non_random()).
CANDIDATE_TASKS = [
    ("community_arc_hin_mcf:easy|0", 4),
    ("community_arc_hin_mcf:challenge|0", 4),
    ("community_boolq_hin|0", 2),
    ("community_hellaswag_hin_mcf|0", 4),
    ("belebele_hin_Deva_mcf|0", 4),
]

# Primary comparison: same architecture family (causal LM) as the target
# model, so the harness's behavior here is the most direct proxy for how it
# will behave against the actual trained checkpoint.
PRIMARY_BASELINE = {
    "name": "goldfish-hindi",
    "hf_repo": "goldfish-models/hin_deva_1000mb",
    "architecture": "causal_lm",
    # See eval_harness/tokenizer_patch.py for the full story: this repo's own
    # tokenizer_config.json declares cls_token="[CLS]"/sep_token="[SEP]" as
    # ADDED tokens (ids 50000/50001), which transformers.AlbertTokenizer
    # cannot resolve at the exact point it builds its BERT-style
    # TemplateProcessing post-processor, crashing with
    # `TypeError: Expected Union[Tuple[str, int], Tuple[int, str], dict]` on
    # EVERY load of this repo -- confirmed independent of lighteval and of
    # use_fast. Harmless override (verified 2026-09-07): lighteval never
    # reads cls_token/sep_token for causal-LM loglikelihood scoring, and
    # bos/eos/pad/unk plus real text encode/decode are unaffected.
    "tokenizer_patch": {"cls_token": "<s>", "sep_token": "</s>"},
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

# ---------------------------------------------------------------------------
# Binary tasks are not usable as a PASS/FAIL gate against a size-matched
# baseline.
#
# Requirement 5.4/5.5 demand a non-random score for every task in the suite,
# and gate_report._is_non_random requires score > chance * 1.15. For a
# 2-choice task that is > 0.575. Goldfish-Hindi is 125M parameters; scoring
# near 0.5 on binary Hindi reading comprehension is the EXPECTED outcome, not
# a harness defect -- so including it would fail Gate 2 spuriously and block
# Requirement 8.2's decay-variant selection, which is the thing Gate 2 exists
# to enable.
#
# The reconciliation is the one ROUND2_HANDOFF.md sec 5 proposes and the one
# the existing two-phase design already implies: this is a SUITE-SCOPING
# decision, so it belongs in Phase A curation, before the gate formally runs --
# not a silent skip once Phase B has started. Phase B stays strictly
# all-or-nothing over whatever Phase A curated.
#
# The task is still RUN and its score still RECORDED; it is just not permitted
# to fail the gate. That keeps the measurement without the false negative.
MIN_DISCRIMINATIVE_CHOICES = 3

# Set True to force binary tasks back into the pass/fail gate (e.g. once the
# target model is strong enough that near-chance would be a real signal).
GATE_ON_BINARY_TASKS = False
