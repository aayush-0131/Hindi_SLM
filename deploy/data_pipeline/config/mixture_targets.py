"""Fixed source identifiers, exclusion guard list, and target mixture ratios (Requirement 3)."""

# The seven acquired base-mixture sources (Requirement 3.1, 3.5).
SOURCES = [
    "fineweb2",
    "sangraha_verified_web",
    "sangraha_verified_pdf",
    "sangraha_verified_speech",
    "sangraha_unverified",
    "finepdfs",
    "indiccorpv2",
]

# Sources explicitly excluded from the corpus (Requirement 3.2) -- never acquired.
EXCLUDED_SOURCES = [
    "PIB",
    "soketlabs/bhasha-wiki",
    "Sangraha Synthetic",
    "CulturaX (direct)",
    "MADLAD-400 (direct)",
    "mC4 (direct)",
]

# Stable-phase target mixture, by token share (Requirement 3.6).
# indiccorpv2's 2% is a cap, not a guaranteed fill -- see mixture_planner.py.
STABLE_MIXTURE_TARGETS = {
    "fineweb2": 0.62,
    "sangraha_verified_web": 0.16,
    "sangraha_verified_pdf": 0.10,
    "sangraha_unverified": 0.04,
    "finepdfs": 0.04,
    "sangraha_verified_speech": 0.02,
    "indiccorpv2": 0.02,
}

assert abs(sum(STABLE_MIXTURE_TARGETS.values()) - 1.0) < 1e-9

# PDF-derived sources and their combined ceiling (Requirement 3.8).
PDF_SOURCES = ["sangraha_verified_pdf", "finepdfs"]
PDF_CEILING = 0.25

# indiccorpv2's cap (Requirement 3.5) -- opportunistic, never backfilled if short.
INDICCORPV2_CAP = 0.02

# fineweb-edu-hindi is decay-phase only (Requirement 3.3) -- must never appear
# in STABLE_MIXTURE_TARGETS above.
DECAY_ONLY_SOURCES = ["fineweb_edu_hindi"]

# Decay-phase mixture variants (Requirement 3.7).
DECAY_A_TARGETS = {
    "sangraha_verified_pdf_textbooks": 0.41,
    "hindi_wikipedia": 0.24,
    "sangraha_verified_speech_nptel": 0.18,
    "fineweb2_top2pct": 0.17,
}
DECAY_B_TARGETS = {
    "sangraha_verified_pdf_textbooks": 0.35,
    "hindi_wikipedia": 0.20,
    "sangraha_verified_speech_nptel": 0.15,
    "fineweb2_top2pct": 0.15,
    "fineweb_edu_hindi": 0.15,
}
assert abs(sum(DECAY_A_TARGETS.values()) - 1.0) < 1e-9
assert abs(sum(DECAY_B_TARGETS.values()) - 1.0) < 1e-9

# Verify-before-including thresholds (Requirement 3.4).
HPLT_MIN_TOKENS_AFTER_DEDUP = 3_000_000_000
VARTA_MAX_SHARE = 0.02
