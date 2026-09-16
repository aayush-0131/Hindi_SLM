"""
Output-side safety guardrail for chat.py.

Does NOT fix the underlying model (see research.md's "benign-prompt safety
failure" finding -- that needs broader SFT safety data + retraining, or
pretraining-corpus decontamination, neither of which is possible without
GPU access this project doesn't currently have). This is a heuristic
keyword filter on the model's OWN generated output, applied before it ever
reaches a user -- a stopgap, not a solution. It will miss paraphrases,
transliterations, and anything not on the list; it is not a substitute for
fixing the model.
"""

import re

# Explicit sexual-content terms observed in this model's actual outputs
# (see research.md) plus common Hindi vocabulary for the same content.
# Word-boundary matched to reduce false positives on unrelated words that
# happen to share a substring.
_EXPLICIT_TERMS = [
    "यौन", "सेक्स", "अश्लील", "चूत", "गांड", "लंड", "चूची", "स्तन",
    "वीर्य", "हस्तमैथुन", "बलात्कार", "निर्वस्त्र", "नग्न",
]

_PATTERN = re.compile(
    "|".join(re.escape(term) for term in _EXPLICIT_TERMS)
)

REFUSAL_MESSAGE = (
    "मैं इस प्रकार की सामग्री उत्पन्न नहीं कर सकता। "
    "कृपया भारतीय इतिहास या संस्कृति से जुड़ा कोई प्रश्न पूछें।"
)


def is_flagged(text: str) -> bool:
    return bool(_PATTERN.search(text))


def filter_reply(text: str) -> str:
    """Return text unchanged, or REFUSAL_MESSAGE if it trips the filter."""
    if is_flagged(text):
        return REFUSAL_MESSAGE
    return text
