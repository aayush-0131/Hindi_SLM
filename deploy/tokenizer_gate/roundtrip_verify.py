"""
Verifies decode(encode(x)) == x across 10,000 sampled documents plus a
deliberately constructed fixture covering conjunct consonants, matras,
nukta, virama, ZWJ/ZWNJ, and mixed Devanagari+Latin script (Requirement 2.5,
2.6) -- these are the specific failure modes research.md's pre-tokenizer
finding identifies as the actual risk, not just a random sample.
"""

# Deliberately tricky Hindi/Devanagari edge cases.
SCRIPT_EDGE_CASES = [
    "क्षत्रिय",              # conjunct: क + ् + ष
    "ज्ञान",                 # conjunct: ज + ् + ञ
    "श्रीमान",               # conjunct: श + ् + र
    "क़िला",                  # nukta: क + ़
    "ख़ुशी",                  # nukta: ख + ़
    "ग़रीब",                  # nukta: ग + ़
    "ज़मीन",                  # nukta: ज + ़
    "फ़र्क",                  # nukta: फ + ़
    "हिन्दी",                 # virama/halant: न + ्
    "काम‍काज",           # ZWJ (zero-width joiner)
    "अच्छा‌नहीं",         # ZWNJ (zero-width non-joiner)
    "यह एक mixed-script वाक्य है",   # mixed Devanagari + Latin
    "नमस्ते! आप कैसे हैं?",      # punctuation
    "१२३ और 456",              # Devanagari + Latin digits
]


def sample_documents(text_iterable, n=10_000):
    docs = []
    for text in text_iterable:
        docs.append(text)
        if len(docs) >= n:
            break
    return docs


def verify_roundtrip(tokenizer, documents):
    """Returns a list of (index, text) for every document that fails
    decode(encode(x)) == x. Empty list means a full pass."""
    all_docs = list(documents) + SCRIPT_EDGE_CASES
    failures = []
    for i, text in enumerate(all_docs):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        if decoded != text:
            failures.append((i, text))
    return failures
