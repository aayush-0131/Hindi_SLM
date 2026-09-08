"""
Measures tokens-per-word fertility on the held-out FineWeb-2 sample and on
FLORES-200 Hindi dev+devtest combined (Requirement 2.2, corrected per
research.md to use openlanguagedata/flores_plus and the dev+devtest-combined
methodology the reference figures were measured with -- facebook/flores is
deprecated and the reference figures pool dev+devtest, not devtest alone).
"""

from datasets import load_dataset


def load_flores_hin_deva_combined():
    dev = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="dev")
    devtest = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="devtest")
    return [row["text"] for row in dev] + [row["text"] for row in devtest]


def compute_fertility(tokenizer, texts):
    """Tokens-per-word, using whitespace-delimited words -- research.md found
    this is the convention the reference fertility figures were computed
    with (no language-aware segmenter documented for the source paper)."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        total_tokens += len(tokenizer.encode(text))
        total_words += len(text.split())
    return total_tokens / total_words if total_words else float("inf")


def evaluate_candidate(tokenizer, heldout_fineweb2_texts, flores_texts):
    return {
        "fertility_fineweb2": compute_fertility(tokenizer, heldout_fineweb2_texts),
        "fertility_flores": compute_fertility(tokenizer, flores_texts),
    }
