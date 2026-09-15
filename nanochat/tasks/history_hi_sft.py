"""Local History & Cultural Heritage SFT task for the Hindi SLM capstone.

Reads the frozen capstone SFT JSONL from data/history_hi_sft_v1. This module
contains no HistoryBench content and performs no network access. It deliberately
validates conversation structure before returning examples to the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

from tasks.common import Task


class HistoryHiSFT(Task):
    """Deterministic local JSONL-backed task for History_HI_SFT_v1."""

    VALID_SPLITS = {"train", "val", "validation"}

    def __init__(self, split: str, data_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        assert split in self.VALID_SPLITS, "HistoryHiSFT split must be train|val|validation"
        canonical_split = "val" if split == "validation" else split

        if data_dir is None:
            # nanochat/tasks/history_hi_sft.py -> repo root is parents[2]
            data_root = Path(__file__).resolve().parents[2] / "data" / "history_hi_sft_v1"
        else:
            data_root = Path(data_dir).expanduser().resolve()

        candidates = (
            data_root / f"{canonical_split}.jsonl",
            data_root / ("validation.jsonl" if canonical_split == "val" else "train.jsonl"),
        )
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            names = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"HistoryHiSFT {canonical_split} JSONL not found; tried: {names}")

        self.path = path
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path}:{lineno}: {e}") from e
                self._validate_row(row, path, lineno)
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"HistoryHiSFT split is empty: {path}")

    @staticmethod
    def _validate_row(row, path: Path, lineno: int) -> None:
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            raise ValueError(f"{path}:{lineno}: expected object with a messages list")
        messages = row["messages"]
        if len(messages) < 2:
            raise ValueError(f"{path}:{lineno}: conversation must have at least two messages")

        start = 1 if messages[0].get("role") == "system" else 0
        rest = messages[start:]
        if len(rest) < 2:
            raise ValueError(f"{path}:{lineno}: conversation needs user and assistant messages")
        for i, message in enumerate(rest):
            if not isinstance(message, dict):
                raise ValueError(f"{path}:{lineno}: message {i} is not an object")
            expected = "user" if i % 2 == 0 else "assistant"
            if message.get("role") != expected:
                raise ValueError(
                    f"{path}:{lineno}: message {i} role={message.get('role')!r}, expected {expected!r}"
                )
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{path}:{lineno}: message {i} content must be a non-empty string")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        # Return only the field consumed by NanoChat's SFT token renderer; ignore
        # any provenance/metadata fields retained in the source JSONL.
        return {"messages": self.rows[index]["messages"]}
