import json
from pathlib import Path
from tasks.common import Task

class HistoryHiSFT(Task):
    def __init__(self, split="train", data_root=None, **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val"}
        repo_root = Path(__file__).resolve().parents[2]
        data_root = Path(data_root) if data_root else repo_root / "data" / "history_hi_sft_v1"
        filename = "history_hi_sft_v1_train.jsonl" if split == "train" else "history_hi_sft_v1_val.jsonl"
        with (data_root / filename).open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return {"messages": self.rows[index]["messages"]}

    def evaluate(self, problem, completion):
        raise NotImplementedError
