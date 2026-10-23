"""bAbI QA dataset for multi-step reasoning.

bAbI (Muennighoff/babi) is a free-form QA task with short-answer responses.
Columns: passage, question, answer, task
Tasks map to approximate reasoning hops (1-3).
"""

import random
from typing import Dict, List

from datasets import load_from_disk, DatasetDict

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_free_form_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)

# Approximate mapping from bAbI task number to reasoning hops
TASK_TO_HOPS = {
    1: 1, 2: 2, 3: 3, 4: 2, 5: 3,
    6: 3, 7: 2, 8: 3, 9: 2, 10: 1,
    11: 2, 12: 1, 13: 2, 14: 2, 15: 1,
    16: 2, 17: 2, 18: 1, 19: 3, 20: 3,
}


class BabiDataset(BaseReasoningDataset):
    """bAbI QA dataset — free-form short answers requiring LLM judge."""

    name = "babi"
    task_type = "free_form"
    needs_judge = True
    difficulty_field = "n_hops"

    system_prompt = build_system_prompt(
        objective="Answer the question by tracking entities and events in the passage.",
        conclusion_schema="[final short answer]",
        examples=[
            {
                "input": (
                    "Passage:\nMary went to the kitchen. John moved to the garden.\n"
                    "Question: Where is John?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The passage says John moved to the garden.\n"
                    "Step 2: No later sentence changes John's location.\n"
                    "Conclusion: garden."
                ),
            }
        ],
        extra_rules=[
            "Track the latest relevant entity state; ignore earlier states overwritten later.",
            "Conclusion must be only the requested entity, location, object, or yes/no answer.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 2000,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None,
                 cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/babi/hf_dataset"
        super().__init__(split=split, max_samples=max_samples,
                         n_hop_values=n_hop_values, cache_dir=cache_dir)

    def _load_data(self):
        log.info("Loading bAbI split=%s from %s", self.split, self.dataset_path)
        ds = load_from_disk(self.dataset_path)
        if isinstance(ds, DatasetDict):
            split_name = self.split
            if split_name == "val":
                split_name = "validation"
            if split_name not in ds:
                raise ValueError(
                    f"Split '{self.split}' not found. Available: {list(ds.keys())}"
                )
            ds = ds[split_name]

        items = list(ds)

        # Add n_hops from task mapping
        for item in items:
            task = item.get("task", 1)
            if isinstance(task, str):
                try:
                    task = int(task)
                except ValueError:
                    task = 1
            item["n_hops"] = TASK_TO_HOPS.get(task, 2)
            item["task"] = task

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d bAbI items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item: dict) -> List[Dict[str, str]]:
        passage = raw_item["passage"]
        if isinstance(passage, list):
            passage = "\n".join(passage)
        question = raw_item["question"]
        user_msg = f"Passage:\n{passage}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated: str, ground_truth: str) -> int:
        return -1  # Always needs LLM judge

    def parse_answer(self, generated_text: str) -> str:
        return parse_free_form_answer(generated_text)

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item: dict) -> str:
        return str(raw_item.get("answer", ""))
