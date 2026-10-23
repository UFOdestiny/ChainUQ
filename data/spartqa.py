"""SpartQA multiple-choice spatial reasoning dataset.

SpartQA (tasksource/spartqa-mchoice) is a 4-option multiple-choice QA task.
Columns: story, question, candidate_answers, answer (0-indexed int)
Classification task — no LLM judge needed.
"""

import random
from typing import Dict, List

from datasets import load_from_disk, DatasetDict

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_choice_index
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


class SpartQADataset(BaseReasoningDataset):
    """SpartQA multiple-choice QA — 4-class classification."""

    name = "spartqa"
    task_type = "classification"
    needs_judge = False
    difficulty_field = "n_hops"

    system_prompt = build_system_prompt(
        objective="Choose the correct option using spatial reasoning from the context.",
        conclusion_schema="The answer is <option number 1/2/3/4>.",
        examples=[
            {
                "input": (
                    "Context:\nThe cup is left of the plate.\n"
                    "Question: Where is the cup relative to the plate?\n"
                    "Options:\n1. Right of the plate\n2. Left of the plate\n3. Above the plate\n4. Below the plate"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The context directly says the cup is left of the plate.\n"
                    "Step 2: Option 2 matches that relation.\n"
                    "Conclusion: The answer is 2."
                ),
            }
        ],
        extra_rules=[
            "Compare the derived spatial relation against the numbered options.",
            "Conclusion must be exactly: The answer is <option number>.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None,
                 cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/spartqa/hf_dataset"
        super().__init__(split=split, max_samples=max_samples,
                         n_hop_values=n_hop_values, cache_dir=cache_dir)

    def _load_data(self):
        log.info("Loading SpartQA split=%s from %s", self.split, self.dataset_path)
        ds = load_from_disk(self.dataset_path)
        if isinstance(ds, DatasetDict):
            split_name = "validation" if self.split == "val" else self.split
            if split_name not in ds:
                raise ValueError(
                    f"Split '{self.split}' not found. Available: {list(ds.keys())}"
                )
            ds = ds[split_name]

        items = list(ds)
        # SpartQA has no native difficulty; set n_hops=1
        for item in items:
            item["n_hops"] = 1

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d SpartQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item: dict) -> List[Dict[str, str]]:
        story = raw_item["story"]
        question = raw_item["question"]
        candidates = raw_item["candidate_answers"]
        options_text = "\n".join(
            f"{i + 1}. {opt.strip()}" for i, opt in enumerate(candidates)
        )
        user_msg = (
            f"Context:\n{story}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options_text}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated: str, ground_truth: str) -> int:
        predicted = self.parse_answer(generated)
        return 1 if predicted == ground_truth else 0

    def parse_answer(self, generated_text: str) -> str:
        """Extract option number (0-3) from LLM output (LLM outputs 1-4)."""
        return parse_choice_index(generated_text, num_options=4, allow_letters=False)

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("n_hops", 1)

    def get_ground_truth(self, raw_item: dict) -> str:
        return str(raw_item["answer"])  # 0-indexed
