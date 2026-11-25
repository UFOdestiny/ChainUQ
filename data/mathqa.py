"""MathQA multiple-choice reasoning dataset."""

import random
import re

from datasets import load_from_disk

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_choice_index
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from data.split_utils import ensure_train_validation_test, resolve_split_name
from utils.log import get_logger

log = get_logger(__name__)


_LETTER_TO_INDEX = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def _parse_option_string(raw_options: str) -> list[str]:
    text = str(raw_options or "").strip()
    if not text:
        return []
    chunks = re.split(r"\s*,\s*(?=[a-e]\s*\))", text, flags=re.IGNORECASE)
    options = []
    for chunk in chunks:
        match = re.match(r"\s*([a-e])\s*\)\s*(.*)\s*$", chunk, flags=re.IGNORECASE)
        if not match:
            continue
        options.append(match.group(2).strip())
    return options


def _correct_to_index(correct_value: str) -> str:
    text = str(correct_value or "").strip().lower()
    if text in _LETTER_TO_INDEX:
        return str(_LETTER_TO_INDEX[text])
    if text.isdigit():
        val = int(text)
        if 0 <= val <= 4:
            return str(val)
    return "unknown"


class MathQADataset(BaseReasoningDataset):
    name = "mathqa"
    task_type = "classification"
    needs_judge = False
    difficulty_field = "n_hops"

    system_prompt = build_system_prompt(
        objective="Solve the math word problem and choose the best option.",
        conclusion_schema="The answer is <option number from 1 to 5>.",
        examples=[
            {
                "input": "Question: If 2x=10, what is x?\nOptions:\n1. 2\n2. 3\n3. 5\n4. 7\n5. 10",
                "output": (
                    "Reasoning:\n"
                    "Step 1: Divide both sides of 2x=10 by 2.\n"
                    "Step 2: This gives x=5, which matches option 3.\n"
                    "Conclusion: The answer is 3."
                ),
            }
        ],
        extra_rules=[
            "Compute the numeric result before matching it to an option.",
            "Conclusion must be exactly: The answer is <option number>.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/MathQA/hf_dataset"
        super().__init__(
            split=split,
            max_samples=max_samples,
            n_hop_values=n_hop_values,
            cache_dir=cache_dir,
        )

    def _load_data(self):
        split_name = resolve_split_name(self.split)
        log.info("Loading MathQA split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        ds, used_fallback = ensure_train_validation_test(ds, seed=GLOBAL_SEED)
        if used_fallback:
            log.info("MathQA missing full train/validation/test; using deterministic 8:1:1 split.")
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
        split_ds = ds[split_name]

        items = []
        for raw in split_ds:
            options = _parse_option_string(raw.get("options", ""))
            if not options:
                continue
            answer_index = _correct_to_index(raw.get("correct", ""))
            if answer_index == "unknown":
                continue
            raw = dict(raw)
            raw["option_list"] = options
            raw["answer"] = answer_index
            raw["n_hops"] = 1
            raw["difficulty"] = 1
            items.append(raw)

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d MathQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        options = raw_item.get("option_list") or []
        options_text = "\n".join(f"{idx + 1}. {opt}" for idx, opt in enumerate(options))
        user_msg = (
            f"Question: {raw_item.get('Problem', '')}\n"
            f"Options:\n{options_text}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        predicted = self.parse_answer(generated)
        return 1 if predicted == str(ground_truth) else 0

    def parse_answer(self, generated_text):
        return parse_choice_index(generated_text, num_options=5, allow_letters=True)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 1)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", ""))
