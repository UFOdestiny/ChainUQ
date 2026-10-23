"""StrategyQA dataset."""

import random
import re

from datasets import DatasetDict, load_from_disk

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_yes_no_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


def _yes_no_text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value or "").strip().lower()
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"
    return text


def _normalize_fact_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    facts = [part.strip() for part in parts if part.strip()]
    return facts or [text]


class StrategyQADataset(BaseReasoningDataset):
    name = "strategyqa"
    task_type = "free_form"
    needs_judge = False
    difficulty_field = "n_hops"
    context_char_budget = 5000

    system_prompt = build_system_prompt(
        objective="Use the provided facts to answer the yes/no question.",
        conclusion_schema="[yes or no]",
        examples=[
            {
                "input": "Facts:\n- Penguins are birds.\n- Birds are animals.\nQuestion: Are penguins animals?",
                "output": (
                    "Reasoning:\n"
                    "Step 1: Penguins are birds according to the facts.\n"
                    "Step 2: Birds are animals according to the facts.\n"
                    "Conclusion: yes."
                ),
            }
        ],
        extra_rules=[
            "Use the facts only; do not rely on outside knowledge.",
            "Conclusion must be exactly 'yes' or 'no'.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/StrategyQA/hf_dataset"
        super().__init__(
            split=split,
            max_samples=max_samples,
            n_hop_values=n_hop_values,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _resolve_split(split: str) -> str:
        key = (split or "train").strip().lower()
        if key in ("validation", "val"):
            return "validation"
        if key == "test":
            return "test"
        return "train"

    def _load_data(self):
        split_name = self._resolve_split(self.split)
        log.info("Loading StrategyQA split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        if not isinstance(ds, DatasetDict):
            raise ValueError("Expected DatasetDict with explicit train/validation/test splits for StrategyQA.")
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
        ds = ds[split_name]

        items = list(ds)
        for item in items:
            decomposition = item.get("decomposition") or []
            facts = _normalize_fact_list(item.get("facts"))
            item["facts"] = facts
            n_hops = len(decomposition) if isinstance(decomposition, list) and decomposition else len(facts)
            item["answer"] = _yes_no_text(item.get("answer"))
            item["n_hops"] = max(2, min(5, n_hops or 2))
            item["difficulty"] = item["n_hops"]

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d StrategyQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        context_parts = []
        term = str(raw_item.get("term", "")).strip()
        description = str(raw_item.get("description", "")).strip()
        if term and description:
            context_parts.append(f"[Topic] {term}: {description}")

        for fact in raw_item.get("facts") or []:
            fact_text = str(fact).strip()
            if fact_text:
                context_parts.append(f"- {fact_text}")

        context_str, truncated = self.pack_context_blocks(
            context_parts,
            max_chars=self.context_char_budget,
        )
        if truncated:
            context_str += "\n\n[Note] Facts truncated for length."

        user_msg = f"Facts:\n{context_str}\n\nQuestion: {raw_item.get('question', '')}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        gen = _yes_no_text(self.parse_answer(generated))
        gt = _yes_no_text(ground_truth)
        if not gen:
            return 0
        return 1 if gen == gt else 0

    def parse_answer(self, generated_text):
        return _yes_no_text(parse_yes_no_answer(generated_text))

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item):
        return _yes_no_text(raw_item.get("answer"))
