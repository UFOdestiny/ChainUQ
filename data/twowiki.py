"""2WikiMultihopQA dataset."""

import random

from datasets import DatasetDict, load_from_disk

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_free_form_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


class TwoWikiMultihopQADataset(BaseReasoningDataset):
    name = "2wikimultihopqa"
    task_type = "free_form"
    needs_judge = True
    difficulty_field = "n_hops"
    context_char_budget = 8000

    system_prompt = build_system_prompt(
        objective="Answer the multi-hop question by connecting evidence across passages.",
        conclusion_schema="[final short answer]",
        examples=[
            {
                "input": (
                    "Context:\n"
                    "[Book] Dune was written by Frank Herbert.\n"
                    "[Author] Frank Herbert was born in Tacoma.\n"
                    "Question: Where was the author of Dune born?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The author of Dune is Frank Herbert.\n"
                    "Step 2: Frank Herbert was born in Tacoma.\n"
                    "Conclusion: Tacoma."
                ),
            }
        ],
        extra_rules=[
            "Each Step should connect exactly one entity or relation across passages.",
            "Conclusion must be only the answer span, not a full sentence.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/2WikiMultihopQA/hf_dataset"
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

    @staticmethod
    def _infer_n_hops(item: dict) -> int:
        supporting = item.get("supporting_facts", {}).get("title", []) or []
        if supporting:
            return max(2, min(4, len(supporting)))
        evidences = item.get("evidences", []) or []
        if evidences:
            return max(2, min(4, len(evidences)))
        return 2

    def _load_data(self):
        split_name = self._resolve_split(self.split)
        log.info("Loading 2WikiMultihopQA split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        if isinstance(ds, DatasetDict):
            if split_name not in ds:
                raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
            ds = ds[split_name]

        items = list(ds)
        for item in items:
            item["n_hops"] = self._infer_n_hops(item)
            item["difficulty"] = item["n_hops"]

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d 2WikiMultihopQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        context_parts = []
        titles = raw_item.get("context", {}).get("title", []) or []
        sentences = raw_item.get("context", {}).get("sentences", []) or []
        for title, sents in zip(titles, sentences):
            if isinstance(sents, list):
                text = "".join(sents)
            else:
                text = str(sents)
            context_parts.append(f"[{title}] {text}")
        context_str, truncated = self.pack_context_blocks(
            context_parts,
            max_chars=self.context_char_budget,
        )
        if truncated:
            context_str += "\n\n[Note] Context truncated for length."

        user_msg = f"Context:\n{context_str}\n\nQuestion: {raw_item.get('question', '')}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        gen_answer = self.parse_answer(generated).strip().lower()
        gt = str(ground_truth).strip().lower()
        if gen_answer == gt:
            return 1
        if gt and (gt in gen_answer or gen_answer in gt):
            return 1
        return -1

    def parse_answer(self, generated_text):
        return parse_free_form_answer(generated_text)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", ""))
