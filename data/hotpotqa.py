"""HotpotQA multi-hop reasoning dataset."""
import random

from datasets import load_from_disk, DatasetDict

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_free_form_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


class HotpotQADataset(BaseReasoningDataset):
    name = "hotpotqa"
    task_type = "free_form"
    needs_judge = True
    difficulty_field = "n_hops"
    context_char_budget = 8000

    system_prompt = build_system_prompt(
        objective="Answer the multi-hop question using only the provided context.",
        conclusion_schema="[final short answer]",
        examples=[
            {
                "input": (
                    "Context:\n"
                    "[A] Ada wrote the first algorithm for Babbage's engine.\n"
                    "[B] Ada Lovelace collaborated with Charles Babbage.\n"
                    "Question: Who collaborated with Charles Babbage?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: Passage B states Ada Lovelace collaborated with Charles Babbage.\n"
                    "Step 2: The asked collaborator is therefore Ada Lovelace.\n"
                    "Conclusion: Ada Lovelace."
                ),
            }
        ],
        extra_rules=[
            "Each Step should cite one passage title when possible.",
            "Conclusion must be only the answer span, not a full sentence.",
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
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/hotpot_qa/hf_dataset"
        super().__init__(split=split, max_samples=max_samples,
                         n_hop_values=n_hop_values, cache_dir=cache_dir)

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
        log.info("Loading HotpotQA split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        if not isinstance(ds, DatasetDict):
            raise ValueError("Expected DatasetDict with explicit train/validation/test splits for HotpotQA.")
        if split_name not in ds:
            raise ValueError(
                f"Split '{split_name}' not found. Available: {list(ds.keys())}"
            )
        ds = ds[split_name]

        items = list(ds)

        # HotpotQA is inherently 2-hop
        for item in items:
            item["n_hops"] = 2
            level_map = {"easy": 1, "medium": 2, "hard": 3}
            item["difficulty"] = level_map.get(item.get("level", "medium"), 2)

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d HotpotQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        # Build context from paragraphs
        context_parts = []
        titles = raw_item.get("context", {}).get("title", [])
        sentences = raw_item.get("context", {}).get("sentences", [])
        for title, sents in zip(titles, sentences):
            context_parts.append(f"[{title}] {''.join(sents)}")
        context_str, truncated = self.pack_context_blocks(
            context_parts,
            max_chars=self.context_char_budget,
        )
        if truncated:
            context_str += "\n\n[Note] Context truncated for length."

        question = raw_item["question"]
        user_msg = f"Context:\n{context_str}\n\nQuestion: {question}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        # Free-form: needs LLM judge for accurate evaluation
        gen_answer = self.parse_answer(generated).strip().lower()
        gt = ground_truth.strip().lower()
        if gen_answer == gt:
            return 1
        if gt in gen_answer or gen_answer in gt:
            return 1
        return -1  # Needs judge

    def parse_answer(self, generated_text):
        return parse_free_form_answer(generated_text)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", ""))
