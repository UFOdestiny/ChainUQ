"""MuSiQue multi-hop sequential question answering dataset."""
import random

from datasets import load_from_disk, DatasetDict

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_free_form_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


class MuSiQueDataset(BaseReasoningDataset):
    name = "musique"
    task_type = "free_form"
    needs_judge = True
    difficulty_field = "n_hops"
    context_char_budget = 8000

    system_prompt = build_system_prompt(
        objective="Solve the question by chaining sub-answers from the provided passages.",
        conclusion_schema="[final short answer]",
        examples=[
            {
                "input": (
                    "Context:\n"
                    "[Film] Interstellar was directed by Christopher Nolan.\n"
                    "[Person] Christopher Nolan was born in London.\n"
                    "Question: In which city was the director of Interstellar born?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The director of Interstellar is Christopher Nolan.\n"
                    "Step 2: Christopher Nolan was born in London.\n"
                    "Conclusion: London."
                ),
            }
        ],
        extra_rules=[
            "Answer sub-questions in order; each Step should carry one intermediate answer.",
            "Conclusion must be only the final short answer, not a full sentence.",
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
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/MuSiQue/hf_dataset"
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
        log.info("Loading MuSiQue split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        if not isinstance(ds, DatasetDict):
            raise ValueError("Expected DatasetDict with explicit train/validation/test splits for MuSiQue.")
        if split_name not in ds:
            raise ValueError(
                f"Split '{split_name}' not found. Available: {list(ds.keys())}"
            )
        ds = ds[split_name]

        items = list(ds)
        for item in items:
            decomp = item.get("question_decomposition", item.get("decomposition", []))
            if isinstance(decomp, list):
                item["n_hops"] = max(len(decomp), 2)
            else:
                item["n_hops"] = 2
            item["decomposition"] = decomp

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d MuSiQue items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        question = raw_item.get("question", "")
        # Build context from paragraphs
        paragraphs = raw_item.get("paragraphs", [])
        if paragraphs and isinstance(paragraphs, list):
            context_parts = []
            for p in paragraphs:
                if isinstance(p, dict):
                    title = p.get("title", "")
                    text = p.get("paragraph_text", p.get("text", ""))
                    context_parts.append(f"[{title}] {text}")
                elif isinstance(p, str):
                    context_parts.append(p)
            context_str, truncated = self.pack_context_blocks(
                context_parts,
                max_chars=self.context_char_budget,
            )
            if truncated:
                context_str += "\n\n[Note] Context truncated for length."
            user_msg = f"Context:\n{context_str}\n\nQuestion: {question}"
        else:
            user_msg = f"Question: {question}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        gen = self.parse_answer(generated).strip().lower()
        gt = ground_truth.strip().lower()
        if gen == gt:
            return 1
        if gt in gen or gen in gt:
            return 1
        return -1

    def parse_answer(self, generated_text):
        return parse_free_form_answer(generated_text)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", raw_item.get("predicted_answer", "")))
