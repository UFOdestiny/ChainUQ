"""ScienceQA multiple-choice reasoning dataset."""

import random
from datasets import load_from_disk

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_choice_index
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from data.split_utils import ensure_train_validation_test, resolve_split_name
from utils.log import get_logger

log = get_logger(__name__)


class ScienceQADataset(BaseReasoningDataset):
    name = "scienceqa"
    task_type = "classification"
    needs_judge = False
    difficulty_field = "n_hops"
    context_char_budget = 5000

    system_prompt = build_system_prompt(
        objective="Use the provided science context to select the best option.",
        conclusion_schema="The answer is <option number>.",
        examples=[
            {
                "input": (
                    "Context:\n[Hint] Plants use sunlight to make food.\n"
                    "Question: Which process uses sunlight to make food?\n"
                    "Options:\n1. Respiration\n2. Photosynthesis\n3. Digestion\n4. Erosion"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The hint says plants make food using sunlight.\n"
                    "Step 2: That process is photosynthesis, which is option 2.\n"
                    "Conclusion: The answer is 2."
                ),
            }
        ],
        extra_rules=[
            "Use the hint or lecture first, then match the concept to an option.",
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
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/ScienceQA/hf_dataset"
        super().__init__(
            split=split,
            max_samples=max_samples,
            n_hop_values=n_hop_values,
            cache_dir=cache_dir,
        )

    def _load_data(self):
        split_name = resolve_split_name(self.split)
        log.info("Loading ScienceQA split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        ds, used_fallback = ensure_train_validation_test(ds, seed=GLOBAL_SEED)
        if used_fallback:
            log.info("ScienceQA missing full train/validation/test; using deterministic 8:1:1 split.")
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
        split_ds = ds[split_name]

        items = []
        for raw in split_ds:
            choices = raw.get("choices") or []
            answer = raw.get("answer")
            if not isinstance(choices, list) or not choices:
                continue
            if not isinstance(answer, int) or not (0 <= answer < len(choices)):
                continue
            raw = dict(raw)
            raw["n_hops"] = 1
            raw["difficulty"] = 1
            items.append(raw)

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d ScienceQA items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        context_parts = []
        hint = str(raw_item.get("hint", "")).strip()
        lecture = str(raw_item.get("lecture", "")).strip()
        if hint:
            context_parts.append(f"[Hint] {hint}")
        if lecture:
            context_parts.append(f"[Lecture] {lecture}")
        image_field = raw_item.get("image")
        if image_field:
            context_parts.append("[Image] Image is provided in dataset, but text-only model cannot inspect pixels.")

        context_text, truncated = self.pack_context_blocks(context_parts, max_chars=self.context_char_budget)
        if truncated:
            context_text += "\n\n[Note] Context truncated for length."

        choices = raw_item.get("choices") or []
        options_text = "\n".join(f"{idx + 1}. {str(choice).strip()}" for idx, choice in enumerate(choices))
        user_msg = (
            f"Context:\n{context_text or '[No extra context]'}\n\n"
            f"Question: {raw_item.get('question', '')}\n"
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
        return parse_choice_index(generated_text, num_options=9, allow_letters=False)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 1)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", ""))
