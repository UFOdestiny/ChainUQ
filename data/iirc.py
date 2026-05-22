"""IIRC dataset with article-level questions flattened into QA items."""

import random
import re

from datasets import DatasetDict, load_from_disk

from config import DATASETS_ROOT, GLOBAL_SEED
from data.answer_parsing import parse_free_form_answer
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)

UNANSWERABLE_TEXT = "cannot be determined from the provided context"


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'")


def _normalize_yes_no(text: str) -> str:
    norm = _normalize_text(text)
    if norm in {"yes", "true"}:
        return "yes"
    if norm in {"no", "false"}:
        return "no"
    return norm


class IIRCDataset(BaseReasoningDataset):
    name = "iirc"
    task_type = "free_form"
    needs_judge = True
    difficulty_field = "n_hops"
    context_char_budget = 9000

    system_prompt = build_system_prompt(
        objective="Answer using only the provided article and linked passages.",
        conclusion_schema="[final short answer]",
        examples=[
            {
                "input": (
                    "Context:\n"
                    "[Main] The article states no release date is given.\n"
                    "Question: What year was it released?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The provided context does not mention any release year.\n"
                    "Step 2: No linked passage adds that missing year.\n"
                    "Conclusion: cannot be determined from the provided context."
                ),
            }
        ],
        extra_rules=[
            "If evidence is insufficient, output exactly: cannot be determined from the provided context.",
            "For span/value answers, Conclusion must be only the answer span or value.",
        ],
    )

    eval_config = {
        "split": "test",
        "n_hop_values": None,
        "ood_split": "test",
        "ood_max_samples": 0,
    }

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/IIRC/hf_dataset"
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
    def _answer_to_text(answer: dict) -> str:
        if not isinstance(answer, dict):
            return str(answer or "")
        answer_type = answer.get("type")
        if answer_type == "span":
            spans = []
            for span in answer.get("answer_spans") or []:
                text = str(span.get("text", "")).strip()
                if text:
                    spans.append(text)
            deduped = list(dict.fromkeys(spans))
            return ", ".join(deduped)
        if answer_type == "value":
            value = str(answer.get("answer_value") or "").strip()
            unit = str(answer.get("answer_unit") or "").strip()
            return f"{value} {unit}".strip()
        if answer_type == "binary":
            return _normalize_yes_no(str(answer.get("answer_value") or ""))
        if answer_type == "none":
            return UNANSWERABLE_TEXT
        return ""

    def _load_data(self):
        split_name = self._resolve_split(self.split)
        log.info("Loading IIRC split=%s from %s", split_name, self.dataset_path)

        ds = load_from_disk(self.dataset_path)
        if isinstance(ds, DatasetDict):
            if split_name not in ds:
                raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
            ds = ds[split_name]

        items = []
        for article in ds:
            article_title = str(article.get("title", "")).strip()
            main_text = str(article.get("text", "")).strip()
            for question in article.get("questions") or []:
                question_text = str(question.get("question", "")).strip()
                if not question_text:
                    continue
                contexts = question.get("context") or []
                question_links = question.get("question_links") or []
                n_hops = max(2, min(4, max(len(contexts), len(question_links), 1)))
                items.append(
                    {
                        "qid": question.get("qid", ""),
                        "question": question_text,
                        "article_title": article_title,
                        "main_passage": main_text,
                        "contexts": contexts,
                        "question_links": question_links,
                        "raw_answer": question.get("answer") or {},
                        "answer": self._answer_to_text(question.get("answer") or {}),
                        "n_hops": n_hops,
                        "difficulty": n_hops,
                    }
                )

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d IIRC items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item):
        context_parts = []
        seen_blocks = set()

        main_title = raw_item.get("article_title", "")
        main_text = raw_item.get("main_passage", "")
        if main_text:
            context_parts.append(f"[{main_title}] {main_text}")
            seen_blocks.add((main_title, main_text))

        for block in raw_item.get("contexts") or []:
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            passage = str(block.get("passage", "")).strip() or main_title or "linked passage"
            label = main_title if passage == "main" and main_title else passage
            key = (label, text)
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
            context_parts.append(f"[{label}] {text}")

        context_str, truncated = self.pack_context_blocks(
            context_parts,
            max_chars=self.context_char_budget,
        )
        if truncated:
            context_str += "\n\n[Note] Context truncated for length."

        user_msg = (
            f"Context:\n{context_str}\n\nQuestion: {raw_item.get('question', '')}\n\n"
            f"If the context is insufficient, answer: {UNANSWERABLE_TEXT}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated, ground_truth):
        gen = self.parse_answer(generated)
        gen_norm = _normalize_text(gen)
        gt_norm = _normalize_text(ground_truth)

        if not gen_norm:
            return 0
        if gt_norm == _normalize_text(UNANSWERABLE_TEXT):
            for phrase in (
                "cannot be determined",
                "not enough information",
                "insufficient information",
                "cannot determine",
                "unknown",
            ):
                if phrase in gen_norm:
                    return 1
            return 0

        if gt_norm in {"yes", "no"}:
            return 1 if _normalize_yes_no(gen_norm) == gt_norm else 0

        if gen_norm == gt_norm or (gt_norm and (gt_norm in gen_norm or gen_norm in gt_norm)):
            return 1

        gen_nums = re.findall(r"-?\d+(?:\.\d+)?", gen_norm)
        gt_nums = re.findall(r"-?\d+(?:\.\d+)?", gt_norm)
        if gen_nums and gt_nums and gen_nums[0] == gt_nums[0]:
            return 1
        return -1

    def parse_answer(self, generated_text):
        return parse_free_form_answer(generated_text)

    def get_difficulty(self, raw_item):
        return raw_item.get("n_hops", 2)

    def get_ground_truth(self, raw_item):
        return str(raw_item.get("answer", ""))
