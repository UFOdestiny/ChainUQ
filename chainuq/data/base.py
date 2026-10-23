"""Base class of the benchmark loaders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

from chainuq import config
from chainuq.data.answers import extract_conclusion_line
from datasets import DatasetDict, load_from_disk


class ReasoningDataset:
    """One split of a prepared benchmark (``<root>/<dir>/hf_dataset``, a ``DatasetDict``).

    Subclasses define the system prompt, the user message, the reference answer,
    and the reasoning depth (``n_hops``) of a question.
    """

    name: str = ""
    dir_name: str = ""
    system_prompt: str = ""
    shuffle_subset = True  # with max_samples: random subset (seed 2026) instead of the first rows

    def __init__(self, split: str = "train", max_samples: int = 0, root: Optional[Path] = None):
        self.split = "validation" if split == "val" else split
        self.max_samples = max_samples
        self.path = Path(root or config.DATASETS_ROOT) / self.dir_name / "hf_dataset"
        self.data = self._load()

    def _load(self) -> List[dict]:
        dd = load_from_disk(str(self.path))
        if not isinstance(dd, DatasetDict) or self.split not in dd:
            raise ValueError(f"split {self.split!r} not found in {self.path}")
        items = [dict(x) for x in dd[self.split]]
        for item in items:
            self.prepare(item)
        if 0 < self.max_samples < len(items):
            if self.shuffle_subset:
                random.Random(config.SEED).shuffle(items)
            items = items[: self.max_samples]
        return items

    def prepare(self, item: dict) -> None:
        """Normalize one raw row in place (fields read by the prompt and ``n_hops``)."""

    def build_chat_messages(self, item: dict) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_message(item)}]

    def user_message(self, item: dict) -> str:
        raise NotImplementedError

    def parse_answer(self, generated_text: str) -> str:
        return extract_conclusion_line(generated_text)

    def check_correctness(self, generated: str, ground_truth: str) -> int:
        """Exact-match label; only defined for datasets whose labels do not come from the judge."""
        raise NotImplementedError(f"{self.name} is labeled by the judge")

    def get_difficulty(self, item: dict) -> int:
        return int(item.get("n_hops", 2))

    def get_ground_truth(self, item: dict) -> str:
        return str(item.get("answer", ""))

    @staticmethod
    def pack_context_blocks(blocks: List[str], max_chars: int = 8000, tail_chars: int = 1200):
        """Join context blocks within a character budget, keeping the head and the tail. Returns (text, truncated)."""
        clean_blocks = [str(b).strip() for b in (blocks or []) if str(b).strip()]
        if not clean_blocks:
            return "", False
        full = "\n".join(clean_blocks)
        if len(full) <= max_chars:
            return full, False

        marker = "... [context truncated] ..."
        tail_budget = max(0, min(tail_chars, max_chars // 3))
        head_budget = max_chars - tail_budget - len(marker) - 2
        if head_budget < 256:
            head_budget = max_chars - len(marker) - 2
            tail_budget = 0

        head_parts, head_len, idx = [], 0, 0
        while idx < len(clean_blocks):
            add_len = len(clean_blocks[idx]) + (1 if head_parts else 0)
            if head_len + add_len > head_budget:
                break
            head_parts.append(clean_blocks[idx])
            head_len += add_len
            idx += 1
        if not head_parts:
            return clean_blocks[0][: max_chars - 16] + "... [truncated]", True

        tail_parts, tail_len, j = [], 0, len(clean_blocks) - 1
        while j >= idx:
            add_len = len(clean_blocks[j]) + (1 if tail_parts else 0)
            if tail_len + add_len > tail_budget:
                break
            tail_parts.append(clean_blocks[j])
            tail_len += add_len
            j -= 1
        tail_parts.reverse()

        merged = "\n".join(head_parts + ([marker] + tail_parts if tail_parts else []))
        return merged[:max_chars], True

    def __len__(self):
        return len(self.data)
