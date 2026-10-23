"""Base class for multi-hop reasoning datasets."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from torch.utils.data import Dataset

from data.prompt_style import build_system_prompt


class BaseReasoningDataset(Dataset, ABC):
    """Abstract base for all multi-hop reasoning task datasets.

    Each dataset must implement:
    - build_chat_messages: Format a raw item into instruction-tuned chat messages
    - check_correctness: Compare generated answer to ground truth
    - parse_answer: Extract answer from generated text
    - get_difficulty: Return reasoning depth (n_hops)
    """

    name: str = ""
    task_type: str = "free_form"  # "classification" or "free_form"
    needs_judge: bool = True  # Whether LLM judge is needed for evaluation
    difficulty_field: str = "n_hops"  # Field name for reasoning depth

    # System prompt for instruction-tuned models - emphasizes step-by-step reasoning
    system_prompt: str = build_system_prompt(
        objective="Solve the question with minimal evidence-based reasoning.",
        conclusion_schema="[your final answer]",
        examples=[
            {
                "input": "Question: What color is the sky on a clear day?",
                "output": (
                    "Reasoning:\n"
                    "Step 1: On clear days, sunlight scattering makes the sky appear blue.\n"
                    "Step 2: No contrary condition is provided in the question.\n"
                    "Conclusion: blue."
                ),
            }
        ],
    )

    eval_config: Dict = {}

    def __init__(self, split: str = "train", max_samples: int = 0,
                 n_hop_values: Optional[List[int]] = None, cache_dir: Optional[str] = None):
        self.split = split
        self.max_samples = max_samples
        self.n_hop_values = n_hop_values
        self.cache_dir = cache_dir
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self) -> list:
        """Load and return dataset items."""
        raise NotImplementedError

    @abstractmethod
    def build_chat_messages(self, raw_item: dict) -> List[Dict[str, str]]:
        """Convert a raw dataset item into chat messages.
        Returns: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        raise NotImplementedError

    def build_prompt(self, raw_item: dict) -> str:
        """Build a single prompt string from chat messages."""
        messages = self.build_chat_messages(raw_item)
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @abstractmethod
    def check_correctness(self, generated: str, ground_truth: str) -> int:
        """Check if generated answer matches ground truth.
        Returns: 1 (correct), 0 (incorrect), -1 (needs judge)
        """
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, generated_text: str) -> str:
        """Extract the final answer from generated text."""
        raise NotImplementedError

    @abstractmethod
    def get_difficulty(self, raw_item: dict) -> int:
        """Return the reasoning depth / number of hops."""
        raise NotImplementedError

    def get_ground_truth(self, raw_item: dict) -> str:
        """Return ground truth answer."""
        return str(raw_item.get("answer", ""))

    @staticmethod
    def pack_context_blocks(blocks: List[str], max_chars: int = 8000, tail_chars: int = 1200):
        """Join context blocks within a character budget, preserving head and tail."""
        clean_blocks = [str(b).strip() for b in (blocks or []) if str(b).strip()]
        if not clean_blocks:
            return "", False

        full = "\n".join(clean_blocks)
        if len(full) <= max_chars:
            return full, False

        marker = "... [context truncated] ..."
        tail_budget = max(0, min(tail_chars, max_chars // 3))
        head_budget = max_chars - tail_budget - len(marker) - 2  # newlines
        if head_budget < 256:
            head_budget = max_chars - len(marker) - 2
            tail_budget = 0

        head_parts = []
        head_len = 0
        idx = 0
        while idx < len(clean_blocks):
            block = clean_blocks[idx]
            add_len = len(block) + (1 if head_parts else 0)
            if head_len + add_len > head_budget:
                break
            head_parts.append(block)
            head_len += add_len
            idx += 1
        if not head_parts:
            clipped = clean_blocks[0][: max_chars - 16] + "... [truncated]"
            return clipped, True

        tail_parts = []
        tail_len = 0
        j = len(clean_blocks) - 1
        while j >= idx:
            block = clean_blocks[j]
            add_len = len(block) + (1 if tail_parts else 0)
            if tail_len + add_len > tail_budget:
                break
            tail_parts.append(block)
            tail_len += add_len
            j -= 1
        tail_parts.reverse()

        merged_parts = head_parts
        if tail_parts:
            merged_parts = head_parts + [marker] + tail_parts
        merged = "\n".join(merged_parts)
        if len(merged) > max_chars:
            merged = merged[:max_chars]
        return merged, True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
