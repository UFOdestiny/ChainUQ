"""StepGame spatial reasoning dataset.

StepGame (ZhengyanShi/StepGame) is a 9-class classification task for spatial
direction prediction. Columns: story, question, label, k_hop.
Classification task — no LLM judge needed.
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
from datasets import (
    Dataset as HFDataset,
    DatasetDict,
    DatasetInfo,
    Features,
    load_from_disk,
)

from config import DATASETS_ROOT, GLOBAL_SEED
from data.base import BaseReasoningDataset
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)

LABELS = [
    "upper-right", "lower-right", "upper-left", "lower-left",
    "above", "left", "below", "right", "overlap",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
LABEL_LIST_TEXT = ", ".join(LABELS)

LABEL_ALIASES = {
    "upper-right": [
        r"top[-\s]?right", r"up[-\s]?right", r"above[-\s]?right",
        r"right[-\s]?above", r"right[-\s]?up", r"north[-\s]?east",
    ],
    "lower-right": [
        r"bottom[-\s]?right", r"down[-\s]?right", r"below[-\s]?right",
        r"right[-\s]?below", r"right[-\s]?down", r"south[-\s]?east",
    ],
    "upper-left": [
        r"top[-\s]?left", r"up[-\s]?left", r"above[-\s]?left",
        r"left[-\s]?above", r"left[-\s]?up", r"north[-\s]?west",
    ],
    "lower-left": [
        r"bottom[-\s]?left", r"down[-\s]?left", r"below[-\s]?left",
        r"left[-\s]?below", r"left[-\s]?down", r"south[-\s]?west",
    ],
    "above": [r"up", r"top", r"north"],
    "below": [r"down", r"bottom", r"south"],
    "right": [r"east"],
    "left": [r"west"],
    "overlap": [r"same[-\s]?place", r"same[-\s]?position", r"same[-\s]?location", r"overlapping"],
}

def _resolve_split_dir(dataset_path: str, split: str) -> Path:
    root = Path(dataset_path)
    split_dir = root / split
    if split_dir.is_dir():
        return split_dir
    if root.name == split and root.is_dir():
        return root
    raise ValueError(f"Cannot resolve split dir for '{split}' from '{dataset_path}'")


def _read_arrow_table(arrow_path: Path) -> pa.Table:
    with pa.memory_map(str(arrow_path), "r") as source:
        try:
            reader = pa.ipc.RecordBatchFileReader(source)
        except pa.ArrowInvalid:
            reader = pa.ipc.RecordBatchStreamReader(source)
        table = reader.read_all()
    return table.replace_schema_metadata(None)


def _load_split_from_arrow(dataset_path: str, split: str) -> HFDataset:
    split_dir = _resolve_split_dir(dataset_path, split)
    data_files = sorted(split_dir.glob("data-*.arrow"))
    if not data_files:
        raise FileNotFoundError(f"No data-*.arrow files in {split_dir}")
    tables = [_read_arrow_table(p) for p in data_files]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    features = Features.from_arrow_schema(table.schema)
    return HFDataset(
        arrow_table=table, info=DatasetInfo(features=features), split=split
    )


def _load_compat(dataset_path: str, split: str):
    """Load with fallback for 'List' metadata mismatch in older dataset artifacts."""
    try:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(
                    f"Split '{split}' not found. Available: {list(ds.keys())}"
                )
            return ds[split]
        return ds
    except ValueError as exc:
        if "Feature type 'List' not found" not in str(exc):
            raise
        log.warning(
            "HF metadata mismatch at %s; using arrow fallback for '%s'.",
            dataset_path, split,
        )
        return _load_split_from_arrow(dataset_path, split)

class StepGameDataset(BaseReasoningDataset):
    """StepGame spatial reasoning — 9-class direction classification."""

    name = "stepgame"
    task_type = "classification"
    needs_judge = False
    difficulty_field = "n_hops"

    system_prompt = build_system_prompt(
        objective="Predict the spatial relation label from the story and question.",
        conclusion_schema=f"The answer is <one label from [{LABEL_LIST_TEXT}]>.",
        max_steps=9,
        max_step_words=12,
        examples=[
            {
                "input": (
                    "Context: The tree is above the house. The cat is right of the tree.\n"
                    "Question: Where is the cat relative to the house?"
                ),
                "output": (
                    "Reasoning:\n"
                    "Step 1: The tree is above the house.\n"
                    "Step 2: The cat is right of the tree.\n"
                    "Step 3: Right plus above gives upper-right from house.\n"
                    "Conclusion: The answer is upper-right."
                ),
            }
        ],
        extra_rules=[
            "Maintain the net vertical and horizontal relation to the queried object.",
            "Conclusion must be exactly: The answer is <allowed label>.",
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
        self.dataset_path = dataset_path or f"{DATASETS_ROOT}/StepGame/hf_dataset"
        super().__init__(split=split, max_samples=max_samples,
                         n_hop_values=n_hop_values, cache_dir=cache_dir)

    def _load_data(self):
        log.info("Loading StepGame split=%s from %s", self.split, self.dataset_path)
        ds = _load_compat(self.dataset_path, self.split)

        items = list(ds)
        for item in items:
            item["n_hops"] = item.get("k_hop", 1)

        if self.n_hop_values:
            items = [it for it in items if it["n_hops"] in self.n_hop_values]

        if self.max_samples > 0 and len(items) > self.max_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(items)
            items = items[:self.max_samples]

        log.info("Loaded %d StepGame items (split=%s)", len(items), self.split)
        return items

    def build_chat_messages(self, raw_item: dict) -> List[Dict[str, str]]:
        story = raw_item["story"]
        context = " ".join(story) if isinstance(story, list) else story
        question = raw_item["question"]
        user_msg = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Allowed labels: [{LABEL_LIST_TEXT}]"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def check_correctness(self, generated: str, ground_truth: str) -> int:
        predicted = self.parse_answer(generated)
        return 1 if predicted == ground_truth else 0

    def parse_answer(self, generated_text: str) -> str:
        text = (generated_text or "").strip().lower()
        if not text:
            return "unknown"

        line_candidates = []
        for pattern in (
            r"conclusion\s*:\s*([^\n\r]+)",
            r"the answer is\s*([^\n\r]+)",
            r"final answer\s*[:\-]?\s*([^\n\r]+)",
        ):
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                line_candidates.append(matches[-1])
        line_candidates.append(text)

        # Match longer labels first to avoid substring collisions
        labels_by_len = sorted(LABELS, key=len, reverse=True)

        def _normalize(x: str) -> str:
            return re.sub(r"[\s_]+", "-", x.lower())

        def _find_label(x: str) -> Optional[str]:
            best_label, best_pos = None, -1
            for label in labels_by_len:
                pat = rf"(?<![a-z-]){re.escape(label)}(?![a-z-])"
                for m in re.finditer(pat, x):
                    if m.start() > best_pos:
                        best_pos = m.start()
                        best_label = label
            if best_label is not None:
                return best_label
            for canonical, alias_patterns in LABEL_ALIASES.items():
                for alias_pat in alias_patterns:
                    if re.search(rf"(?<![a-z-])(?:{alias_pat})(?![a-z-])", x):
                        return canonical
            return best_label

        for candidate in line_candidates:
            picked = _find_label(_normalize(candidate))
            if picked is not None:
                return picked
        return "unknown"

    def get_difficulty(self, raw_item: dict) -> int:
        return raw_item.get("n_hops", raw_item.get("k_hop", 1))

    def get_ground_truth(self, raw_item: dict) -> str:
        return str(raw_item.get("label", ""))
