"""Utilities for resolving dataset splits with deterministic fallback."""

from __future__ import annotations

from typing import Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets


def resolve_split_name(split: str) -> str:
    key = (split or "train").strip().lower()
    if key in ("validation", "val", "dev"):
        return "validation"
    if key == "test":
        return "test"
    return "train"


def ensure_train_validation_test(
    dataset_obj: Dataset | DatasetDict,
    seed: int,
) -> Tuple[DatasetDict, bool]:
    """Return DatasetDict(train/validation/test), fallback to deterministic 8:1:1 if needed.

    Returns:
        (dataset_dict, used_fallback_split)
    """
    if isinstance(dataset_obj, DatasetDict):
        train_key = "train" if "train" in dataset_obj else None
        val_key = next((k for k in ("validation", "val", "dev") if k in dataset_obj), None)
        test_key = "test" if "test" in dataset_obj else None
        if train_key and val_key and test_key:
            return DatasetDict(
                train=dataset_obj[train_key],
                validation=dataset_obj[val_key],
                test=dataset_obj[test_key],
            ), False

        parts = [dataset_obj[k] for k in dataset_obj.keys()]
        merged = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    else:
        merged = dataset_obj

    split_80_20 = merged.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    split_10_10 = split_80_20["test"].train_test_split(test_size=0.5, seed=seed, shuffle=True)
    return DatasetDict(
        train=split_80_20["train"],
        validation=split_10_10["train"],
        test=split_10_10["test"],
    ), True
