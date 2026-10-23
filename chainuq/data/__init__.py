"""Benchmark loaders, prompts, and response parsing."""

from chainuq.data.babi import BAbI
from chainuq.data.hotpotqa import HotpotQA
from chainuq.data.math import GSM8K, MATH
from chainuq.data.musique import MuSiQue
from chainuq.data.strategyqa import StrategyQA

LOADERS = {cls.name: cls for cls in (HotpotQA, MuSiQue, StrategyQA, BAbI, GSM8K, MATH)}


def load_dataset(name: str, split: str, max_samples: int = 0, root=None):
    """One split of a prepared benchmark; ``root`` overrides ``CHAINUQ_DATASETS`` (e.g. adaptation pools)."""
    return LOADERS[name](split=split, max_samples=max_samples, root=root)
