"""Dataset registry for multi-hop reasoning tasks."""

from data.hotpotqa import HotpotQADataset
from data.musique import MuSiQueDataset
from data.babi import BabiDataset
from data.iirc import IIRCDataset
from data.mathqa import MathQADataset
from data.scienceqa import ScienceQADataset
from data.spartqa import SpartQADataset
from data.stepgame import StepGameDataset
from data.strategyqa import StrategyQADataset
from data.truthfulqa import TruthfulQADataset
from data.twowiki import TwoWikiMultihopQADataset

DATASET_REGISTRY = {
    # Multi-hop reasoning (free-form, needs judge)
    "hotpotqa": HotpotQADataset,
    "hotpot_qa": HotpotQADataset,
    "musique": MuSiQueDataset,
    "MuSiQue": MuSiQueDataset,
    "babi": BabiDataset,
    "2wikimultihopqa": TwoWikiMultihopQADataset,
    "2WikiMultihopQA": TwoWikiMultihopQADataset,
    "2wiki": TwoWikiMultihopQADataset,
    "iirc": IIRCDataset,
    "IIRC": IIRCDataset,
    "strategyqa": StrategyQADataset,
    "StrategyQA": StrategyQADataset,
    "truthfulqa": TruthfulQADataset,
    "TruthfulQA": TruthfulQADataset,
    "truthful_qa": TruthfulQADataset,
    "mathqa": MathQADataset,
    "MathQA": MathQADataset,
    "math_qa": MathQADataset,
    "scienceqa": ScienceQADataset,
    "ScienceQA": ScienceQADataset,
    "science_qa": ScienceQADataset,
    # Spatial reasoning (classification, no judge)
    "spartqa": SpartQADataset,
    "SpartQA": SpartQADataset,
    "stepgame": StepGameDataset,
    "StepGame": StepGameDataset,
}


def get_dataset_cls(name: str):
    key = name.strip()
    if key not in DATASET_REGISTRY:
        key = key.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key]


def get_dataset(name: str, **kwargs):
    cls = get_dataset_cls(name)
    return cls(**kwargs)
