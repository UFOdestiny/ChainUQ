"""Model / dataset registry and artifact layout for the IJCAI re-run.

Every artifact path is derived here so that stages never guess each other's
file names. Override the roots with CUQ_MODELS_ROOT / CUQ_DATASETS_ROOT /
CUQ_ARTIFACTS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path(os.environ.get("CUQ_MODELS_ROOT", "/u/kytrio/proj/ChainUQ/models"))
DATASETS_ROOT = Path(os.environ.get("CUQ_DATASETS_ROOT", "/u/kytrio/proj/ChainUQ/datasets"))
ARTIFACTS = Path(os.environ.get("CUQ_ARTIFACTS", "/u/kytrio/nvme_bduw/chainuq_artifacts"))

SEED = 2026
HEAD_SEEDS = list(range(2026, 2036))  # 10 seeds for every trained head
FINAL_HEAD = "chainuq2pcs"  # Stage-1 head of the final ChainUQ (used by evaluate / paper scripts)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    chat_template_kwargs: dict = field(default_factory=dict)
    # Extra kwargs for the OpenAI-compatible server when used as a judge.
    judge_request_kwargs: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return MODELS_ROOT / self.name


BACKBONES = {
    "llama": ModelSpec("Llama-3.1-8B-Instruct", "llama"),
    "qwen8b": ModelSpec("Qwen3-8B", "qwen", chat_template_kwargs={"enable_thinking": False}),
    "ministral": ModelSpec("Ministral-8B-Instruct-2410", "mistral"),
}

JUDGES = {
    # Main judge: different family from all three backbones.
    "gptoss": ModelSpec("gpt-oss-20b", "openai", judge_request_kwargs={"reasoning_effort": "low"}),
    # Weak judge for the E3 judge-strength comparison.
    "qwen1b": ModelSpec("Qwen3-1.7B", "qwen", chat_template_kwargs={"enable_thinking": False}),
    # Second response-label judge (label robustness on test; not a backbone).
    "qwen4b": ModelSpec("Qwen3-4B-Instruct-2507", "qwen"),
}
# Self-judges reuse the backbone specs ("self:<backbone>").
for _key, _spec in BACKBONES.items():
    JUDGES[f"self-{_key}"] = _spec

MAIN_JUDGE = "gptoss"
RM_MODEL = ModelSpec("Skywork-Reward-V2-Llama-3.1-8B", "llama")
NLI_MODEL = ModelSpec("deberta-large-mnli", "deberta")

# Dataset roles: HotpotQA is the source (ID) benchmark; the rest are OOD.
DATASETS = {
    "hotpotqa": {"loader": "hotpotqa", "role": "id", "label": "judge"},
    "musique": {"loader": "musique", "role": "ood", "label": "judge"},
    "strategyqa": {"loader": "strategyqa", "role": "ood", "label": "em"},
    "babi": {"loader": "babi", "role": "ood", "label": "judge"},
    # Math reasoning OOD: labels by answer matching, no judge involved.
    "gsm8k": {"loader": "gsm8k", "role": "ood", "label": "em"},
    "math500": {"loader": "math500", "role": "ood", "label": "em"},
}
SOURCE_DATASET = "hotpotqa"

FORMATS = ("step", "numbered", "bullet", "free")


def gen_path(backbone: str, dataset: str, split: str, fmt: str = "step", tag: str = "") -> Path:
    suffix = f".{tag}" if tag else ""
    return ARTIFACTS / "gen" / backbone / dataset / fmt / f"{split}{suffix}.jsonl"


def judge_path(judge: str, task: str, backbone: str, dataset: str, split: str, fmt: str = "step") -> Path:
    return ARTIFACTS / "judge" / judge / task / backbone / dataset / fmt / f"{split}.jsonl"


def feat_dir(backbone: str, dataset: str, split: str, fmt: str = "step") -> Path:
    return ARTIFACTS / "feat" / backbone / dataset / fmt / split


def pred_path(method: str, backbone: str, seed: int, dataset: str, split: str, fmt: str = "step") -> Path:
    return ARTIFACTS / "preds" / method / backbone / f"seed{seed}" / dataset / fmt / f"{split}.jsonl"


def ckpt_dir(method: str, backbone: str, seed: int) -> Path:
    return ARTIFACTS / "ckpt" / method / backbone / f"seed{seed}"


def results_dir() -> Path:
    return ARTIFACTS / "results"
