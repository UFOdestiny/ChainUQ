"""Models, datasets, and method names used throughout the experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from chainuq import config


@dataclass(frozen=True)
class ModelSpec:
    name: str  # directory name under CHAINUQ_MODELS
    family: str
    chat_template_kwargs: dict = field(default_factory=dict)
    judge_request_kwargs: dict = field(default_factory=dict)  # extra request fields when served as a judge

    @property
    def path(self) -> Path:
        return config.MODELS_ROOT / self.name


BACKBONES = {
    "llama": ModelSpec("Llama-3.1-8B-Instruct", "llama"),
    "qwen8b": ModelSpec("Qwen3-8B", "qwen", chat_template_kwargs={"enable_thinking": False}),
    "ministral": ModelSpec("Ministral-8B-Instruct-2410", "mistral"),
}

JUDGES = {
    # Main judge, from a different family than every backbone.
    "gptoss": ModelSpec("gpt-oss-20b", "openai", judge_request_kwargs={"reasoning_effort": "low"}),
    # Weaker step verifier (verifier swap) and weak pool verifier (adaptation).
    "qwen1b": ModelSpec("Qwen3-1.7B", "qwen", chat_template_kwargs={"enable_thinking": False}),
    # Second correctness labeler (label robustness) and mid-strength pool verifier (adaptation).
    "qwen4b": ModelSpec("Qwen3-4B-Instruct-2507", "qwen"),
}
for _key, _spec in BACKBONES.items():  # each backbone verifying its own steps
    JUDGES[f"self-{_key}"] = _spec

MAIN_JUDGE = "gptoss"
SECOND_LABEL_JUDGE = "qwen4b"
RM_MODEL = ModelSpec("Skywork-Reward-V2-Llama-3.1-8B", "llama")
NLI_MODEL = ModelSpec("deberta-large-mnli", "deberta")

# label: "judge" = correctness from the main judge, "em" = exact/answer matching (no judge involved).
DATASETS = {
    "hotpotqa": {"dir": "hotpot_qa", "role": "id", "label": "judge"},
    "musique": {"dir": "MuSiQue", "role": "ood", "label": "judge"},
    "strategyqa": {"dir": "StrategyQA", "role": "ood", "label": "em"},
    "babi": {"dir": "babi", "role": "ood", "label": "judge"},
    "gsm8k": {"dir": "gsm8k", "role": "ood", "label": "em"},
    "math500": {"dir": "math500", "role": "ood", "label": "em"},
}
SOURCE_DATASET = "hotpotqa"  # heads are trained on this dataset only
OOD_DATASETS = [d for d, v in DATASETS.items() if v["role"] == "ood"]

# Methods of the main comparison: (group, [(key, tier, display name)]).
# Tier A uses no external model at inference; tier B may query a judge or reward model.
BASELINE_GROUPS = [
    ("Single-response, unsupervised", [("mcp", "tier_a", "MCP"), ("ppl", "tier_a", "Perplexity"),
                                       ("mte", "tier_a", "MTE"), ("ccp", "tier_a", "CCP")]),
    ("Extra backbone calls", [("cotuq", "tier_a", "CoT-UQ"), ("se", "tier_a", "Semantic Entropy"),
                              ("selfcheck", "tier_a", "SelfCheckGPT")]),
    ("Supervised probes", [("saplma", "tier_a", "SAPLMA"), ("factoscope", "tier_a", "Factoscope"),
                           ("lookback", "tier_a", "Lookback Lens"), ("uhead", "tier_a", "UHead"),
                           ("meanpool-mlp", "tier_a", "MeanPool-MLP"), ("attnpool-mlp", "tier_a", "AttnPool-MLP")]),
    ("External verifier", [("judge-step", "tier_b", "Judge-Step"), ("judge-final", "tier_b", "Judge-Final"),
                           ("rm", "tier_b", "Reward Model")]),
]
TIER_A_BASELINES = [k for _, ms in BASELINE_GROUPS for k, t, _ in ms if t == "tier_a"]
TIER_B_BASELINES = [k for _, ms in BASELINE_GROUPS for k, t, _ in ms if t == "tier_b"]
TRAINED_BASELINES = ["saplma", "factoscope", "lookback", "uhead", "meanpool-mlp", "attnpool-mlp"]
SCORED_BASELINES = ["mcp", "ppl", "mte", "ccp", "cotuq", "se", "selfcheck", "judge-step", "judge-final", "rm"]

FINAL_HEAD = "chainuq"  # Stage-1 head of ChainUQ (trained with privileged consistency supervision)
OURS = "chainuq"  # full ChainUQ: head + R-Log on all evidence, including the judge's step labels and verdict
OURS_A = "chainuq-a"  # judge-free ChainUQ-A: no external model at inference
