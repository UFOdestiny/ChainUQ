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
    # The same model with thinking enabled (long reasoning traces); the thinking becomes context of the response.
    "qwen8b-think": ModelSpec("Qwen3-8B", "qwen", chat_template_kwargs={"enable_thinking": True}),
    "ministral": ModelSpec("Ministral-8B-Instruct-2410", "mistral"),
}

JUDGES = {
    # Main judge, from a different family than every backbone. Its answer-blind outputs supervise PCS.
    "gptoss": ModelSpec("gpt-oss-20b", "openai", judge_request_kwargs={"reasoning_effort": "low"}),
    # The same judge shown the thinking trace of thinking-mode backbones as well as the visible response.
    "gptoss-trace": ModelSpec("gpt-oss-20b", "openai", judge_request_kwargs={"reasoning_effort": "low"}),
    # Judge swap at inference (PCS keeps the main judge): a stronger judge, at low and high reasoning effort,
    # and a judge from another model family (Qwen3-8B in thinking mode).
    "gptoss120b": ModelSpec("gpt-oss-120b", "openai", judge_request_kwargs={"reasoning_effort": "low"}),
    "gptoss120b-high": ModelSpec("gpt-oss-120b", "openai", judge_request_kwargs={"reasoning_effort": "high"}),
    "qwen3judge": ModelSpec("Qwen3-8B", "qwen", chat_template_kwargs={"enable_thinking": True}),
    # Black-box API judge (Anthropic Messages API, no log-probabilities): chainuq.pipeline.judge_api.
    "haiku": ModelSpec("claude-haiku-5-5", "anthropic"),
    # Second correctness labeler (label robustness).
    "qwen4b": ModelSpec("Qwen3-4B-Instruct-2507", "qwen"),
}

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
SOURCE_DATASET = "hotpotqa"  # the only dataset with correctness labels in head training
OOD_DATASETS = [d for d, v in DATASETS.items() if v["role"] == "ood"]
POOL_DATASETS = ["musique", "babi", "gsm8k", "math500", "strategyqa"]  # unlabeled train pools, in training order

# Methods of the main comparison: (group, [(key, tier, display name)]).
# Tier A uses no external model at inference; tier B may query a judge or reward model.
BASELINE_GROUPS = [
    ("Single-response, unsupervised", [("mcp", "tier_a", "MCP"), ("ppl", "tier_a", "Perplexity"), ("mte", "tier_a", "MTE"), ("ccp", "tier_a", "CCP")]),
    ("Extra backbone calls", [("cotuq", "tier_a", "CoT-UQ"), ("se", "tier_a", "Semantic Entropy"), ("selfcheck", "tier_a", "SelfCheckGPT")]),
    (
        "Supervised probes",
        [
            ("saplma", "tier_a", "SAPLMA"),
            ("factoscope", "tier_a", "Factoscope (adapted)"),
            ("lookback", "tier_a", "Lookback Lens"),
            ("uhead", "tier_a", "UHead"),
            ("meanpool-mlp", "tier_a", "MeanPool-MLP"),
            ("attnpool-mlp", "tier_a", "AttnPool-MLP"),
        ],
    ),
    ("External verifier", [("judge-step", "tier_b", "Judge-Step"), ("judge-final", "tier_b", "Judge-Final"), ("rm", "tier_b", "Reward Model")]),
]
TIER_A_BASELINES = [k for _, ms in BASELINE_GROUPS for k, t, _ in ms if t == "tier_a"]
TIER_B_BASELINES = [k for _, ms in BASELINE_GROUPS for k, t, _ in ms if t == "tier_b"]
SCORED_BASELINES = ["mcp", "ppl", "mte", "ccp", "cotuq", "se", "selfcheck", "judge-step", "judge-final", "rm"]

FINAL_HEAD = "chainuq"  # Stage-1 head: PCS on the labeled source and the unlabeled pools (chainuq.heads.HEADS)
SOURCE_HEAD = "chainuq-source"  # the same head trained with PCS on the labeled source only
EXTRA_PROBES = ("saplma",)  # further internal-state probes whose scores join the evidence
OURS = "chainuq"  # ChainUQ: head + R-Log on the evidence and one answer-blind judge call
OURS_A = "chainuq-a"  # judge-free ChainUQ-A: no external model at inference
