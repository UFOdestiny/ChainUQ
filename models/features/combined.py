"""Combine multiple feature extractors by concatenation."""
import torch
import torch.nn as nn
from typing import List

class CombinedExtractor(nn.Module):
    def __init__(self, extractors: List[nn.Module]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def feature_dim(self) -> int:
        return sum(e.feature_dim() for e in self.extractors)

    def output_attention(self) -> bool:
        return any(getattr(ext, "output_attention", lambda: False)() for ext in self.extractors)

    def forward(self, hidden_states=None, logits=None, attentions=None, attention_mask=None):
        from models.features.hidden_states import HiddenStateExtractor
        from models.features.token_probs import TokenProbExtractor
        from models.features.attention import AttentionExtractor
        features = []
        for ext in self.extractors:
            if isinstance(ext, HiddenStateExtractor) and hidden_states is not None:
                features.append(ext(hidden_states))
            elif isinstance(ext, TokenProbExtractor) and logits is not None:
                features.append(ext(logits))
            elif isinstance(ext, AttentionExtractor) and attentions is not None:
                features.append(ext(attentions, attention_mask))
        if not features:
            raise ValueError("No features extracted.")
        return torch.cat(features, dim=-1)
