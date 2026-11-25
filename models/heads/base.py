"""Abstract base class for uncertainty heads."""
import os, yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class UncertaintyHeadBase(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, features, attention_mask=None):
        raise NotImplementedError

    def pool_features(self, features, attention_mask=None):
        if attention_mask is None:
            return features.mean(dim=1)
        mask = attention_mask.unsqueeze(-1)
        if mask.dtype != features.dtype:
            mask = mask.to(features.dtype)
        summed = (features * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    @staticmethod
    def pack_claim_logits(logits_per_sample, num_classes, device):
        """Pad per-sample claim logits to uniform length and stack."""
        if not logits_per_sample:
            return torch.zeros(0, num_classes, device=device)
        max_claims = max(x.shape[0] for x in logits_per_sample)
        padded = [
            F.pad(x, (0, 0, 0, max_claims - x.shape[0]), value=-100.0)
            for x in logits_per_sample
        ]
        return torch.stack(padded, dim=0)

    def save(self, output_dir, config=None):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "head_weights.pth"))
        if config:
            with open(os.path.join(output_dir, "head_config.yaml"), "w") as f:
                yaml.dump(config, f)

    def load(self, path):
        weights_path = os.path.join(path, "head_weights.pth")
        if os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))
        elif os.path.isfile(path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"No head weights at {path}")
