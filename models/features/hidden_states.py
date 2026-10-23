"""Extract hidden-state features from selected LLM layers."""

from typing import List, Optional

import torch
import torch.nn as nn


class HiddenStateExtractor(nn.Module):
    def __init__(
        self,
        layer_nums: Optional[List[int]] = None,
        hidden_size: int = 4096,
        fusion: str = "concat",
        layer_weights: Optional[List[float]] = None,
        num_hidden_layers: Optional[int] = None,
    ):
        super().__init__()
        # ``None`` means "all transformer layers" (excluding embedding output).
        self.layer_nums = layer_nums
        self.hidden_size = int(hidden_size)
        self.fusion = str(fusion or "concat").strip().lower()
        if self.fusion not in {"concat", "weighted_sum"}:
            raise ValueError(f"Unsupported hidden-state fusion: {fusion}")
        self.layer_weights = [float(w) for w in layer_weights] if layer_weights else None
        self.num_hidden_layers = int(num_hidden_layers) if num_hidden_layers else None

    def _resolve_layer_indices(self, n_layers: int) -> List[int]:
        if self.layer_nums is None:
            return list(range(1, n_layers + 1))

        resolved_indices: List[int] = []
        for layer_idx in self.layer_nums:
            if layer_idx < 0:
                resolved = n_layers + 1 + layer_idx
            else:
                resolved = layer_idx + 1
            resolved = max(0, min(resolved, n_layers))
            resolved_indices.append(resolved)
        return resolved_indices

    def _selected_layer_count(self) -> int:
        if self.layer_nums is not None:
            return len(self.layer_nums)
        if self.num_hidden_layers is not None and self.num_hidden_layers > 0:
            return int(self.num_hidden_layers)
        # Fallback for unknown configs; ``forward`` will still infer from runtime tensors.
        return 1

    def feature_dim(self) -> int:
        if self.fusion == "weighted_sum":
            return self.hidden_size
        return self.hidden_size * self._selected_layer_count()

    def _normalized_weights(self, n_selected: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.layer_weights is None or len(self.layer_weights) != n_selected:
            weights = torch.ones(n_selected, device=device, dtype=dtype)
        else:
            weights = torch.tensor(self.layer_weights, device=device, dtype=dtype)
        weights_sum = weights.sum().clamp(min=1e-8)
        return weights / weights_sum

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        n_layers = len(hidden_states) - 1
        resolved_indices = self._resolve_layer_indices(n_layers=n_layers)
        selected = [hidden_states[idx] for idx in resolved_indices]
        if not selected:
            raise ValueError("No hidden layers selected for feature extraction.")

        if self.fusion == "weighted_sum":
            stacked = torch.stack(selected, dim=0)
            weights = self._normalized_weights(
                n_selected=len(selected),
                device=stacked.device,
                dtype=stacked.dtype,
            ).view(-1, 1, 1, 1)
            return (stacked * weights).sum(dim=0)

        return torch.cat(selected, dim=-1)
