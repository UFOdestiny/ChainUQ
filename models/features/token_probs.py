"""Extract top-N token probability features from LLM logits."""
import torch
import torch.nn as nn

class TokenProbExtractor(nn.Module):
    def __init__(self, top_n: int = 4, temperature: float = 1.0, append_stats: bool = True):
        super().__init__()
        self.top_n = max(1, int(top_n))
        self.temperature = temperature
        self.append_stats = bool(append_stats)

    def feature_dim(self) -> int:
        # Extra token-uncertainty summary stats:
        # [max_logp, margin_logp, mean_topk_logp, std_topk_logp].
        return self.top_n + (4 if self.append_stats else 0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / max(self.temperature, 1e-8)
        log_probs = torch.log_softmax(scaled, dim=-1)
        top_values, _ = torch.topk(log_probs, self.top_n, dim=-1)
        if not self.append_stats:
            return top_values

        max_logp = top_values[..., :1]
        mean_topk_logp = top_values.mean(dim=-1, keepdim=True)
        if self.top_n > 1:
            margin_logp = top_values[..., :1] - top_values[..., 1:2]
            std_topk_logp = top_values.std(dim=-1, keepdim=True, unbiased=False)
        else:
            margin_logp = torch.zeros_like(max_logp)
            std_topk_logp = torch.zeros_like(max_logp)

        stats = torch.cat([max_logp, margin_logp, mean_topk_logp, std_topk_logp], dim=-1)
        return torch.cat([top_values, stats], dim=-1)
