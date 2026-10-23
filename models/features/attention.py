"""Attention weight feature extractor following LUH pattern."""
import torch
import torch.nn as nn

class AttentionExtractor(nn.Module):
    def __init__(self, layer_nums, head_nums="all", attn_history_sz=10,
                 pool=True, num_layers=32, num_heads=32):
        super().__init__()
        self._layer_nums = [(l % num_layers) if l < 0 else l for l in layer_nums]
        self._num_heads = num_heads
        self._attn_history_sz = attn_history_sz
        self._pool = pool
        if head_nums == "all":
            self._head_nums = {l: list(range(num_heads)) for l in self._layer_nums}
        else:
            heads = [int(h) for h in head_nums.split(",")]
            self._head_nums = {l: heads for l in self._layer_nums}
        n_selected_heads = len(list(self._head_nums.values())[0])
        if pool:
            self._input_size = n_selected_heads
        else:
            self._input_size = sum(len(h) for h in self._head_nums.values())

    def feature_dim(self) -> int:
        return self._input_size * self._attn_history_sz

    def output_attention(self) -> bool:
        return True

    def forward(self, attentions, attention_mask):
        batch_size = attentions[0].shape[0] if attentions[0] is not None else attentions[self._layer_nums[0]].shape[0]
        # Find a non-None layer to get seq_len
        ref_layer = next(la for la in (attentions[l] for l in self._layer_nums) if la is not None)
        seq_len = ref_layer.shape[2]
        all_features = []
        for pos in range(seq_len):
            pos_features = []
            for layer_num in self._layer_nums:
                cur_attn = attentions[layer_num]
                if cur_attn is None:
                    # Selective attention: this layer was skipped, fill zeros
                    head_indices = self._head_nums[layer_num]
                    zeros = torch.zeros(
                        batch_size, self._attn_history_sz, len(head_indices),
                        device=ref_layer.device, dtype=ref_layer.dtype,
                    )
                    pos_features.append(zeros)
                    continue
                cur_attn = cur_attn[:, :, pos, :]
                indices = torch.arange(pos, pos - self._attn_history_sz, -1, device=cur_attn.device)
                valid = indices >= 0
                indices = indices.clamp(min=0)
                gathered = cur_attn[:, :, indices].permute(0, 2, 1)
                head_indices = self._head_nums[layer_num]
                gathered = gathered[:, :, head_indices]
                gathered[:, ~valid, :] = 0.0
                pos_features.append(gathered)
            stacked = torch.stack(pos_features, dim=-1)
            all_features.append(stacked)
        all_features = torch.stack(all_features, dim=1)
        if self._pool:
            all_features = torch.amax(all_features, dim=-1)
        return all_features.reshape(batch_size, seq_len, -1)
