"""Custom attention, positional encoding, and transformer blocks"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ALiBi(nn.Module):
    """Implementation of ALiBi relative positional encoding"""

    def __init__(self, num_heads):
        super().__init__()

        self.num_heads = num_heads

    def forward(self, x):

        # Get shape
        B, L, _ = x.shape
        device = x.device

        # positions [L]
        positions = torch.arange(L, device = device)

        # dist [L, L]
        dist = -torch.abs(positions[:, None] - positions[None, :])

        # slopes [num_heads]
        init_slope = 2**(-8 / self.num_heads)
        slopes = torch.full((self.num_heads,), init_slope, device = device)
        slopes = torch.cumprod(slopes, dim = 0)

        # biases [num_heads, L, L]
        biases = dist.unsqueeze(0).expand(self.num_heads, -1, -1)
        biases = slopes[:, None, None] * biases

        # out [B, num_heads, L, L]
        out = biases.unsqueeze(0).expand(B, -1, -1, -1)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        # Queries, keys, values
        self.q_map = nn.Linear(d_model, d_model)
        self.k_map = nn.Linear(d_model, d_model)
        self.v_map = nn.Linear(d_model, d_model)

        # Alibi positional encodings
        self.alibi = ALiBi(num_heads = self.num_heads)

        # Final FC
        self.o_map = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask = None):

        B, L, D = x.shape

        q = self.q_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        # Add alibi scores
        scores = scores + self.alibi(x)

        # Padding mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        a = torch.softmax(scores, dim = -1)

        out = a @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_map(out)

        return out


class SwiGLU(nn.Module):
    def __init__(self, input_dim, h_dim):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim

        self.gate_proj = nn.Linear(input_dim, h_dim)
        self.up_proj = nn.Linear(input_dim, h_dim)
        self.down_proj = nn.Linear(h_dim, input_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        out = self.down_proj(swish * up)

        return out


class MoELayer(nn.Module):
    def __init__(self, input_dim, h_dim, num_experts, top_k):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim,
        self.num_experts = num_experts
        self.top_k = top_k

        # Initialize the swiglu experts
        self.experts = nn.ModuleList([
            SwiGLU(input_dim, h_dim) for _ in range(num_experts)
        ])

        # Router for per-expert logits
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):

        B, L, D = x.shape

        # Reshape for expert processing [B * L, D]
        x_reshaped = x.reshape(-1, D)

        # Logits to [B * L, num_experts]
        router_logits = self.router(x_reshaped)

        # Get the top-k experts, then softmax to probabilty distribution over those k experts
        # Output [B * L, k]
        top_k_logits, top_k_idx = torch.topk(router_logits, self.top_k, dim = -1)
        top_k_probs = F.softmax(top_k_logits, dim = -1)

        # Initialize output tensor
        out = torch.zeros(
            B * L, D,
            device = x.device,
            dtype = x.dtype
        )

        # Process through selected experts
        unique_experts = torch.unique(top_k_idx)

        for i in unique_experts:
            expert_id = int(i)

            # Token mask [B*L] to decide which token of input should use this expert
            mask = (top_k_idx == expert_id)
            token_mask = mask.any(dim = 1)
            assert token_mask.any()

            # Select tokens, apply the expert, and add to output
            expert_input = x_reshaped[token_mask]
            expert_weight = top_k_probs[mask].unsqueeze(-1) # [N, 1]
            expert_output = self.experts[expert_id](expert_input) # [N, hidden_dim]

            out[token_mask] += expert_output * expert_weight

        # Reshape to original
        out = out.reshape(B, L, D)

        return out
    

class MoETransformer():
    def __init__(self, d_model, num_heads, h_dim, num_experts, top_k, p_drop):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = h_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Layers
        self.attention = MultiHeadAttention(
            d_model = self.d_model,
            num_heads = self.num_heads
        )
        self.dropout1 = nn.Dropout(p = p_drop)
        self.norm1 = nn.LayerNorm()
        self.moe = MoELayer(
            input_dim = self.d_model,
            h_dim = self.h_dim,
            num_experts = self.num_experts,
            top_k = self.top_k
        )
        self.dropout2 = nn.Dropout(p = p_drop)
        self.norm2 = nn.LayerNorm()

    def forward(self, x, attn_mask):

        x = self.norm1(x + self.dropout1(self.attention(x, attn_mask)))
        out = self.norm2(x + self.dropout2(self.moe(x)))

        return out