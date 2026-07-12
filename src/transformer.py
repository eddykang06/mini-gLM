"""Custom attention, positional encoding, and transformer blocks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.manual_seed(111)
flex_attention = torch.compile(flex_attention, dynamic = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ALiBi(nn.Module):
    """Implementation of ALiBi relative positional encoding from scratch"""

    def __init__(
        self, 
        num_heads: int
    ):
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


class ScratchMultiHeadAttention(nn.Module):
    """Implementation of multi-head attention from scratch"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int
    ):
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
        scores = scores + self.alibi(x).to(dtype = scores.dtype)

        # Padding mask
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]  # [B, 1, 1, L]
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        a = torch.softmax(scores, dim = -1)

        out = a @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_map(out)

        return out


def generate_alibi_slopes(num_heads):
    """
    Generate tensor of per-head ALiBi slopes using geometric sequence

    Args:
        num_heads : Number of heads in multi-head attention implementation
    
    Returns:
        slopes : Per-head slopes
    """
    slopes = 2**(-torch.arange(1, num_heads + 1) * 8 / num_heads)
    return slopes.to(device)


class FlexMultiHeadAttention(nn.Module):
    """Implementation of multi-head attention with Flex attention and ALiBi"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        self.alibi_slopes = generate_alibi_slopes(self.num_heads)

        self.q_map = nn.Linear(d_model, d_model)
        self.k_map = nn.Linear(d_model, d_model)
        self.v_map = nn.Linear(d_model, d_model)
        self.o_map = nn.Linear(d_model, d_model)
    
    def forward(self, x, attn_mask):

        # Define internal ALiBi function
        def alibi(score, b, h, q_idx, kv_idx):
            slope = self.alibi_slopes[h]
            bias = slope * -torch.abs(q_idx - kv_idx)
            return score + bias

        B, L, D = x.shape

        q = self.q_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_map(x).reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

        attn_mask_bool = attn_mask.to(device = q.device, dtype = torch.bool)

        # Define internal padding mask function
        def padding_mask(b, h, q_idx, kv_idx):
            q_valid = attn_mask_bool[b, q_idx]
            kv_valid = attn_mask_bool[b, kv_idx]
            return q_valid & kv_valid
    
        # Construct padding mask compatible with Flex attn
        block_mask = create_block_mask(
            padding_mask,
            B = B,
            H = self.num_heads,
            Q_LEN = L,
            KV_LEN = L,
            device = q.device,
            _compile = True
        )

        out = flex_attention(
            q, k, v, 
            score_mod = alibi, 
            block_mask = block_mask      # CHANGE BACK AFTER TRYING*****
        )

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_map(out)
        
        return out


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        p_drop: float
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.attention = FlexMultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads
        )
        self.dropout1 = nn.Dropout(p=p_drop)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Linear(d_model, d_model)
        self.relu = F.relu

        self.dropout2 = nn.Dropout(p=p_drop)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_out = self.attention(x, attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.relu(self.ff(x))
        out = self.norm2(x + self.dropout2(ff_out))

        return out
    
    
class SwiGLU(nn.Module):
    def __init__(
        self, 
        input_dim :int, 
        h_dim: int
    ):
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
    def __init__(
        self, 
        input_dim: int, 
        h_dim: int, 
        num_experts: int, 
        top_k: int
    ):
        super().__init__() 
        
        assert 1 <= top_k <= num_experts
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Initialize the swiglu experts
        self.experts = nn.ModuleList([
            SwiGLU(input_dim, h_dim) for _ in range(num_experts)
        ])

        # Router for per-expert logits
        self.router = nn.Linear(input_dim, num_experts)


    def forward(self, x, attn_mask = None):
        B, L, D = x.shape
        x_reshaped = x.reshape(B * L, D)

        if attn_mask is None:
            valid_mask = torch.ones(
                B * L,
                dtype=torch.bool,
                device=x.device
            )
        else:
            valid_mask = attn_mask.reshape(B * L).to(
                device=x.device,
                dtype=torch.bool
            )

        # Exclude padding before routing
        valid_x = x_reshaped[valid_mask]

        router_logits = self.router(valid_x)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_logits, top_k_idx = torch.topk(
            router_logits,
            self.top_k,
            dim=-1
        )
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Full output stays zero at padding positions
        out = torch.zeros_like(x_reshaped)
        valid_out = torch.zeros_like(valid_x)

        for expert_id_tensor in torch.unique(top_k_idx):
            expert_id = int(expert_id_tensor.item())

            selected = top_k_idx == expert_id
            token_mask = selected.any(dim=1)

            expert_input = valid_x[token_mask]
            expert_output = self.experts[expert_id](expert_input)
            expert_weight = top_k_probs[selected].unsqueeze(-1)

            valid_out[token_mask] += expert_output * expert_weight

        # Scatter valid token results back into [B*L, D]
        out[valid_mask] = valid_out
        out = out.reshape(B, L, D)

        expert_mask = F.one_hot(
            top_k_idx,
            num_classes=self.num_experts
        ).float()

        f = expert_mask.mean(dim=(0, 1))
        p = router_probs.mean(dim=0)
        aux_loss = torch.dot(f, p) * self.num_experts

        return out, aux_loss


class MoETransformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        h_dim: int, 
        num_experts: int, 
        top_k: int, 
        p_drop: float
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.h_dim = h_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Layers
        self.attention = FlexMultiHeadAttention(
            d_model = self.d_model,
            num_heads = self.num_heads
        )
        self.dropout1 = nn.Dropout(p = p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            input_dim = self.d_model,
            h_dim = self.h_dim,
            num_experts = self.num_experts,
            top_k = self.top_k
        )
        self.dropout2 = nn.Dropout(p = p_drop)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):

        attn_out = self.attention(x, attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        moe_out, aux_loss = self.moe(x, attn_mask)
        out = self.norm2(x + self.dropout2(moe_out))

        return out, aux_loss
