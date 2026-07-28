"""Custom models built using custom transformer building blocks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import (
    SimpleTransformer,
    LinearTransformerWithPositional,
    LinearTransformerStandard
)


class DenseGLM(nn.Module):
    """
    Model with dense attention and full FFN in each transformer block
    """
    def __init__(
        self, 
        vocab_size: int, 
        num_blocks: int, 
        d_model: int,
        num_heads: int,
        p_drop: float
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.p_drop = p_drop

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size + 2, # For masking and padding tokens
            embedding_dim = self.d_model
        )

        self.model = nn.ModuleList([
            SimpleTransformer(
                d_model = self.d_model,
                num_heads = self.num_heads,
                p_drop = self.p_drop
            ) for _ in range(num_blocks)
        ])

        # Final mapping to vocab size
        self.final = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, attn_mask):
        
        x = self.embedding(x)
        for block in self.model:
            x = block(x, attn_mask)

        logits = self.final(x)

        return logits
    
"""Linear Attention Only Model"""
class LinearGLM(nn.Module):
    def __init__(
        self, 
        num_blocks: int,
        d_model: int,
        num_heads: int,
        p_drop: float,
        vocab_size: int,
        max_seq_len: int
    ):     
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.p_drop = p_drop
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.block_1 = LinearTransformerWithPositional(
            d_model = d_model,
            num_heads = num_heads,
            p_drop = p_drop,
            vocab_size = vocab_size,
            max_seq_len = max_seq_len
        )

        self.model = nn.ModuleList(
            [self.block_1] + 
            [
                LinearTransformerStandard(
                    d_model = d_model,
                    num_heads = num_heads,
                    p_drop = p_drop,
                    vocab_size = vocab_size,
                    max_seq_len = max_seq_len
                ) for _ in range(num_blocks)
            ]
        )
        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, attn_mask):
        for block in self.model:
            x = block(x, attn_mask)
        logits = self.final(x)
        return logits