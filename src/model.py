"""Custom models built using custom transformer building blocks"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGLM(nn.Module):
    def __init__(self, vocab_size, num_blocks, d_model, num_heads, h_dim, num_experts, top_k, p_drop):
        
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.h_dim = h_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.p_drop = p_drop

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.d_model
        )

        self.moe_transformer = MoETransformer(
            d_model = self.d_model,
            num_heads = self.num_heads,
            h_dim = self.h_dim,
            num_experts = self.num_experts,
            top_k = self.top_k,
            p_drop = self.p_drop
        )

        self.model = nn.ModuleList([
            self.moe_transformer for _ in range(num_blocks)
        ])

        # Final mapping to vocab size
        self.final = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):

        x = self.model(x)
        logits = self.final(x)

        return logits