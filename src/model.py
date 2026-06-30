"""Custom models built using custom transformer building blocks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tranformer import SimpleTransformer, MoETransformer


class DenseGLM(nn.Module):
    """
    Model with dense attention and full FFN in each transformer block
    """
    def __init__(self, vocab_size, num_blocks, d_model, num_heads, p_drop):
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
    

class MoEGLM(nn.Module):
    def __init__(self, vocab_size, num_blocks, d_model, num_heads, h_dim, num_experts, top_k, p_drop):
        """
        Model with dense attention and MoE routing with SwiGLU experts
        """
        super().__init__()
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

        self.model = nn.ModuleList([
            MoETransformer(
                d_model = self.d_model,
                num_heads = self.num_heads,
                h_dim = self.h_dim,
                num_experts = self.num_experts,
                top_k = self.top_k,
                p_drop = self.p_drop
            ) for _ in range(num_blocks)
        ])

        # Final mapping to vocab size
        self.final = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, attn_mask):
        
        temp = self.embedding(x)
        aux_loss = 0.0

        for block in self.model:
            temp, loss = block(temp, attn_mask)
            aux_loss += loss

        logits = self.final(temp)

        return logits, aux_loss
    


# Finally, define a model with alternating dense and local attention
# Or even try interspersing linear attention ??
# Implement other sparsity tricks
