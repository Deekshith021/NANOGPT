import torch
import torch.nn as nn
from config import block_size, n_embd, dropout


# Single Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask (not trainable)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)

        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        # apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)
