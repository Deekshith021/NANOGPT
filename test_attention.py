import torch
from attention import MultiHeadAttention
from config import n_embd, n_head

x = torch.randn(2, 32, n_embd)
attn = MultiHeadAttention(n_head, n_embd // n_head)

y = attn(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
