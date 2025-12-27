# Test the Block (MANDATORY)
import torch
from block import Block
from config import n_embd

x = torch.randn(2, 32, n_embd)
block = Block()

y = block(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
