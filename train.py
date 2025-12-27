import torch
from tokenizer import CharTokenizer
from data_loader import DataLoader
from model import GPT
from config import (
    block_size,
    batch_size,
    learning_rate,
    max_iters,
    eval_interval
)

# ------------------
# Load data
# ------------------
text = open("data/input.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)

loader = DataLoader(
    text=text,
    tokenizer=tokenizer,
    block_size=block_size,
    batch_size=batch_size
)

# ------------------
# Device
# ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------
# Model
# ------------------
model = GPT(tokenizer.vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# ------------------
# Training loop
# ------------------
for step in range(max_iters):
    model.train()

    xb, yb = loader.get_batch()
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step} | loss {loss.item():.4f}")

# ------------------
# Save model
# ------------------
torch.save(model.state_dict(), "gpt_model.pth")
print("Model saved as gpt_model.pth")
