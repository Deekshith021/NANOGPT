import torch
from model import GPT
from tokenizer import CharTokenizer
from config import block_size

# --------------------
# Load data & tokenizer
# --------------------
text = open("data/input.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)

# --------------------
# Load model
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
model.eval()

# --------------------
# Text generation
# --------------------
def generate(
    model,
    idx,
    max_new_tokens=800,
    temperature=0.8
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx
