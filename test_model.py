import torch
from model import GPT
from tokenizer import CharTokenizer

text = open("data/input.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)

model = GPT(tokenizer.vocab_size)

x = torch.randint(0, tokenizer.vocab_size, (2, 32))
logits, loss = model(x, x)

print("Logits shape:", logits.shape)
print("Loss:", loss.item())
