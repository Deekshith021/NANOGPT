from tokenizer import CharTokenizer
from config import block_size, batch_size
from data_loader import DataLoader

text = open("data/input.txt", "r", encoding="utf-8").read()

tokenizer = CharTokenizer(text)
loader = DataLoader(text, tokenizer, block_size, batch_size)

x, y = loader.get_batch()

print("Vocab size:", tokenizer.vocab_size)
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Sample decode:", tokenizer.decode(x[0].tolist()))
