import os
import re

FILES = [
    "wikitext-103/wiki.train.tokens",
    "wikitext-103/wiki.valid.tokens",
    "wikitext-103/wiki.test.tokens",
]

OUTPUT = "data/input.txt"

# remove non-printable characters
def clean_line(line):
    line = line.replace("\x00", "")          # remove null bytes
    line = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", line)
    return " ".join(line.strip().split())

with open(OUTPUT, "w", encoding="utf-8", errors="ignore") as out:
    # chat markers (important)
    out.write("User:\nAssistant:\n\n")

    for file in FILES:
        print(f"Processing {file}...")
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    line = clean_line(line)
                    if line:
                        out.write(line + "\n")
                except Exception:
                    # skip any problematic line
                    continue

print("✅ input.txt created safely on Windows")
