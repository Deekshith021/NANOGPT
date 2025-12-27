import torch
from generate import generate, model, tokenizer

print("\n🧠 GPT Chatbot (type 'exit' to quit)\n")

history = ""

SYSTEM_PROMPT = (
    "Assistant is a helpful AI that answers clearly in simple English.\n\n"
)

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # build prompt
    prompt = SYSTEM_PROMPT + history + f"User: {user_input}\nAssistant:"

    # encode
    idx = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    # generate response
    out = generate(
        model,
        idx,
        max_new_tokens=800
    )

    # decode
    decoded = tokenizer.decode(out[0].tolist())

    # extract assistant reply
    reply = decoded[len(prompt):].split("User:")[0].strip()

    print("Bot:", reply, "\n")

    # update history
    history += f"User: {user_input}\nAssistant: {reply}\n"
