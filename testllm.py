import torch
import tiktoken
import llm
checkpoint = torch.load(
    "mini_gpt.pt",
    map_location="cpu"
)

config = checkpoint["config"]

model = MiniGPT(
    vocab_size=config["vocab_size"],
    block_size=config["block_size"],
    n_embd=config["n_embd"],
    n_head=config["n_head"],
    n_layer=config["n_layer"],
    dropout=config["dropout"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

enc = tiktoken.get_encoding("gpt2")

prompt = "Olá, meu nome é"

tokens = enc.encode(prompt)

idx = torch.tensor(
    [tokens],
    dtype=torch.long
)

with torch.no_grad():

    output = model.generate(
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

texto = enc.decode(
    output[0].tolist()
)

print(texto)
