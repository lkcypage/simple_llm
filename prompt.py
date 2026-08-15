import torch
import tiktoken
from llm import MiniGPT

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
