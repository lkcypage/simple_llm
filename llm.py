import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BATCH_SIZE = 16
BLOCK_SIZE = 256

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6

DROPOUT = 0.1

LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILE = "mini_gpt.pt"
CORPUS_FILE = "corpus.txt"


# ============================================================
# TOKENIZER
# ============================================================

# Usamos o tokenizer GPT-2 do tiktoken.
enc = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = enc.n_vocab

print(f"Device: {DEVICE}")
print(f"Vocabulário: {VOCAB_SIZE}")


# ============================================================
# DADOS
# ============================================================

if not os.path.exists(CORPUS_FILE):
    raise FileNotFoundError(
        f"Crie um arquivo '{CORPUS_FILE}' contendo o texto "
        "usado para treinamento."
    )

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    text = f.read()

if len(text.strip()) == 0:
    raise ValueError("O corpus.txt está vazio.")

tokens = enc.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

print(f"Caracteres: {len(text):,}")
print(f"Tokens: {len(tokens):,}")


# 90% treino / 10% validação
split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]


def get_batch(split_name):
    dataset = train_data if split_name == "train" else val_data

    if len(dataset) <= BLOCK_SIZE:
        raise ValueError(
            f"O corpus precisa ter mais de {BLOCK_SIZE} tokens."
        )

    starts = torch.randint(
        0,
        len(dataset) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        dataset[i:i + BLOCK_SIZE]
        for i in starts
    ])

    y = torch.stack([
        dataset[i + 1:i + BLOCK_SIZE + 1]
        for i in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)


# ============================================================
# SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Máscara causal:
        # cada token só pode olhar para tokens anteriores.
        mask = torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)
        )

        self.register_buffer(
            "mask",
            mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # [B, T, C]
        # ->
        # [B, heads, T, head_dim]

        q = q.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        # Attention:
        # QK^T / sqrt(d)
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Impede olhar para o futuro
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attention = F.softmax(scores, dim=-1)

        attention = self.attn_dropout(attention)

        out = attention @ v

        # [B, heads, T, head_dim]
        # ->
        # [B, T, C]

        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        out = self.proj(out)
        out = self.resid_dropout(out)

        return out


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            dropout
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(
            n_embd,
            dropout
        )

    def forward(self, x):

        # Pre-LN Transformer
        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


# ============================================================
# GPT
# ============================================================

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout
    ):
        super().__init__()

        self.block_size = block_size

        # Embedding dos tokens
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Embedding das posições
        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd,
                n_head,
                dropout
            )
            for _ in range(n_layer)
        ])

        self.ln_final = nn.LayerNorm(n_embd)

        # Converte representação para logits
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        # Weight tying:
        # embedding e saída compartilham pesos.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequência maior que BLOCK_SIZE={self.block_size}"
            )

        positions = torch.arange(
            0,
            T,
            device=idx.device
        )

        token_emb = self.token_embedding(idx)

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Mantém somente o contexto suportado
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            # Último token
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Top-K sampling
            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                threshold = values[:, [-1]]

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ============================================================
# MODELO
# ============================================================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT
).to(DEVICE)

number_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parâmetros: "
    f"{number_parameters / 1e6:.2f}M"
)


# ============================================================
# OTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.1
)


# ============================================================
# AVALIAÇÃO
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split_name in ["train", "val"]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            x, y = get_batch(split_name)

            _, loss = model(x, y)

            losses[k] = loss.item()

        results[split_name] = losses.mean().item()

    model.train()

    return results


# ============================================================
# TREINAMENTO
# ============================================================

print("\nIniciando treinamento...\n")

for iteration in range(MAX_ITERS):

    if iteration % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        print(
            f"step {iteration:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(x, y)

    loss.backward()

    # Evita gradientes explosivos
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()


# ============================================================
# SALVAR
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT
        }
    },
    MODEL_FILE
)

print(f"\nModelo salvo em {MODEL_FILE}")


# ============================================================
# GERAÇÃO
# ============================================================

prompt = "Olá, meu nome é"

prompt_tokens = enc.encode(prompt)

input_tensor = torch.tensor(
    [prompt_tokens],
    dtype=torch.long,
    device=DEVICE
)

output = model.generate(
    input_tensor,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50
)

generated_text = enc.decode(
    output[0].tolist()
)

print("\n==============================")
print("GERAÇÃO")
print("==============================\n")
print(generated_text)