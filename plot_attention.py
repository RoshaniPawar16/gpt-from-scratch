"""
Attention weight heatmap for Stage 3 transformer checkpoint.

Loads checkpoints/ckpt_005000.pt, runs a short Shakespeare input,
captures attention weights from every head in every layer, and
saves a heatmap grid to plots/attention.png.

Architecture must match transformer.py exactly.
Config: n_embd=512, n_head=8, n_layer=4, block_size=128, vocab_size=65
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("plots", exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ── Tokenizer ────────────────────────────────────────────────
with open("shakespeare.txt", "r") as f:
    text = f.read()

chars  = sorted(set(text))
vocab_size = len(chars)
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# ── Model config ─────────────────────────────────────────────
n_embd     = 512
n_head     = 8
n_layer    = 4
block_size = 128
head_size  = n_embd // n_head
dropout    = 0.0

# ── Architecture (attribute names must match checkpoint) ──────

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # Attention weights are stored here after each forward pass.
        self.last_attn = None

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        w = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        self.last_attn = w.detach().cpu()   # shape (B, T, T)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads   = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa   = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(
            torch.arange(T, device=device)
        )
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x)


# ── Load checkpoint ───────────────────────────────────────────
print("Loading checkpoint...")
transformer = TransformerLM().to(device)
ckpt = torch.load("checkpoints/ckpt_005000.pt",
                  map_location=device, weights_only=False)
transformer.load_state_dict(ckpt["model"])
transformer.eval()
print(f"  loaded step {ckpt['step']}")

# ── Run forward pass on input sentence ───────────────────────
INPUT = "Hear me, people!"
tokens = encode(INPUT)
T = len(tokens)
idx = torch.tensor([tokens], dtype=torch.long, device=device)

print(f"Input: {repr(INPUT)}  ({T} tokens)")
print(f"Tokens: {tokens}")
print(f"Decoded: {[itos[i] for i in tokens]}")

with torch.no_grad():
    transformer(idx)

# ── Collect attention weights ─────────────────────────────────
# Shape per head: (1, T, T) → squeeze to (T, T)
attn_all = []   # attn_all[layer][head] = (T, T) tensor
for layer_idx, block in enumerate(transformer.blocks):
    layer_heads = []
    for head in block.sa.heads:
        w = head.last_attn[0]   # (T, T)
        layer_heads.append(w)
    attn_all.append(layer_heads)

# ── Describe what we see ──────────────────────────────────────
token_labels = [repr(itos[i])[1:-1] for i in tokens]   # strip outer quotes

def is_uniform(w, threshold=0.15):
    """True if max attention weight is close to 1/T (near-uniform)."""
    return w.max().item() < threshold

uniform_count = sum(
    is_uniform(attn_all[l][h])
    for l in range(n_layer)
    for h in range(n_head)
)
print(f"\nHeads with near-uniform weights (<0.15 max): {uniform_count}/{n_layer * n_head}")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(n_layer, n_head,
                          figsize=(n_head * 1.9, n_layer * 2.0),
                          constrained_layout=True)
fig.patch.set_facecolor("white")

for layer_idx in range(n_layer):
    for head_idx in range(n_head):
        ax = axes[layer_idx][head_idx]
        w  = attn_all[layer_idx][head_idx].numpy()   # (T, T)

        im = ax.imshow(w, vmin=0, vmax=1, cmap="Blues", aspect="auto")

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(token_labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(token_labels, fontsize=6)

        title = f"L{layer_idx+1} H{head_idx+1}"
        if is_uniform(w):
            title += " (uniform)"
        ax.set_title(title, fontsize=7, pad=3)

# Row labels on the left
for layer_idx in range(n_layer):
    axes[layer_idx][0].set_ylabel(f"Layer {layer_idx+1}", fontsize=8)

# Column labels on top
for head_idx in range(n_head):
    axes[0][head_idx].set_xlabel(f"Head {head_idx+1}", fontsize=7, labelpad=1)
    axes[0][head_idx].xaxis.set_label_position("top")

fig.suptitle(
    f'Attention weights — "{INPUT}"\n'
    f"Stage 3 checkpoint (step 5000, 12.7M params, 8 heads × 4 layers)",
    fontsize=10, fontweight="bold", y=1.01,
)

plt.savefig("plots/attention.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("\nSaved: plots/attention.png")
