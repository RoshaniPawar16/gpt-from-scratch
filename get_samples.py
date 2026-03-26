"""
Generate sample outputs for Stage 1 and Stage 3.
Prints results. Used to populate README samples section.

Stage 5 BPE: compare.py doesn't save checkpoints.
Output for that stage was captured during the training run.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(1337)

# ── Shared: load Shakespeare text ───────────────────────────
with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split, block_size=128, batch_size=32):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i : i + block_size] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ════════════════════════════════════════════════════════════
# STAGE 1 — Bigram model
# ════════════════════════════════════════════════════════════
print("Training Stage 1 bigram (3000 steps)...")

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

bigram = BigramLM().to(device)
opt = torch.optim.AdamW(bigram.parameters(), lr=1e-3)
for step in range(3000):
    xb, yb = get_batch("train")
    _, loss = bigram(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"  step {step}: loss {loss.item():.4f}")

seed = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
out = bigram.generate(seed, 200)
stage1_sample = decode(out[0].tolist())
print(f"  final loss: {loss.item():.4f}")
print("Stage 1 done.\n")

# ════════════════════════════════════════════════════════════
# STAGE 3 — Full transformer from checkpoint
# ════════════════════════════════════════════════════════════
print("Loading Stage 3 checkpoint (ckpt_005000.pt)...")

# Model must match the checkpoint config exactly.
# Config: n_embd=512, n_head=8, n_layer=4, block_size=128
n_embd    = 512
n_head    = 8
n_layer   = 4
block_size = 128
head_size  = n_embd // n_head
dropout    = 0.0   # no dropout at inference

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        # Attribute names must match transformer.py exactly to load the checkpoint.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        w = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        return w @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads   = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

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
        self.ffwd = FeedForward()   # must be 'ffwd', not 'ff'
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
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_c = idx[:, -block_size:]
            logits, _ = self(idx_c)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

transformer = TransformerLM().to(device)
ckpt = torch.load("checkpoints/ckpt_005000.pt",
                  map_location=device, weights_only=False)
transformer.load_state_dict(ckpt["model"])
transformer.eval()

seed = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
with torch.no_grad():
    out = transformer.generate(seed, 300)
stage3_sample = decode(out[0].tolist())
print("Stage 3 done.\n")

# ════════════════════════════════════════════════════════════
# Print both
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("STAGE 1 — Bigram (3k steps, 4K params)")
print("=" * 60)
print(stage1_sample)

print()
print("=" * 60)
print("STAGE 3 — Transformer from checkpoint (5k steps, 12.7M params)")
print("=" * 60)
print(stage3_sample)
