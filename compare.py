"""
Stage 5 Step 3+4: Train char model vs BPE model, compare honestly.

Both models:
  n_embd=256, n_head=4, n_layer=4, dropout=0.2
  batch_size=32, block_size=128
  max_iters=5000, lr=3e-4

Only difference: tokenizer (and therefore vocab_size).

How to read the results:
  Raw val loss is NOT comparable across models.
    Char model predicts over ~96 classes.
    BPE model predicts over 8000 classes.
    A BPE loss of 3.0 is predicting over a much harder distribution.

  Bits-per-character (BPC) IS comparable.
    BPC = loss / ln(2) * tokens_per_char
    This normalizes both models to the same unit: bits needed to
    predict one character of the original text.

  Context window:
    block_size=128 tokens means different things:
      Char: 128 chars ~= 25 words
      BPE:  128 tokens ~= 437 chars ~= 87 words
    BPE sees ~3.4x more context. This is a real advantage.
    Not a flaw in the comparison -- it's what you get with BPE.
"""

import time
import csv
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer

# ---- Hyperparameters (identical for both models) ----
batch_size    = 32
block_size    = 128
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters    = 200
n_embd        = 256
n_head        = 4
n_layer       = 4
head_size     = n_embd // n_head
dropout       = 0.2

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
torch.manual_seed(1337)

# ---- Load corpus ----
print("Loading corpus...")
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"  {len(text):,} chars")

# ==============================================================
# TOKENIZER A: Character-level
# ==============================================================
print("\n--- Char tokenizer ---")
chars     = sorted(set(text))
vocab_size_char = len(chars)
stoi      = {ch: i for i, ch in enumerate(chars)}
itos      = {i: ch for i, ch in enumerate(chars)}

encode_char = lambda s: [stoi[c] for c in s]
decode_char = lambda l: "".join(itos[i] for i in l)

data_char = torch.tensor(encode_char(text), dtype=torch.long)
n = int(0.9 * len(data_char))
train_char = data_char[:n]
val_char   = data_char[n:]

# chars/token = 1.0 by definition.
char_compression = 1.0

print(f"  vocab_size: {vocab_size_char}")
print(f"  train tokens: {len(train_char):,}")
print(f"  context: {block_size} chars ~= {block_size//5} words")

# ==============================================================
# TOKENIZER B: BPE
# ==============================================================
print("\n--- BPE tokenizer ---")
bpe = ByteLevelBPETokenizer("bpe-vocab.json", "bpe-merges.txt")
vocab_size_bpe = bpe.get_vocab_size()

# Encode corpus in chunks to avoid memory issues.
CHUNK = 50_000
bpe_ids = []
for i in range(0, len(text), CHUNK):
    ids = bpe.encode(text[i : i + CHUNK]).ids
    bpe_ids.extend(ids)

data_bpe = torch.tensor(bpe_ids, dtype=torch.long)
n_bpe    = int(0.9 * len(data_bpe))
train_bpe_data = data_bpe[:n_bpe]
val_bpe_data   = data_bpe[n_bpe:]

bpe_compression = len(text) / len(bpe_ids)  # chars per token

print(f"  vocab_size: {vocab_size_bpe}")
print(f"  total tokens: {len(bpe_ids):,}")
print(f"  compression: {bpe_compression:.2f} chars/token")
print(f"  train tokens: {len(train_bpe_data):,}")
print(f"  context: {block_size} tokens ~= {int(block_size * bpe_compression)} chars ~= {int(block_size * bpe_compression / 5)} words")

# ==============================================================
# MODEL ARCHITECTURE (shared)
# ==============================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


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
    """Same architecture as transformer.py. vocab_size is passed in."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(
            torch.arange(T, device=device)
        )
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond   = idx[:, -block_size:]
            logits, _  = self(idx_cond)
            probs      = F.softmax(logits[:, -1, :], dim=-1)
            idx_next   = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat([idx, idx_next], dim=1)
        return idx


# ==============================================================
# TRAINING LOOP (reusable)
# ==============================================================

def get_batch(train_data, val_data, split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i : i + block_size] for i in ix])
    y  = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model_name, model, train_data, val_data, compression):
    """
    Train a model and return the loss history.
    compression: chars per token (1.0 for char, ~3.4 for BPE)
    """
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"{'='*50}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    log_rows   = []
    step_times = []

    for step in range(max_iters + 1):

        if step % eval_interval == 0:
            losses     = estimate_loss(model, train_data, val_data)
            train_l    = losses["train"]
            val_l      = losses["val"]
            # BPC = bits needed to predict one character.
            # Each BPE token covers `compression` chars on average.
            # bits_per_token = loss / ln(2)
            # bits_per_char  = bits_per_token / chars_per_token
            # BPC = loss / (ln(2) * compression)
            # For char: compression=1, so BPC = loss/ln(2). Same formula.
            val_bpc    = val_l / (math.log(2) * compression)
            avg_tps    = (batch_size * block_size) / (sum(step_times) / len(step_times)) if step_times else 0
            print(f"  step {step:5d}:  val_loss={val_l:.4f}  val_bpc={val_bpc:.4f}  tok/s={avg_tps:,.0f}")
            log_rows.append({
                "model": model_name, "step": step,
                "train_loss": round(train_l, 4),
                "val_loss": round(val_l, 4),
                "val_bpc": round(val_bpc, 4),
            })
            step_times = []  # Reset timer each interval.

        if step == max_iters:
            break

        t0 = time.time()
        xb, yb = get_batch(train_data, val_data, "train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_times.append(time.time() - t0)

    return log_rows


# ==============================================================
# RUN BOTH MODELS
# ==============================================================

# Reset seed before each run so weight init is the same.
torch.manual_seed(1337)
model_char = TransformerLM(vocab_size_char).to(device)
rows_char  = train("char", model_char, train_char, val_char, char_compression)

torch.manual_seed(1337)
model_bpe  = TransformerLM(vocab_size_bpe).to(device)
rows_bpe   = train("bpe", model_bpe, train_bpe_data, val_bpe_data, bpe_compression)

# ==============================================================
# SAVE LOG
# ==============================================================
log_path = "comparison_log.csv"
all_rows = rows_char + rows_bpe
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "step", "train_loss", "val_loss", "val_bpc"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nLog saved: {log_path}")

# ==============================================================
# FINAL COMPARISON
# ==============================================================

def final_bpc(rows):
    return rows[-1]["val_bpc"]

def final_val(rows):
    return rows[-1]["val_loss"]

print("\n" + "="*60)
print("FINAL COMPARISON (step 5000)")
print("="*60)
print(f"\n{'Metric':<30} {'Char':>10} {'BPE':>10}")
print("-" * 52)
print(f"{'val_loss (raw)':<30} {final_val(rows_char):>10.4f} {final_val(rows_bpe):>10.4f}")
print(f"{'val_bpc (comparable)':<30} {final_bpc(rows_char):>10.4f} {final_bpc(rows_bpe):>10.4f}")
print(f"{'vocab_size':<30} {vocab_size_char:>10} {vocab_size_bpe:>10}")
print(f"{'context (chars)':<30} {block_size:>10} {int(block_size * bpe_compression):>10}")
print(f"{'compression (chars/token)':<30} {char_compression:>10.2f} {bpe_compression:>10.2f}")
print()

# Who wins on BPC?
c_bpc = final_bpc(rows_char)
b_bpc = final_bpc(rows_bpe)
if b_bpc < c_bpc:
    diff = 100 * (c_bpc - b_bpc) / c_bpc
    print(f"BPE wins on BPC by {diff:.1f}%")
else:
    diff = 100 * (b_bpc - c_bpc) / b_bpc
    print(f"Char wins on BPC by {diff:.1f}%")
    print("Note: this may improve with more training steps for BPE.")

# ==============================================================
# SAMPLE OUTPUTS
# ==============================================================

print("\n" + "="*60)
print("SAMPLE OUTPUTS (200 tokens each)")
print("="*60)

# Char model sample.
print("\n--- Char model ---")
seed_char = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
out_char  = model_char.generate(seed_char, max_new_tokens=200)
print(decode_char(out_char[0].tolist()))

# BPE model sample.
print("\n--- BPE model ---")
# Start from newline token.
nl_token = bpe.encode("\n").ids[0]
seed_bpe = torch.tensor([[nl_token]], dtype=torch.long, device=device)
out_bpe  = model_bpe.generate(seed_bpe, max_new_tokens=200)
print(bpe.decode(out_bpe[0].tolist()))
