"""
Stage 4: Proper training — checkpointing, loss logging, longer runs.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- Hyperparameters ----
batch_size         = 32
block_size         = 128
max_iters          = 10000
eval_interval      = 500
learning_rate      = 3e-4
eval_iters         = 200
n_embd             = 256
n_head             = 4
n_layer            = 4
head_size          = n_embd // n_head
dropout            = 0.2
checkpoint_dir     = "checkpoints"
checkpoint_interval = 1000   # Save a checkpoint every this many steps.
log_file           = "loss_log.csv"

# MPS = Apple Silicon GPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
torch.manual_seed(1337)

# ---- Data ----
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ---- Batch loader ----
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---- Step 2: Single-head self-attention ----

class Head(nn.Module):
    """
    One head of self-attention.

    MATRIX MATH (read this before the code):
    -----------------------------------------
    Input X: shape (B, T, n_embd)
        B = batch size
        T = sequence length (block_size)
        n_embd = embedding dimension

    We project X into three different spaces:
        Q = X @ W_q   shape (B, T, head_size)   "query:  what am I looking for?"
        K = X @ W_k   shape (B, T, head_size)   "key:    what do I contain?"
        V = X @ W_v   shape (B, T, head_size)   "value:  what do I send if selected?"

    Attention scores (how much does each position attend to each other?):
        wei = Q @ K^T   shape (B, T, T)
        wei[b, i, j] = dot product of query[i] with key[j]
        High value = position i finds position j relevant.

    Scale to prevent softmax saturation:
        wei = wei / sqrt(head_size)
        Without this: as head_size grows, dot products grow → softmax
        pushes toward one-hot → gradients near zero.

    Causal mask (language models can't see the future):
        For each row i, set wei[i, j] = -inf for all j > i.
        After softmax, -inf → 0. Position i only sees positions 0..i.

    Softmax to get weights that sum to 1:
        wei = softmax(wei, dim=-1)   shape (B, T, T)
        wei[b, i, :] sums to 1. It's a probability distribution over positions.

    Weighted sum of values:
        out = wei @ V   shape (B, T, head_size)
        Each output position is a weighted average of all value vectors.
        The weights came from the query-key match.
    -----------------------------------------
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections. No bias is standard practice in attention.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # The causal mask. We register it as a buffer (not a parameter).
        # It's part of the model state but has no gradients.
        # tril = lower triangular matrix of 1s.
        # tril[i, j] = 1 if j <= i, else 0.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V. Shape: (B, T, head_size) each.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Raw attention scores. Shape: (B, T, T).
        # We need to transpose K's last two dims: (B, T, head_size) → (B, head_size, T).
        scale = k.shape[-1] ** -0.5   # 1 / sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * scale

        # Apply causal mask. Positions that are future (j > i) become -inf.
        # tril[:T, :T] handles variable sequence lengths.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax over the key dimension. Shape stays (B, T, T).
        wei = F.softmax(wei, dim=-1)

        # Weighted sum of values. Shape: (B, T, head_size).
        out = wei @ v
        return out

    # Note: we could add dropout on wei here (attention dropout).
    # Skipping for now to keep things readable.


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.

    Each head has its own Q, K, V weights and learns different patterns.
    We don't tell them what to learn — they figure it out via backprop.

    After all heads run:
    - Concatenate outputs: (B, T, n_head * head_size) = (B, T, n_embd)
    - Project through W_o (proj): (B, T, n_embd)

    Why the output projection?
        Concat already gives us n_embd. But the projection lets the model
        mix information across heads before passing it on.
    """

    def __init__(self, n_head, head_size):
        super().__init__()
        # Create n_head independent Head objects.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        # Output projection: mixes the concatenated head outputs.
        self.proj    = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run each head on the same input x.
        # Each head returns (B, T, head_size).
        # Concatenate along the last dim → (B, T, n_head * head_size).
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Two-layer MLP applied position-wise (independently to each token).

    Architecture:
        Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)

    Attention mixes information across positions.
    Feed-forward processes each position's representation individually.
    You can think of it as: attention = communication, FF = computation.

    The 4x expansion: from the original "Attention is All You Need" paper.
    It's empirically a good ratio. The wider middle layer gives the model
    more capacity to compute things per position.
    """

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
    """
    One transformer block: attention + feed-forward, each with pre-norm + residual.

    Pattern:
        x = x + attn(ln1(x))
        x = x + ffwd(ln2(x))

    We put layernorm BEFORE each sub-layer (pre-norm).
    Original paper put it after. Pre-norm is more stable for deep networks.

    The residual (+x) lets gradients flow straight back through the network.
    Without it, stacking blocks makes training hard — each layer has to
    "invent" everything from scratch instead of refining it.
    """

    def __init__(self):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Full GPT-style transformer.

    Architecture:
        idx
        → token_emb + pos_emb
        → Block × n_layer  (each: multi-head attn + ffwd, pre-norm + residual)
        → LayerNorm
        → lm_head (Linear → vocab_size)
        → logits
    """

    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # Stack n_layer blocks sequentially.
        # Each block is: attn(ln1(x)) + ffwd(ln2(x)), with residuals.
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        # Final LayerNorm before the lm_head. GPT-2 style.
        # Normalizes the output of the last block before projection.
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb                                             # (B, T, n_embd)

        # Run through all transformer blocks.
        x = self.blocks(x)         # (B, T, n_embd)

        # Final LayerNorm.
        x = self.ln_f(x)           # (B, T, n_embd)

        # Project to vocab. Shape: (B, T, vocab_size).
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context to block_size. Position embeddings only go up to block_size.
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ---- Checkpoint helpers ----

def save_checkpoint(step, model, optimizer):
    """
    Save model weights, optimizer state, and current step.
    We save all three so we can resume training exactly where we stopped.
    Optimizer state includes Adam's momentum buffers — without it,
    the first steps after resume behave differently.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # Save hyperparams so we can spot mismatches later.
        "config": {
            "n_embd": n_embd, "n_head": n_head,
            "n_layer": n_layer, "block_size": block_size,
        },
    }, path)
    print(f"  [checkpoint saved → {path}]")

def find_latest_checkpoint():
    """
    Look for the highest-numbered checkpoint file.
    Returns the path if found, None otherwise.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not files:
        return None
    # Sort by the step number embedded in the filename.
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])

def load_checkpoint(path, model, optimizer):
    """
    Load weights and optimizer state. Returns the step we left off at.
    If the checkpoint was saved with different hyperparams, warn but don't crash.
    """
    # weights_only=False because our checkpoint includes a config dict (not just tensors).
    # This is fine since we wrote the file ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Check for hyperparameter mismatch.
    saved_cfg = ckpt.get("config", {})
    for key, val in saved_cfg.items():
        current = {"n_embd": n_embd, "n_head": n_head,
                   "n_layer": n_layer, "block_size": block_size}[key]
        if current != val:
            print(f"  [WARNING] checkpoint {key}={val} but current {key}={current}. "
                  f"Loading anyway — this will likely fail or give wrong results.")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]

# ---- CSV log helpers ----

def init_log(path, resume_step):
    """
    Open the CSV log for writing.
    If we're resuming, append to the existing file.
    If starting fresh, write the header first.
    """
    is_new = not os.path.exists(path) or resume_step == 0
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["step", "train_loss", "val_loss"])
    return f, writer

def log_losses(writer, step, train_loss, val_loss):
    writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}"])

# ---- Build model + optimizer ----
model = TransformerLM().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---- Resume from checkpoint if one exists ----
start_step = 0
ckpt_path  = find_latest_checkpoint()
if ckpt_path:
    print(f"\nFound checkpoint: {ckpt_path}")
    start_step = load_checkpoint(ckpt_path, model, optimizer)
    print(f"Resuming from step {start_step}")
else:
    print("\nNo checkpoint found. Starting from scratch.")

# ---- Open CSV log ----
log_f, log_writer = init_log(log_file, start_step)

# ---- Training loop ----
print(f"\n--- Stage 4 training  (steps {start_step} → {max_iters}) ---")
for step in range(start_step, max_iters + 1):

    # Evaluate and log every eval_interval steps.
    if step % eval_interval == 0:
        losses = estimate_loss()
        train_l = losses["train"].item()
        val_l   = losses["val"].item()
        print(f"step {step:6d}:  train {train_l:.4f}  |  val {val_l:.4f}")
        log_losses(log_writer, step, train_l, val_l)
        log_f.flush()   # Write to disk immediately. Don't lose data if interrupted.

    if step == max_iters:
        break

    # Save checkpoint periodically.
    if step > 0 and step % checkpoint_interval == 0:
        save_checkpoint(step, model, optimizer)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Always save a final checkpoint.
save_checkpoint(max_iters, model, optimizer)
log_f.close()

# ---- Generate sample ----
seed = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
out  = model.generate(seed, max_new_tokens=300)
print("\n--- Sample output ---")
print(decode(out[0].tolist()))
