"""
Stage 1: Bigram language model.

A bigram model predicts the next character from only the current character.
No attention. No memory. Just a lookup table.

The model is literally: logits = embedding[x]
That's it. We embed the current token, and the embedding vector IS the prediction.

Why start here?
- Forces us to get the training loop right first.
- Gives us a baseline to beat later.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- Hyperparameters ----
# Start small. We can tune later.
batch_size  = 32    # How many sequences we process at once.
block_size  = 128   # How long each sequence is. Bigram only looks at 1 char,
                    # but we still use block_size to build batches efficiently.
max_iters   = 5000  # Number of training steps.
eval_interval = 500 # How often we print loss.
learning_rate = 1e-3
eval_iters  = 200   # How many batches we average over when estimating loss.

# Use MPS (Apple Silicon GPU) if available, else CPU.
# MPS is Apple's Metal Performance Shaders — it's the M-chip GPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.manual_seed(1337)  # Reproducibility.

# ---- Load data ----
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters in dataset: {len(text):,}")
print(f"First 200 chars:\n{text[:200]}")

# ---- Character tokenizer ----
# Step 1: find every unique character.
chars = sorted(set(text))
vocab_size = len(chars)
print(f"\nVocab size (unique chars): {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Step 2: make lookup tables.
# stoi = string to int. itos = int to string.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode: turn a string into a list of ints.
# decode: turn a list of ints back into a string.
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Quick test.
test_str = "Hello"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"\nTokenizer test: '{test_str}' → {encoded} → '{decoded}'")

# ---- Train / val split ----
# Convert entire text to a tensor of token ids.
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nData tensor shape: {data.shape}, dtype: {data.dtype}")

# 90% train, 10% val. No shuffling — it's a sequence.
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# ---- Batch loader ----
def get_batch(split):
    """
    Returns a batch of (inputs, targets).

    For a bigram model, input[i] predicts target[i].
    target[i] = input[i+1], i.e., the next character.

    We still use block_size sequences — it's more efficient than
    grabbing one char at a time. Each position in the sequence is
    one training example for the bigram model.
    """
    data_split = train_data if split == "train" else val_data

    # Pick batch_size random starting positions.
    # We need at least block_size+1 chars from each start.
    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    # x: the input sequences. Shape: (batch_size, block_size).
    x = torch.stack([data_split[i : i + block_size] for i in ix])

    # y: the target sequences. Shifted by 1. Shape: (batch_size, block_size).
    # y[b, t] is the correct next token after x[b, t].
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# ---- Loss estimation ----
@torch.no_grad()  # Don't track gradients during evaluation. Saves memory.
def estimate_loss():
    """Average loss over eval_iters batches. More stable than one-batch loss."""
    out = {}
    model.eval()  # Puts model in eval mode (affects dropout, batchnorm — not relevant now but good habit).
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Back to train mode.
    return out

# ---- Model ----
class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    It's just one embedding table of shape (vocab_size, vocab_size).

    For each input token x, the model returns embedding[x] as the logits.
    Logits are the raw scores before softmax. The model says:
    "given character x, here are my scores for what comes next."

    That's the whole model. One layer, no attention.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # Token embedding table. Each token maps to a row of size vocab_size.
        # We use vocab_size as embedding dim because the output IS the logit over vocab.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) — batch of token id sequences.
        targets: (B, T) — optional. If given, we compute loss.

        Returns logits and (optionally) loss.
        """
        # Look up embeddings. Shape: (B, T, vocab_size).
        # Each token id → a row in the embedding table.
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # cross_entropy expects (B, C, T) or (B*T, C).
            # We reshape to (B*T, C) — flatten batch and time dimensions.
            B, T, C = logits.shape
            logits_2d  = logits.view(B * T, C)   # (B*T, vocab_size)
            targets_1d = targets.view(B * T)      # (B*T,)
            loss = F.cross_entropy(logits_2d, targets_1d)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Auto-regressively generate max_new_tokens new characters.

        idx: (B, T) — the starting context.

        At each step:
        1. Run forward pass to get logits.
        2. Take only the last time step's logits (that's all bigram uses).
        3. Softmax → probabilities.
        4. Sample one token.
        5. Append it and repeat.
        """
        for _ in range(max_new_tokens):
            # Get logits. We only care about the last token for bigram.
            logits, _ = self(idx)

            # Take the last time step. Shape: (B, vocab_size).
            logits_last = logits[:, -1, :]

            # Convert to probabilities.
            probs = F.softmax(logits_last, dim=-1)

            # Sample from the distribution. Shape: (B, 1).
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token. Shape: (B, T+1).
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ---- Init model ----
model = BigramLanguageModel(vocab_size)
model = model.to(device)

# Count parameters.
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")
# For a bigram model this is just vocab_size * vocab_size = 65*65 = 4225. Tiny.

# ---- Optimizer ----
# AdamW is standard. lr=1e-3 is a reasonable starting point.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---- Training loop ----
print("\n--- Training ---")
for step in range(max_iters):

    # Evaluate loss periodically.
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d}:  train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

    # Get a batch.
    xb, yb = get_batch("train")

    # Forward pass. Compute loss.
    logits, loss = model(xb, yb)

    # Backward pass.
    optimizer.zero_grad(set_to_none=True)  # Clear old gradients.
    loss.backward()                         # Compute new gradients.
    optimizer.step()                        # Update weights.

# Final evaluation.
losses = estimate_loss()
print(f"step {max_iters:5d}:  train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

# ---- Generate text ----
print("\n--- Generated text (untrained baseline has ~4.17 loss, random char output) ---\n")

# Start from a newline character as the seed.
seed_token = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)  # (1, 1)

# Generate 500 characters.
generated_ids = model.generate(seed_token, max_new_tokens=500)

# Decode and print.
generated_text = decode(generated_ids[0].tolist())
print(generated_text)
