"""
Stage 5 Step 2: Train a BPE tokenizer on our mixed corpus.

Using ByteLevelBPETokenizer:
  - Operates on raw bytes, not characters.
  - Handles any input without UNK tokens (every byte is representable).
  - This is how GPT-2 tokenizes.

vocab_size=8000:
  - Large enough to learn common words and subwords.
  - Small enough to train fast on our corpus.
  - For comparison: GPT-2 uses 50,257; BERT uses 30,000.

After training we measure compression ratio (chars per token).
This is critical context for interpreting the loss comparison:
  - Char model: 1 token = 1 char
  - BPE model:  1 token = X chars on average
  A BPE model with higher loss may still be "better" if X is large enough.
  We compare using bits-per-character (BPC) to make losses comparable.
"""

import json
from tokenizers import ByteLevelBPETokenizer

CORPUS   = "corpus.txt"
SAVE_DIR = "."          # Saves bpe-vocab.json and bpe-merges.txt

# ---- Train ----
print("Training BPE tokenizer on corpus.txt ...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[CORPUS],
    vocab_size=8000,
    min_frequency=2,       # Merge must appear at least twice.
    special_tokens=["<|endoftext|>"],
)

tokenizer.save_model(SAVE_DIR, "bpe")
print("Saved: bpe-vocab.json, bpe-merges.txt")

# ---- Stats ----
print("\n--- Tokenizer stats ---")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Load corpus to measure compression ratio.
text = open(CORPUS, encoding="utf-8").read()
total_chars = len(text)

# Tokenize in chunks (encoding 21MB at once is slow).
CHUNK = 50_000
token_count = 0
for i in range(0, len(text), CHUNK):
    chunk = text[i : i + CHUNK]
    ids = tokenizer.encode(chunk).ids
    token_count += len(ids)

ratio = total_chars / token_count
print(f"Total chars:   {total_chars:,}")
print(f"Total tokens:  {token_count:,}")
print(f"Compression:   {ratio:.2f} chars/token")
print(f"  (char model: 1.00 chars/token)")
print(f"  (BPE model:  {ratio:.2f} chars/token → sees {ratio:.1f}x more context per block)")

# ---- Sample encodings ----
# Show how BPE tokenizes different text types.
print("\n--- Sample encodings ---")
examples = [
    ("Book prose",  "Congress shall make no law respecting an establishment of religion"),
    ("Lyrics",      "[Chorus] I will always love you, yeah yeah yeah"),
    ("News",        "The Federal Reserve raised interest rates by 25 basis points"),
    ("Rare word",   "Thou dost protest too much, methinks"),
]
for label, ex in examples:
    enc = tokenizer.encode(ex)
    tokens = enc.tokens
    print(f"\n{label}:")
    print(f"  text:   {ex}")
    print(f"  tokens: {tokens}")
    print(f"  ratio:  {len(ex)/len(tokens):.1f} chars/token")

# ---- Vocabulary sample ----
# The learned vocab shows what BPE decided to merge.
vocab = tokenizer.get_vocab()
# Sort by id (order they were added — merges happen in frequency order).
vocab_by_id = sorted(vocab.items(), key=lambda x: x[1])
print(f"\n--- Vocab sample (tokens 200-250, early merges) ---")
for token, idx in vocab_by_id[200:250]:
    print(f"  {idx:5d}: {repr(token)}")
