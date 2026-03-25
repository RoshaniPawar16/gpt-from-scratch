# gpt-from-scratch

Building a GPT from scratch on a MacBook with 8GB RAM. No shortcuts.

## Why
I wanted to understand how LLMs actually work. Not use one. Build one.

## Hardware
MacBook, Apple Silicon, 8GB unified memory. MPS backend for PyTorch.

## Progress

### Stage 1 — Bigram model
- Character-level tokenizer. 65 unique chars.
- 4,225 parameters. Just an embedding table.
- Val loss: 2.50
- Output is garbage but structured garbage. It learned basic char patterns.

### Stage 3 — Full transformer
- Added embeddings, self-attention, feed-forward, LayerNorm, residuals, stacked blocks.
- 12.7M parameters.
- Val loss: 1.54 at 5k steps.
- Generates valid dialogue structure. Not real Shakespeare but close enough to see it's learning.

### What I learned so far
- The feed-forward layer does more than attention. Attention decides where to look, FF does the reasoning.
- Loss table per component is the best way to see what actually matters.
- 8GB is enough if you stay under 15M params.

## What's next
- Stage 5: Train on mixed corpus — screenplays, song lyrics, news articles.
- Option C: Compare BPE tokenizer vs char tokenizer on same corpus.

## Files
- `bigram.py` — Stage 1
- `transformer.py` — Stage 3, full GPT