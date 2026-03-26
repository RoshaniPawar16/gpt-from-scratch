"""
GPT pipeline diagram. Saves diagram.png and diagram.svg.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 22))
ax.set_xlim(0, 13)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── colour palette ──────────────────────────────────────────
C = dict(
    raw    = "#f0f0f0",
    char   = "#d4e8f7",
    bpe    = "#d4f5d4",
    ids    = "#e8e8e8",
    emb    = "#daf0da",
    xvec   = "#eaf5d4",
    block  = "#fffbf0",
    ln     = "#e8e8e8",
    mha    = "#ffe0b3",
    res    = "#ecddf5",
    ff     = "#fff5b3",
    head   = "#f0d4f0",
    logits = "#f5e0f0",
    loss   = "#ffd4d4",
    skip   = "#7a5af8",
    arrow  = "#444444",
)

# ── helpers ──────────────────────────────────────────────────

def box(x, y, w, h, title, sub=None,
        color=C["ids"], ec="#555", lw=1.5,
        tfont=9, sfont=7.5, bold=False):
    """Draw a rounded rectangle with a title and optional subtitle."""
    r = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(r)
    ty = y + (0.13 if sub else 0.0)
    ax.text(x, ty, title,
            ha="center", va="center",
            fontsize=tfont,
            fontweight="bold" if bold else "normal",
            color="#111")
    if sub:
        ax.text(x, y - 0.20, sub,
                ha="center", va="center",
                fontsize=sfont, color="#555", style="italic")

def arr(x1, y1, x2, y2, color=C["arrow"]):
    """Straight arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )

def polyline_arrow(xs, ys, color=C["skip"]):
    """
    Draw a multi-segment line and put an arrowhead on the last segment.
    xs, ys are lists of x/y points.
    """
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.5, solid_capstyle="round")
    ax.annotate(
        "", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )

def hline(x1, x2, y, color="#ccc", ls="--"):
    ax.plot([x1, x2], [y, y], color=color, lw=1, ls=ls)


# ════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════
ax.text(6.5, 21.6,
        "GPT from scratch — full pipeline",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#111")

# ════════════════════════════════════════════════════════════
# RAW TEXT
# ════════════════════════════════════════════════════════════
box(6.5, 20.7, 6.5, 0.70,
    'Raw text:  "hello world, how are you?"',
    color=C["raw"], tfont=10)

# fork arrows
arr(4.8, 20.35, 3.2, 19.60)   # → char
arr(8.2, 20.35, 9.8, 19.60)   # → bpe

# ════════════════════════════════════════════════════════════
# TOKENIZERS
# ════════════════════════════════════════════════════════════
# Char tokenizer
box(3.2, 19.20, 5.6, 0.65,
    "CHAR TOKENIZER",
    "splits into individual characters",
    color=C["char"], bold=True)

ax.text(3.2, 18.62,
        '"hello"  →  [h, e, l, l, o]  →  [20, 43, 50, 50, 53]',
        ha="center", va="center",
        fontsize=7.5, family="monospace", color="#222")
ax.text(3.2, 18.32,
        "vocab ~96 chars  |  128 tokens = 128 chars ≈ 25 words of context",
        ha="center", va="center", fontsize=7, color="#555")

# BPE tokenizer
box(9.8, 19.20, 5.6, 0.65,
    "BPE TOKENIZER",
    "merges frequent char pairs into subword tokens",
    color=C["bpe"], bold=True)

ax.text(9.8, 18.62,
        '"hello"  →  [hello]  →  [4521]',
        ha="center", va="center",
        fontsize=7.5, family="monospace", color="#222")
ax.text(9.8, 18.32,
        "vocab 8000  |  128 tokens ≈ 437 chars ≈ 87 words of context",
        ha="center", va="center", fontsize=7, color="#555")

# dashed separator
hline(6.5, 6.5, 17.95)          # just a dot, effectively
ax.plot([6.5, 6.5], [18.0, 19.62], color="#bbb", lw=1, ls="--")

# converge arrows
arr(3.2, 18.00, 6.0, 17.35)
arr(9.8, 18.00, 7.0, 17.35)

# ════════════════════════════════════════════════════════════
# TOKEN IDs
# ════════════════════════════════════════════════════════════
box(6.5, 17.10, 6.0, 0.60,
    "Token IDs  [ 20, 43, 50, ... ]",
    "each token is an integer index into the vocabulary",
    color=C["ids"])
arr(6.5, 16.80, 6.5, 16.25)

# ════════════════════════════════════════════════════════════
# EMBEDDINGS
# ════════════════════════════════════════════════════════════
box(4.3, 15.95, 3.6, 0.60,
    "Token embedding",
    "what is this token?",
    color=C["emb"])
box(8.7, 15.95, 3.6, 0.60,
    "Position embedding",
    "where in the sequence?",
    color=C["emb"])
ax.text(6.5, 15.95, "+", ha="center", va="center",
        fontsize=14, color="#333", fontweight="bold")

arr(4.3, 15.65, 5.9, 15.10)
arr(8.7, 15.65, 7.1, 15.10)

box(6.5, 14.85, 6.0, 0.55,
    "x  =  token_emb  +  pos_emb",
    "shape (B, T, n_embd) — each token is a dense vector of size 512",
    color=C["xvec"])

arr(6.5, 14.57, 6.5, 14.10)

# ════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK  (outer dashed box)
# ════════════════════════════════════════════════════════════
# Content spans y ≈ 8.3 (bottom of res2) to y ≈ 14.0 (top of ln1).
# Add padding → outer box from y=7.90 to y=14.10.
BLK_X, BLK_Y0, BLK_Y1 = 1.4, 7.90, 14.10
blk_h  = BLK_Y1 - BLK_Y0
blk_cy = (BLK_Y0 + BLK_Y1) / 2
blk_w  = 10.2

outer = FancyBboxPatch(
    (BLK_X, BLK_Y0), blk_w, blk_h,
    boxstyle="round,pad=0.15",
    facecolor=C["block"], edgecolor="#e8a020", lw=2, ls="--",
)
ax.add_patch(outer)
ax.text(BLK_X + blk_w - 0.12, BLK_Y1 - 0.05,
        "× N blocks",
        ha="right", va="top",
        fontsize=9, color="#b86000", fontweight="bold")

# x enters block
ax.text(6.5, 14.02, "x in", ha="center", va="center",
        fontsize=8, color="#666")

# ── LayerNorm 1 ────────────────────────────────────────────
LN1_Y = 13.60
arr(6.5, 13.92, 6.5, 13.90)   # tiny nudge into LN1
box(6.5, LN1_Y, 4.5, 0.55,
    "LayerNorm",
    "rescale each token vector to mean=0, std=1",
    color=C["ln"])
arr(6.5, 13.32, 6.5, 12.85)

# ── Multi-head attention ───────────────────────────────────
MHA_Y = 12.55
box(6.5, MHA_Y, 6.0, 0.65,
    "Multi-head self-attention  (4 heads)",
    "each token looks at other tokens to decide what matters",
    color=C["mha"])

# Q K V annotations
ax.text(4.2, 12.03, "Q  what am I looking for?",
        fontsize=6.8, color="#884400")
ax.text(4.2, 11.80, "K  what do I contain?",
        fontsize=6.8, color="#884400")
ax.text(4.2, 11.57, "V  what do I send if selected?",
        fontsize=6.8, color="#884400")

arr(6.5, 12.22, 6.5, 11.82)

# ── Residual add 1 ─────────────────────────────────────────
RES1_Y = 11.52
box(6.5, RES1_Y, 4.5, 0.55,
    "add residual   x = x + attn(x)",
    "adds the original x back — gradients can skip straight through",
    color=C["res"])

# Skip connection 1: from just above LN1 → left → down → Res1
polyline_arrow(
    [5.4, 2.6, 2.6, 4.25],
    [13.88, 13.88, 11.52, 11.52],
)
ax.text(2.25, 12.7, "skip", fontsize=7.5, color=C["skip"],
        ha="center", va="center", rotation=90)

arr(6.5, 11.24, 6.5, 10.82)

# ── LayerNorm 2 ────────────────────────────────────────────
LN2_Y = 10.52
box(6.5, LN2_Y, 4.5, 0.55,
    "LayerNorm",
    "normalize again before feed-forward",
    color=C["ln"])
arr(6.5, 10.24, 6.5, 9.82)

# ── Feed-forward ───────────────────────────────────────────
FF_Y = 9.52
box(6.5, FF_Y, 6.5, 0.65,
    "Feed-forward   n_embd → 4×n_embd → n_embd",
    "each token thinks independently about what it just learned",
    color=C["ff"])
arr(6.5, 9.19, 6.5, 8.82)

# ── Residual add 2 ─────────────────────────────────────────
RES2_Y = 8.52
box(6.5, RES2_Y, 4.5, 0.55,
    "add residual   x = x + ff(x)",
    "same skip pattern as above",
    color=C["res"])

# Skip connection 2: from just above LN2 → left → down → Res2
polyline_arrow(
    [5.4, 3.0, 3.0, 4.25],
    [10.80, 10.80, 8.52, 8.52],
)
ax.text(2.65, 9.66, "skip", fontsize=7.5, color=C["skip"],
        ha="center", va="center", rotation=90)

# x exits block
ax.text(6.5, 8.05, "x out", ha="center", va="center",
        fontsize=8, color="#666")
arr(6.5, 7.96, 6.5, 7.52)

# ════════════════════════════════════════════════════════════
# FINAL LAYERNORM
# ════════════════════════════════════════════════════════════
box(6.5, 7.22, 4.5, 0.55,
    "Final LayerNorm",
    "normalize once more before projecting to vocab",
    color=C["ln"])
arr(6.5, 6.94, 6.5, 6.48)

# ════════════════════════════════════════════════════════════
# LM HEAD
# ════════════════════════════════════════════════════════════
box(6.5, 6.18, 5.5, 0.55,
    "lm_head   Linear( n_embd → vocab_size )",
    "project each position to one score per vocab item",
    color=C["head"])
arr(6.5, 5.90, 6.5, 5.44)

# ════════════════════════════════════════════════════════════
# LOGITS
# ════════════════════════════════════════════════════════════
box(6.5, 5.14, 5.0, 0.55,
    "Logits   shape (B, T, vocab_size)",
    "raw un-normalised score for every possible next token",
    color=C["logits"])
arr(6.5, 4.86, 6.5, 4.42)

# ════════════════════════════════════════════════════════════
# LOSS
# ════════════════════════════════════════════════════════════
box(6.5, 4.12, 5.5, 0.55,
    "Cross-entropy loss",
    "compare predicted distribution to actual next token",
    color=C["loss"], ec="#cc4444", lw=2)

# training / inference note
ax.text(6.5, 3.60,
        "training:   compute loss → backprop → update weights",
        ha="center", fontsize=8, color="#cc4444")
ax.text(6.5, 3.30,
        "inference:  skip loss → softmax logits → sample next token",
        ha="center", fontsize=8, color="#2255aa")

# ════════════════════════════════════════════════════════════
# LEGEND — tokenizer comparison callout
# ════════════════════════════════════════════════════════════
legend_y = 2.55
ax.text(1.2, legend_y + 0.30,
        "Tokenizer comparison at block_size=128:",
        fontsize=8, fontweight="bold", color="#333")
ax.add_patch(mpatches.FancyBboxPatch(
    (0.5, legend_y - 0.55), 5.4, 0.75,
    boxstyle="round,pad=0.1",
    facecolor=C["char"], edgecolor="#aaa", lw=1))
ax.text(3.2, legend_y - 0.18,
        "Char:  128 tokens = 128 chars ≈ 25 words",
        ha="center", fontsize=7.5, color="#333")

ax.add_patch(mpatches.FancyBboxPatch(
    (6.1, legend_y - 0.55), 5.4, 0.75,
    boxstyle="round,pad=0.1",
    facecolor=C["bpe"], edgecolor="#aaa", lw=1))
ax.text(8.8, legend_y - 0.18,
        "BPE:   128 tokens ≈ 437 chars ≈ 87 words",
        ha="center", fontsize=7.5, color="#333")

ax.text(6.5, legend_y - 0.75,
        "BPE sees 3.4× more context per forward pass",
        ha="center", fontsize=7.5, color="#555", style="italic")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig("diagram.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig("diagram.svg", bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: diagram.png  diagram.svg")
