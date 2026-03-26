"""
Generate loss curve plots from training CSVs.
Saves to plots/loss_curves.png.

Two panels:
  Left:  Char vs BPE — bits per character (BPC). Fair comparison.
  Right: Val loss by model size (Stage 4 size experiments).

Note: val_bpc in comparison_log.csv is wrong (old bug).
We recalculate BPC here:
  char: BPC = val_loss / ln(2)
  BPE:  BPC = val_loss / (ln(2) * 3.41)  -- divide by compression ratio
"""

import os
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("plots", exist_ok=True)

LN2   = math.log(2)
BPE_R = 3.41   # chars per BPE token, measured in train_bpe.py

# ── Load comparison_log.csv ─────────────────────────────────
char_steps, char_bpc = [], []
bpe_steps,  bpe_bpc  = [], []

with open("comparison_log.csv") as f:
    for row in csv.DictReader(f):
        step = int(row["step"])
        loss = float(row["val_loss"])
        if row["model"] == "char":
            char_steps.append(step)
            char_bpc.append(loss / LN2)
        else:
            bpe_steps.append(step)
            bpe_bpc.append(loss / (LN2 * BPE_R))   # correct formula

# ── Load size experiment CSVs ───────────────────────────────
def load_val(path):
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            vals.append(float(row["val_loss"]))
    return steps, vals

s_384, v_384 = load_val("loss_log_384_6h.csv")
s_512, v_512 = load_val("loss_log_512_8h.csv")

# ── Plot ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")

# Panel 1 — BPC comparison
ax1.plot(char_steps, char_bpc, color="#2563eb", lw=2, marker="o",
         markersize=4, label="Char tokenizer")
ax1.plot(bpe_steps,  bpe_bpc,  color="#16a34a", lw=2, marker="s",
         markersize=4, label="BPE tokenizer")

ax1.set_title("Char vs BPE — bits per character", fontsize=12, pad=10)
ax1.set_xlabel("Training step")
ax1.set_ylabel("Bits per character (BPC)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Annotate final values
ax1.annotate(f"  {char_bpc[-1]:.2f}", xy=(char_steps[-1], char_bpc[-1]),
             fontsize=8.5, color="#2563eb", va="center")
ax1.annotate(f"  {bpe_bpc[-1]:.2f}", xy=(bpe_steps[-1], bpe_bpc[-1]),
             fontsize=8.5, color="#16a34a", va="center")

# Note on what BPC means
ax1.text(0.03, 0.05,
         "Lower = better. BPC is comparable across tokenizers.\n"
         "BPE wins by 18% at step 5000.",
         transform=ax1.transAxes, fontsize=7.5,
         color="#555", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#ddd"))

# Panel 2 — size experiment val loss
ax2.plot(s_384, v_384, color="#d97706", lw=2, marker="o",
         markersize=4, label="n_embd=384, 6 heads  (7.2M params)")
ax2.plot(s_512, v_512, color="#7c3aed", lw=2, marker="s",
         markersize=4, label="n_embd=512, 8 heads  (12.7M params)")

ax2.set_title("Val loss by model size (Stage 4)", fontsize=12, pad=10)
ax2.set_xlabel("Training step")
ax2.set_ylabel("Val loss")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax2.annotate(f"  {v_384[-1]:.3f}", xy=(s_384[-1], v_384[-1]),
             fontsize=8.5, color="#d97706", va="center")
ax2.annotate(f"  {v_512[-1]:.3f}", xy=(s_512[-1], v_512[-1]),
             fontsize=8.5, color="#7c3aed", va="center")

ax2.text(0.03, 0.05,
         "Same 5k steps, same batch/block size.\n"
         "Bigger model helps but gains are small at this step count.",
         transform=ax2.transAxes, fontsize=7.5,
         color="#555", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#ddd"))

plt.suptitle("Training curves — GPT from scratch", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved: plots/loss_curves.png")
