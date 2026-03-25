"""
Stage 5 Step 1: Build a mixed corpus from three sources.

Sources:
  - Books:  sedthh/gutenberg_english  (~7MB)
  - Lyrics: amishshah/song_lyrics     (~7MB)
  - News:   cc_news                   (~6MB)

Total target: ~20MB. Small enough that BPE training is fast and
5000 training steps don't over-repeat the data.

Filtering rules:
  - Remove nulls
  - Remove entries shorter than 100 chars (fragments, not useful)
  - No other cleaning — keep punctuation, structure tags, etc.
    The tokenizer should see the real data distribution.
"""

import unicodedata
from datasets import load_dataset

def normalize(text):
    """
    Convert to ASCII. Drops accented chars, emoji, non-Latin scripts.
    Loses 0.2% of content on this corpus — acceptable.
    Without this, char vocab is 778 (vs ~100 for clean ASCII).
    The char model can't learn 600+ rare Unicode classes anyway.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii")

TARGET_BYTES = 7 * 1024 * 1024   # 7MB per source
OUT_FILE     = "corpus.txt"
MIN_LEN      = 100                # chars

def stream_source(name, text_fn, target_bytes):
    """
    Stream a dataset, extract text with text_fn, filter, yield strings.
    Stops when we've collected target_bytes.
    Returns (entries_kept, bytes_collected, entries_skipped).
    """
    ds = load_dataset(name, split="train", streaming=True, trust_remote_code=True)
    kept = 0
    skipped = 0
    collected = 0
    texts = []

    for row in ds:
        text = text_fn(row)
        if not text or len(text) < MIN_LEN:
            skipped += 1
            continue
        text = normalize(text)
        if len(text) < MIN_LEN:   # Re-check after normalization.
            skipped += 1
            continue
        texts.append(text)
        kept += 1
        collected += len(text.encode("utf-8"))
        if collected >= target_bytes:
            break

    return texts, kept, collected, skipped


print("=== Building corpus ===\n")

all_sections = []

# ---- Books ----
print("Loading books (sedthh/gutenberg_english)...")
# TEXT field contains the full book text. One row = one book chapter or section.
texts, kept, nbytes, skip = stream_source(
    "sedthh/gutenberg_english",
    lambda r: r.get("TEXT", ""),
    TARGET_BYTES,
)
all_sections.append(("BOOKS", texts))
print(f"  kept {kept} sections, skipped {skip}, {nbytes/1024/1024:.1f} MB")

# ---- Lyrics ----
print("Loading lyrics (amishshah/song_lyrics)...")
# lyrics field. Keep the structural tags ([Chorus:] etc) — real data structure.
texts, kept, nbytes, skip = stream_source(
    "amishshah/song_lyrics",
    lambda r: r.get("lyrics", ""),
    TARGET_BYTES,
)
all_sections.append(("LYRICS", texts))
print(f"  kept {kept} songs, skipped {skip}, {nbytes/1024/1024:.1f} MB")

# ---- News ----
print("Loading news (cc_news)...")
# title + text. Combines headline context with body.
def news_text(r):
    title = r.get("title") or ""
    body  = r.get("text") or ""
    combined = f"{title}\n{body}".strip()
    return combined

texts, kept, nbytes, skip = stream_source(
    "cc_news",
    news_text,
    TARGET_BYTES,
)
all_sections.append(("NEWS", texts))
print(f"  kept {kept} articles, skipped {skip}, {nbytes/1024/1024:.1f} MB")

# ---- Combine and write ----
print(f"\nWriting {OUT_FILE}...")
total_chars = 0
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for source_name, texts in all_sections:
        f.write(f"<<<{source_name}>>>\n\n")
        for text in texts:
            f.write(text.strip())
            f.write("\n\n")   # Two newlines between entries.
        total_chars += sum(len(t) for t in texts)

print(f"\n=== Done ===")
print(f"Total chars:  {total_chars:,}")
print(f"Total MB:     {total_chars / 1024 / 1024:.1f}")
print(f"Unique chars: {len(set(open(OUT_FILE).read()))}")
