#!/usr/bin/env python3
"""Builds app/src/main/assets/en_bigrams.bin from Norvig's count_2w.txt.

Keeps bigrams whose two words are both in en_words.txt, packs them as
(word1_id << 16 | word2_id) int64 keys sorted ascending, with a byte score
(10 * ln(count), clamped to 1..255). Loaded by dict/Bigrams.kt.

usage: build_bigrams.py count_2w.txt [max_pairs]
"""
import math
import struct
import sys

src = sys.argv[1]
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000

words = [w.strip() for w in open("app/src/main/assets/en_words.txt", encoding="utf-8")]
ids = {w: i for i, w in enumerate(words)}
assert len(words) <= 0xFFFF, "word ids must fit 16 bits"

pairs = []
for line in open(src, encoding="utf-8", errors="ignore"):
    parts = line.split()
    if len(parts) != 3:
        continue
    a, b, count = parts[0].lower(), parts[1].lower(), int(parts[2])
    ia, ib = ids.get(a), ids.get(b)
    if ia is None or ib is None:
        continue
    pairs.append((count, (ia << 16) | ib))

pairs.sort(reverse=True)
pairs = pairs[:cap]
entries = sorted((key, max(1, min(255, round(10 * math.log(count)))) ) for count, key in pairs)

out = "app/src/main/assets/en_bigrams.bin"
with open(out, "wb") as f:
    f.write(b"TBG1")
    f.write(struct.pack(">i", len(entries)))
    for key, _ in entries:
        f.write(struct.pack(">q", key))
    f.write(bytes(score for _, score in entries))
print(f"{len(entries)} bigrams -> {out}")
