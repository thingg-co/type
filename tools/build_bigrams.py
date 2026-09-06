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


def load_vocab(path: str) -> dict:
    """Load vocabulary from file, returning dict mapping word -> id.

    Only words within 16-bit range are included in the id mapping.
    """
    with open(path, encoding="utf-8") as f:
        words = [w.strip() for w in f]
    # The packed key holds 16-bit ids; words past that line have no bigrams
    max_id = 1 << 16
    return {w: i for i, w in enumerate(words[:max_id])}


def parse_count_file(path: str) -> list:
    """Parse Norvig's count_2w.txt format.

    Returns list of (word1, word2, count) tuples.
    """
    pairs = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            a, b, count = parts[0].lower(), parts[1].lower(), int(parts[2])
            pairs.append((a, b, count))
    return pairs


def filter_and_pack_pairs(pairs: list, ids: dict) -> list:
    """Filter pairs where both words are in vocab and pack into 64-bit keys.

    Returns list of (count, packed_key) tuples where packed_key = (prev_id << 16) | next_id.
    """
    result = []
    for a, b, count in pairs:
        ia, ib = ids.get(a), ids.get(b)
        if ia is None or ib is None:
            continue
        packed_key = (ia << 16) | ib
        result.append((count, packed_key))
    return result


def compute_scores_and_sort(counts_and_keys: list) -> list:
    """Convert counts to scores: round(10 * ln(count)), clamped to 1..255.

    Input: list of (count, key) tuples
    Returns: list of (key, score) tuples sorted by key ascending.
    """
    entries = []
    for count, key in counts_and_keys:
        score = max(1, min(255, round(10 * math.log(count))))
        entries.append((key, score))
    # Sort by key ascending
    entries.sort(key=lambda x: x[0])
    return entries


def write_binary(path: str, entries: list) -> None:
    """Write binary format: magic "TBG1" + count (big-endian int32) + keys + scores.

    Format:
      - 4 bytes: magic "TBG1"
      - 4 bytes: number of entries (big-endian int32)
      - N * 8 bytes: keys (big-endian int64)
      - N * 1 byte: scores (uint8)
    """
    with open(path, "wb") as f:
        f.write(b"TBG1")
        f.write(struct.pack(">i", len(entries)))
        for key, _ in entries:
            f.write(struct.pack(">q", key))
        f.write(bytes(score for _, score in entries))


def main(argv: list) -> None:
    """Main entry point."""
    src = argv[0]
    cap = int(argv[1]) if len(argv) > 1 else 300_000

    vocab_path = "app/src/main/assets/en_words.txt"
    words = [w.strip() for w in open(vocab_path, encoding="utf-8")]
    # The packed key holds 16-bit ids; words past that line have no bigrams
    ids = {w: i for i, w in enumerate(words[: 1 << 16])}

    pairs = parse_count_file(src)
    # Filter and pack pairs: (count, packed_key)
    packed = filter_and_pack_pairs(pairs, ids)

    # Sort by count descending, take top cap
    packed.sort(reverse=True)
    packed = packed[:cap]

    # Compute scores and sort by key
    entries = compute_scores_and_sort(packed)

    out = "app/src/main/assets/en_bigrams.bin"
    write_binary(out, entries)
    print(f"{len(entries)} bigrams -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
