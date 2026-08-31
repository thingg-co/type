#!/usr/bin/env python3
"""Turns raw text into training streams for the next-word network.

Reads the app vocabulary (en_words.txt; ids are line numbers, matching the app), adds two
specials, BOS = V and UNK = V+1, tokenizes input text into lowercase word tokens, and writes
train.bin / val.bin as little-endian uint16 streams where 0xFFFF separates sentences.
Sentences keep only runs with a high in-vocab rate so UNK does not dominate training.

usage: prepare_data.py out_dir input.txt [input2.txt ...]
  Tatoeba eng_sentences.tsv (id\tlang\ttext) and plain text (one or more sentences per
  line, wikitext-style) are both accepted; format is sniffed per line.
"""
import random
import re
import struct
import sys

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
SPLIT = re.compile(r"[.!?\n]+")
SEP = 0xFFFF

out_dir = sys.argv[1]
inputs = sys.argv[2:]

vocab = {w.strip(): i for i, w in enumerate(open("app/src/main/assets/en_words.txt", encoding="utf-8"))}
V = len(vocab)
UNK = V + 1
print(f"vocab {V} words, BOS={V}, UNK={UNK}")

random.seed(7)
train = open(f"{out_dir}/train.bin", "wb")
val = open(f"{out_dir}/val.bin", "wb")
kept = skipped = tokens = 0

def emit(sentence_ids):
    global kept, tokens
    buf = struct.pack(f"<{len(sentence_ids) + 1}H", *sentence_ids, SEP)
    (val if random.random() < 0.01 else train).write(buf)
    kept += 1
    tokens += len(sentence_ids)

def handle_text(text):
    global skipped
    for raw in SPLIT.split(text):
        words = WORD.findall(raw.lower())
        if not 3 <= len(words) <= 40:
            continue
        ids = [vocab.get(w, UNK) for w in words]
        unk_rate = sum(1 for i in ids if i == UNK) / len(ids)
        if unk_rate > 0.2:
            skipped += 1
            continue
        emit(ids)

for path in inputs:
    n0 = kept
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.split("\t")
        handle_text(parts[2] if len(parts) == 3 and parts[1] == "eng" else line)
    print(f"{path}: {kept - n0} sentences")

train.close()
val.close()
print(f"kept {kept} sentences ({tokens} tokens), skipped {skipped} high-unk")
