#!/usr/bin/env python3
"""Turns raw text into training streams for the next-word network.

Reads the app vocabulary (en_words.txt; ids are line numbers, matching the app), adds two
specials, BOS = V and UNK = V+1, tokenizes input text into lowercase word tokens, and writes
train.bin / val.bin as little-endian uint32 streams where 0xFFFFFFFF separates sentences
(uint16 died when the vocabulary passed 64k). Sentences keep only runs with a high
in-vocab rate so UNK does not dominate training.

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
SEP = 0xFFFFFFFF


def load_vocab(path: str) -> dict:
    """Load vocabulary from file, returning dict mapping word -> id."""
    with open(path, encoding="utf-8") as f:
        return {w.strip(): i for i, w in enumerate(f)}


def tokenize(text: str) -> list:
    """Tokenize text into list of lists of words.

    Splits on . ! ? and newlines, lowercases, keeps apostrophe contractions
    as one token, converts curly apostrophes, and drops runs shorter than 3
    or longer than 40 words.
    """
    # Convert curly apostrophes to straight ones
    text = text.replace("’", "'")
    result = []
    for raw in SPLIT.split(text):
        words = WORD.findall(raw.lower())
        if 3 <= len(words) <= 40:
            result.append(words)
    return result


def to_ids(words: list, vocab: dict, unk: int) -> list:
    """Convert list of words to list of ids, mapping unknown words to unk."""
    return [vocab.get(w, unk) for w in words]


def keep_sentence(ids: list, unk: int, max_unk_rate: float = 0.2) -> bool:
    """Check if sentence should be kept based on UNK rate.

    Rejects if unk_rate > max_unk_rate, accepts ifunk_rate <= max_unk_rate.
    """
    unk_rate = sum(1 for i in ids if i == unk) / len(ids)
    return unk_rate <= max_unk_rate


def sniff_line(line: str) -> str:
    """Extract text from a line.

    For Tatoeba-style TSV (id\tlang\ttext): uses column 3 only when there are
    exactly three columns and column 2 is "eng".

    For other lines: returns the whole line.
    """
    parts = line.split("\t")
    if len(parts) == 3 and parts[1] == "eng":
        return parts[2]
    return line


class Writer:
    """Writes tokenized sentences to train.bin or val.bin based on random sampling."""

    def __init__(self, out_dir: str, rng: random.Random):
        self.out_dir = out_dir
        self.rng = rng
        self.train = open(f"{out_dir}/train.bin", "wb")
        self.val = open(f"{out_dir}/val.bin", "wb")
        self.kept = 0
        self.tokens = 0

    def emit(self, ids: list) -> None:
        """Write a sentence as little-endian uint32 stream followed by SEP."""
        buf = struct.pack(f"<{len(ids) + 1}I", *ids, SEP)
        (self.val if self.rng.random() < 0.01 else self.train).write(buf)
        self.kept += 1
        self.tokens += len(ids)

    def close(self) -> None:
        """Close output files."""
        self.train.close()
        self.val.close()


def main(argv: list) -> None:
    """Main entry point."""
    out_dir = argv[0]
    args = argv[1:]
    vocab_path = "app/src/main/assets/en_words.txt"
    if "--vocab" in args:
        i = args.index("--vocab")
        vocab_path = args[i + 1]
        del args[i:i + 2]
    inputs = args

    vocab = load_vocab(vocab_path)
    V = len(vocab)
    UNK = V + 1
    print(f"vocab {V} words, BOS={V}, UNK={UNK}")

    random.seed(7)
    writer = Writer(out_dir, random)

    for path in inputs:
        n0 = writer.kept
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = sniff_line(line)
                for words in tokenize(text):
                    ids = to_ids(words, vocab, UNK)
                    if keep_sentence(ids, UNK):
                        writer.emit(ids)
        print(f"{path}: {writer.kept - n0} sentences")

    writer.close()
    print(f"kept {writer.kept} sentences ({writer.tokens} tokens), skipped high-unk")


if __name__ == "__main__":
    main(sys.argv[1:])
