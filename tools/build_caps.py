#!/usr/bin/env python3
"""Builds assets/en_caps.txt: canonical capitalization mined from corpus casing.

A word earns an entry when, mid-sentence (never counting the sentence-initial token),
at least 90% of its occurrences are written capitalized and it was seen at least 25
times. That yields months, weekdays, languages, places and names, while ambiguous
words whose lowercase reading dominates real text ("may", "march") stay out.

usage: build_caps.py sentences.tsv [more.txt ...]
  Tatoeba per-language tsv (id\tlang\ttext) and plain text are both accepted.
"""
import re
import sys
from collections import Counter, defaultdict

TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MIN_SEEN = 25
MIN_RATIO = 0.9
VOCAB_PATH = "app/src/main/assets/en_words.txt"
CAPS_PATH = "app/src/main/assets/en_caps.txt"


def load_vocab(path: str) -> set:
    """Load vocabulary from file, returning set of words."""
    return set(w.strip() for w in open(path, encoding="utf-8"))


def tokenize(text: str) -> list:
    """Tokenize text into list of words."""
    return TOK.findall(text.replace("’", "'"))


def count_forms(path: str, vocab: set) -> dict:
    """Count capitalized forms of vocabulary words mid-sentence.

    Returns dict mapping lowercase word -> Counter of capitalized forms.
    """
    forms = defaultdict(Counter)
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.split("\t")
        text = parts[2] if len(parts) == 3 and parts[1] == "eng" else line
        toks = tokenize(text)
        for t in toks[1:]:  # Skip first token (sentence-initial)
            lo = t.lower()
            if lo in vocab:
                forms[lo][t] += 1
    return forms


def build_entries(forms: dict) -> list:
    """Build entries from form counts.

    A word earns an entry when:
    - At least MIN_SEEN occurrences total
    - At least MIN_RATIO (90%) are the same capitalized form
    - The capitalized form differs from lowercase

    Returns list of (lowercase, form) tuples sorted alphabetically.
    """
    out = []
    for lo, counter in forms.items():
        total = sum(counter.values())
        if total < MIN_SEEN:
            continue
        form, n = counter.most_common(1)[0]
        if form != lo and form.lower() == lo and n / total >= MIN_RATIO:
            out.append((lo, form))
    out.sort()
    return out


def write_caps(path: str, entries: list) -> None:
    """Write caps file with header and entries."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Canonical casing mined from corpus statistics (tools/build_caps.py):\n")
        f.write("# capitalized in >=90% of mid-sentence occurrences, seen >=25 times.\n")
        for lo, form in entries:
            f.write(f"{lo}\t{form}\n")


def main(argv: list) -> None:
    """Main entry point."""
    vocab = load_vocab(VOCAB_PATH)

    forms = defaultdict(Counter)
    for path in argv:
        for line in open(path, encoding="utf-8", errors="ignore"):
            parts = line.split("\t")
            text = parts[2] if len(parts) == 3 and parts[1] == "eng" else line
            toks = tokenize(text)
            for t in toks[1:]:
                lo = t.lower()
                if lo in vocab:
                    forms[lo][t] += 1

    out = build_entries(forms)
    write_caps(CAPS_PATH, out)
    print(f"wrote {len(out)} entries")


if __name__ == "__main__":
    main(sys.argv[1:])
