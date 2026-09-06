#!/usr/bin/env python3
"""Append-only vocabulary expansion. Existing line numbers (= word ids in the binary
assets) never move; new words go at the end, ordered by frequency. Ids past 16 bits
have no bigram pairs (Bigrams guards the packed-key range) and score as UNK in the
network — they exist to pass the known-word gate, which is what stops autocorrect
from mangling real words like "minefield". Admission: a scanned word joins if it is a
dictionary headword, or derives from a trusted word by standard affixes (with
e-restoration and consonant undoubling), or is a prefixed form of a trusted word
(rewrote, foresaw, overrode)."""
import re
from wordfreq import top_n_list

CAP = 128000
PREFIXES = ("re", "un", "over", "under", "mis", "out", "pre", "dis", "co", "non", "fore")
SUFFIXES = ("ed", "ing", "es", "s", "er", "est", "ly", "ness", "ment", "ful", "less")
VOCAB_PATH = "app/src/main/assets/en_words.txt"


def load_vocab(path: str) -> list:
    """Load vocabulary from file, returning list of words in order."""
    return [w.strip() for w in open(path, encoding="utf-8")]


def load_web2_dict(path: str = "/usr/share/dict/words") -> set:
    """Load web2 dictionary, returning set of lowercase words."""
    return set(w.strip().lower() for w in open(path))


def make_trusted_set(current: list, web2: set) -> set:
    """Create set of trusted words from current vocab and web2."""
    return set(current) | web2


def compound(w: str, trusted: set) -> bool:
    """Check if word is a compound of two trusted words.

    Both parts must be 4+ letters: that admits real compounds while keeping out
    run-together function-word typos (andthe, ofthe) whose parts are short.
    """
    for i in range(4, len(w) - 3):
        if w[:i] in trusted and w[i:] in trusted:
            return True
    return False


def bases(w: str, suffixes: tuple = SUFFIXES, prefixes: tuple = PREFIXES) -> list:
    """Generate potential base forms of a word.

    Yields base forms by removing common suffixes (with e-restoration and
    consonant undoubling) and prefixes.
    """
    result = []
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            result.append(stem)
            result.append(stem + "e")  # making -> make
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                result.append(stem[:-1])  # stopped -> stop
    for pre in prefixes:
        if w.startswith(pre) and len(w) - len(pre) >= 3:
            result.append(w[len(pre):])
    return result


def word_matches_pattern(w: str) -> bool:
    """Check if word matches the allowed pattern: [a-z]+(?:'[a-z]+)?"""
    return bool(re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w))


def filter_candidates(
    current: list,
    added: list,
    web2: set,
    trusted: set,
    max_cap: int = CAP,
    max_word_len: int = 24,
) -> list:
    """Filter and return candidate words to add.

    A word is added if:
    - It hasn't been added already
    - It matches the word pattern
    - It's within length limit
    - It's in web2, or has a base in trusted, or is a compound of trusted words
    """
    have = set(current)
    added_set = set(added)

    candidates = []
    for w in top_n_list("en", 300000):
        if len(have) + len(candidates) >= max_cap:
            break
        if w in have or w in added_set:
            continue
        if not word_matches_pattern(w) or len(w) > max_word_len:
            continue
        ok = w in web2 or any(b in trusted for b in bases(w)) or compound(w, trusted)
        if ok:
            candidates.append(w)
    return candidates


def write_vocab(path: str, current: list, added: list) -> None:
    """Append new words to vocabulary file."""
    with open(path, "a", encoding="utf-8") as f:
        for w in added:
            f.write(w + "\n")


def main(argv: list) -> None:
    """Main entry point."""
    current = load_vocab(VOCAB_PATH)
    have = set(current)
    web2 = load_web2_dict()
    trusted = make_trusted_set(current, web2)

    added = filter_candidates(current, [], web2, trusted)

    write_vocab(VOCAB_PATH, current, added)
    print(f"appended {len(added)}; total {len(have) + len(added)}")

    # Print probe results
    probe_words = ["rewrote", "rewritten", "foresaw", "overrode", "quokka", "selfie",
                   "minefield", "weeknight", "andthe"]
    for probe in probe_words:
        print(probe, "in list" if (probe in have or probe in added) else "STILL MISSING")


if __name__ == "__main__":
    main([])
