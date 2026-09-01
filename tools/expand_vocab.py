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

current = [w.strip() for w in open("app/src/main/assets/en_words.txt", encoding="utf-8")]
have = set(current)
web2 = set(w.strip().lower() for w in open("/usr/share/dict/words"))
trusted = have | web2

def compound(w):
    # Closed compounds of two trusted words (minefield, weeknight, campsite). Both
    # parts must be 4+ letters: that admits real compounds while keeping out
    # run-together function-word typos (andthe, ofthe) whose parts are short.
    for i in range(4, len(w) - 3):
        if w[:i] in trusted and w[i:] in trusted:
            return True
    return False

def bases(w):
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            yield stem
            yield stem + "e"                       # making -> make
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                yield stem[:-1]                     # stopped -> stop
    for pre in PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 3:
            yield w[len(pre):]

added = []
for w in top_n_list("en", 300000):
    if len(have) + len(added) >= CAP:
        break
    if w in have or not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w) or len(w) > 24:
        continue
    ok = w in web2 or any(b in trusted for b in bases(w)) or compound(w)
    if ok:
        added.append(w)

with open("app/src/main/assets/en_words.txt", "a", encoding="utf-8") as f:
    for w in added:
        f.write(w + "\n")
print(f"appended {len(added)}; total {len(have) + len(added)}")
for probe in ["rewrote", "rewritten", "foresaw", "overrode", "quokka", "selfie", "minefield", "weeknight", "andthe"]:
    print(probe, "in list" if (probe in have or probe in added) else "STILL MISSING")
