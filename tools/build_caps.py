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

vocab = set(w.strip() for w in open("app/src/main/assets/en_words.txt", encoding="utf-8"))
forms = defaultdict(Counter)

for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.split("\t")
        text = parts[2] if len(parts) == 3 and parts[1] == "eng" else line
        toks = TOK.findall(text.replace("’", "'"))
        for t in toks[1:]:
            lo = t.lower()
            if lo in vocab:
                forms[lo][t] += 1

out = []
for lo, c in forms.items():
    total = sum(c.values())
    if total < MIN_SEEN:
        continue
    form, n = c.most_common(1)[0]
    if form != lo and form.lower() == lo and n / total >= MIN_RATIO:
        out.append((lo, form))
out.sort()

with open("app/src/main/assets/en_caps.txt", "w", encoding="utf-8") as f:
    f.write("# Canonical casing mined from corpus statistics (tools/build_caps.py):\n")
    f.write("# capitalized in >=90% of mid-sentence occurrences, seen >=25 times.\n")
    for lo, form in out:
        f.write(f"{lo}\t{form}\n")
print(f"wrote {len(out)} entries")
