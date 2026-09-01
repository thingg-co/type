#!/usr/bin/env python3
"""Calibration for the confusable-word thresholds in dict/Confusables.kt.

Two subcommands:

  extract cases.tsv input1.txt [input2.txt ...]
      Pulls occurrences of confusable words out of natural text. Every occurrence
      yields a "correct" row (the author's own usage; measures false positives) and
      one "swapped" row per alternative (a simulated mix-up; measures catch rate).
      Row format: label \t context \t typed \t intended \t next.

  report margins.tsv
      Aggregates the margins produced by ConfusableCalibrationHarness (run it with
      CONFUSABLE_CASES=cases.tsv, it writes confusable_margins.tsv next to the cases):
      false-positive and catch rates by threshold, plus a per-set table.

The forward margin decides at the word's own boundary from prior context; the lookback
margin re-scores the previous word once the word after it is known. Thresholds in
Confusables.kt were chosen from this report at roughly 0.25% false positives.
"""
import csv
import math
import random
import re
import sys
from collections import defaultdict

SETS = [
    ["were", "we're"], ["well", "we'll"], ["ill", "i'll"], ["id", "i'd"], ["its", "it's"],
    ["lets", "let's"], ["hell", "he'll"], ["shell", "she'll"], ["wed", "we'd"], ["whose", "who's"],
    ["your", "you're"], ["there", "their", "they're"], ["then", "than"], ["lose", "loose"],
    ["quite", "quiet"], ["weather", "whether"], ["accept", "except"], ["affect", "effect"],
    ["advice", "advise"], ["passed", "past"], ["brake", "break"],
]
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
CAP = 500  # rows per set per label; keeps the harness run to seconds


def extract(out_path, inputs):
    setof = {w: s for s in SETS for w in s}
    sents = []
    for path in inputs:
        text = open(path, encoding="utf-8-sig").read()
        m = re.search(r"\*\*\* START.*?\*\*\*(.*)\*\*\* END", text, re.S)  # Gutenberg wrapper
        if m:
            text = m.group(1)
        text = text.replace("’", "'").replace("_", " ")
        for p in re.split(r"\n\s*\n", text):
            for s in re.split(r"[.!?;]+", " ".join(p.split())):
                toks = WORD.findall(s.lower())
                if 3 <= len(toks) <= 40:
                    sents.append(toks)
    rows, per = [], defaultdict(int)
    for toks in sents:
        for i, w in enumerate(toks):
            if w not in setof:
                continue
            # next == "" marks a sentence-final confusable (the send-time forward+EOS
            # case); final == 1 marks rows whose following word ends the sentence (the
            # send-time lookback+EOS case).
            ctx, key = toks[max(0, i - 5):i], tuple(setof[w])
            nxt = toks[i + 1] if i < len(toks) - 1 else ""
            final = "1" if i >= len(toks) - 2 else "0"
            if per[key, "correct"] < CAP:
                rows.append(("correct", " ".join(ctx), w, w, nxt, final))
                per[key, "correct"] += 1
            for alt in setof[w]:
                if alt != w and per[key, "swapped"] < CAP:
                    rows.append(("swapped", " ".join(ctx), alt, w, nxt, final))
                    per[key, "swapped"] += 1
    random.seed(42)
    random.shuffle(rows)
    with open(out_path, "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(sents)} sentences -> {len(rows)} rows in {out_path}")


def report(margins_path):
    rows = [r for r in csv.DictReader(open(margins_path), delimiter="\t")
            if not r["skip"] and int(r["ctxlen"]) >= 1]

    def rate(rows, key, t, need_winner):
        app = [r for r in rows if r.get(key) and not math.isnan(float(r[key]))]
        hits = sum(1 for r in app
                   if float(r[key]) > t
                   and (not need_winner or r[key.replace("margin", "winner")] == r["intended"]))
        return hits, max(len(app), 1)

    correct = [r for r in rows if r["label"] == "correct"]
    swapped = [r for r in rows if r["label"] == "swapped"]
    keys = [("fwd_margin", "forward"), ("look_margin", "lookback")]
    if "fwd_eos_margin" in (rows[0] if rows else {}):
        keys += [("fwd_eos_margin", "send-time forward+EOS"), ("look_eos_margin", "send-time lookback+EOS")]
    for key, name in keys:
        print(f"== {name}: false positives on correct usage / catches on swapped")
        for t in [2, 3, 4, 5, 6, 8, 10]:
            fp, nc = rate(correct, key, t, False)
            c, ns = rate(swapped, key, t, True)
            print(f"  T={t}: FP {fp}/{nc} ({100 * fp / nc:.2f}%)  catch {c}/{ns} ({100 * c / ns:.1f}%)")
    print("== per set at fwd T=6 / look T=6")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["set"]][r["label"]].append(r)
    for s in sorted(by):
        cfp, cn = rate(by[s]["correct"], "fwd_margin", 6, False)
        cc, cs = rate(by[s]["swapped"], "fwd_margin", 6, True)
        lfp, ln = rate(by[s]["correct"], "look_margin", 6, False)
        lc, ls = rate(by[s]["swapped"], "look_margin", 6, True)
        print(f"  {s:24s} fwd FP {cfp}/{cn} catch {cc}/{cs}   look FP {lfp}/{ln} catch {lc}/{ls}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
