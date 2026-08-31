#!/usr/bin/env python3
"""Head-to-head: bigram table vs next-word network on held-out sentences.

Replays val.bin (from prepare_data.py) and reports next-word top-1/top-3 hit rates for
both models, plus prefix-completion accuracy: after 2 typed letters of the target word,
does the model's reranking of the trie candidates put the right word first?

usage: eval_nextword.py data_dir/val.bin [--limit N]
"""
import struct
import sys

import numpy as np

SEP = 0xFFFF
K = 3

val_path = sys.argv[1]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 50000

words = [w.strip() for w in open("app/src/main/assets/en_words.txt", encoding="utf-8")]
ids = {w: i for i, w in enumerate(words)}
NW = len(words)
BOS, UNK = NW, NW + 1

# bigrams
data = open("app/src/main/assets/en_bigrams.bin", "rb").read()
bn = struct.unpack(">i", data[4:8])[0]
bkeys = np.frombuffer(data[8:8 + 8 * bn], dtype=">i8").astype(np.int64)
bscores = np.frombuffer(data[8 + 8 * bn:], dtype=np.uint8)

def bigram_top3(prev):
    lo = np.searchsorted(bkeys, prev << 16)
    hi = np.searchsorted(bkeys, (prev + 1) << 16)
    if hi <= lo:
        return []
    seg = np.argsort(-bscores[lo:hi].astype(np.int32), kind="stable")[:3]
    return [int(bkeys[lo + i] & 0xFFFF) for i in seg]

def bigram_score(prev, nxt):
    k = (prev << 16) | nxt
    i = np.searchsorted(bkeys, k)
    return int(bscores[i]) if i < bn and bkeys[i] == k else 0

# network (quantized arithmetic, mirroring the app)
raw = open("app/src/main/assets/en_nextword.bin", "rb").read()
V, KK, E = struct.unpack(">iii", raw[4:16])
o = 16
q = np.frombuffer(raw[o:o + V * E], dtype=np.int8).reshape(V, E).astype(np.int32); o += V * E
scale = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32); o += 4 * V
w1 = np.frombuffer(raw[o:o + 4 * E * KK * E], dtype=">f4").astype(np.float32).reshape(E, KK * E); o += 4 * E * KK * E
b1 = np.frombuffer(raw[o:o + 4 * E], dtype=">f4").astype(np.float32); o += 4 * E
bout = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)

def nn_logits(ctx):
    ctx = ([BOS] * K + ctx)[-K:]
    x = (q[ctx].astype(np.float32) * scale[ctx, None]).flatten()
    h = np.maximum(w1 @ x + b1, 0)
    hs = max(np.abs(h).max() / 127.0, 1e-8)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)
    return (q @ hq) * scale * hs + bout

# trie candidates for prefix reranking: words by 2-letter prefix, top by unigram rank
from collections import defaultdict
by_prefix = defaultdict(list)
for i, w in enumerate(words):
    if len(w) >= 3:
        by_prefix[w[:2]].append(i)
for k2 in by_prefix:
    by_prefix[k2] = by_prefix[k2][:8]

stream = np.fromfile(val_path, dtype="<u2").astype(np.int64)
stats = dict(n=0, b1=0, b3=0, n1=0, n3=0, pn=0, pb1=0, pn1=0)
sent = []
for t in stream:
    if t == SEP:
        sent = []
        continue
    t = int(t)
    if sent and t < NW:
        stats["n"] += 1
        prev = sent[-1]
        bt = bigram_top3(prev) if prev < NW else []
        stats["b1"] += bt[:1] == [t]
        stats["b3"] += t in bt
        lg = nn_logits(sent)
        lg[BOS] = lg[UNK] = -1e30
        top = np.argpartition(-lg, 3)[:3]
        top = top[np.argsort(-lg[top])]
        stats["n1"] += int(top[0]) == t
        stats["n3"] += t in top.tolist()
        # prefix reranking after two letters
        cands = by_prefix.get(words[t][:2], [])
        if t in cands and len(cands) > 1:
            stats["pn"] += 1
            if prev < NW:
                bs = [bigram_score(prev, c) for c in cands]
                stats["pb1"] += cands[int(np.argmax(bs))] == t if max(bs) > 0 else cands[0] == t
            else:
                stats["pb1"] += cands[0] == t
            ns = lg[cands]
            stats["pn1"] += cands[int(np.argmax(ns))] == t
        if stats["n"] >= limit:
            break
    sent.append(t)

n, pn = stats["n"], max(stats["pn"], 1)
print(f"next-word ({n} cases):  bigram top1 {stats['b1']/n:.3f} top3 {stats['b3']/n:.3f}   "
      f"network top1 {stats['n1']/n:.3f} top3 {stats['n3']/n:.3f}")
print(f"2-letter completion ({stats['pn']} cases):  bigram top1 {stats['pb1']/pn:.3f}   network top1 {stats['pn1']/pn:.3f}")
