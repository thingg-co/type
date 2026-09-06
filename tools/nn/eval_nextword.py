#!/usr/bin/env python3
"""Head-to-head: bigram table vs next-word network on held-out sentences.

Replays val.bin (from prepare_data.py) and reports next-word top-1/top-3 hit rates for
both models, plus prefix-completion accuracy: after 2 typed letters of the target word,
does the model's reranking of the trie candidates put the right word first?

usage: eval_nextword.py data_dir/val.bin [--limit N]
"""
import struct
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # run as a file or under pytest alike
import tnw

SEP = 0xFFFFFFFF

val_path = sys.argv[1]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 50000
bin_path = sys.argv[sys.argv.index("--bin") + 1] if "--bin" in sys.argv else "app/src/main/assets/en_nextword.bin"

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
net = tnw.read_tnw(bin_path)
q = net["q"]
scale = net["scale"]
bout = net["bout"]
layers = net["layers"]
is_gru = net["kind"] == "gru"

# trie candidates for prefix reranking: words by 2-letter prefix, top by unigram rank
from collections import defaultdict
by_prefix = defaultdict(list)
for i, w in enumerate(words):
    if len(w) >= 3:
        by_prefix[w[:2]].append(i)
for k2 in by_prefix:
    by_prefix[k2] = by_prefix[k2][:8]

stream = np.fromfile(val_path, dtype="<u4").astype(np.int64)
stats = dict(n=0, b1=0, b3=0, n1=0, n3=0, pn=0, pb1=0, pn1=0)
CHUNK = 2048


def score_gru(cases):
    """Score a chunk of (context, target) cases for GRU networks with variable-length contexts."""
    if not cases:
        return

    # GRU: pass full sentence prefix (without target) to predict_logits_batch
    # We need to pad to the same length within each batch
    # Group by length first
    groups = {}
    for i, (sent, t) in enumerate(cases):
        length = len(sent)
        if length not in groups:
            groups[length] = []
        groups[length].append((i, sent, t))

    # Process each group
    for length, items in groups.items():
        # Build batch: each context is the sentence (no BOS padding needed inside GRU)
        ctx_list = [sent for _, sent, _ in items]
        ctx = _pad_contexts_gru(ctx_list)
        # ctx_lengths is the actual length of each context (before padding with 0s)
        ctx_lengths = [length for _, _, _ in items]
        logits = tnw.predict_logits_batch(net, ctx, ctx_lengths)

        # Update results
        for j, (orig_idx, sent, t) in enumerate(items):
            lg = logits[j]
            lg[BOS] = lg[UNK] = -1e30
            prev = sent[-1]
            bt = bigram_top3(prev) if prev < NW else []
            stats["b1"] += bt[:1] == [t]
            stats["b3"] += t in bt
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


def _pad_contexts_gru(ctx_list):
    """Pad variable-length contexts for GRU batch processing.

    GRU networks process the full sentence prefix. We pad shorter contexts
    with a sentinel value (0) that the GRU implementation will handle.

    Args:
        ctx_list: list of context word id lists

    Returns:
        np.ndarray of shape (N, max_len) padded with 0s
    """
    if not ctx_list:
        return np.zeros((0, 0), dtype=np.int64)

    max_len = max(len(c) for c in ctx_list)
    N = len(ctx_list)

    # Pad with 0s (which will be treated as padding by the GRU forward)
    result = np.zeros((N, max_len), dtype=np.int64)
    for i, c in enumerate(ctx_list):
        if len(c) > 0:
            result[i, :len(c)] = c

    return result


def score_ffn(cases, K):
    """Score a chunk of (context, target) cases for FFN networks with fixed K context."""
    if not cases:
        return
    ctx = np.array([([BOS] * K + c)[-K:] for c, _ in cases], dtype=np.int64)
    logits = tnw.predict_logits_batch(net, ctx)
    logits[:, BOS] = logits[:, UNK] = -1e30
    for (sent, t), lg in zip(cases, logits):
        prev = sent[-1]
        bt = bigram_top3(prev) if prev < NW else []
        stats["b1"] += bt[:1] == [t]
        stats["b3"] += t in bt
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


sent = []
pending = []
for t in stream:
    if t == SEP:
        sent = []
        continue
    t = int(t)
    if sent and t < NW:
        stats["n"] += 1
        pending.append((list(sent), t))
        if len(pending) >= CHUNK:
            if is_gru:
                score_gru(pending)
            else:
                score_ffn(pending, net["K"])
            pending = []
        if stats["n"] >= limit:
            break
    sent.append(t)
if is_gru:
    score_gru(pending)
else:
    score_ffn(pending, net["K"])
n, pn = stats["n"], max(stats["pn"], 1)
print(f"next-word ({n} cases):  bigram top1 {stats['b1']/n:.3f} top3 {stats['b3']/n:.3f}   "
      f"network top1 {stats['n1']/n:.3f} top3 {stats['n3']/n:.3f}")
print(f"2-letter completion ({stats['pn']} cases):  bigram top1 {stats['pb1']/pn:.3f}   network top1 {stats['pn1']/pn:.3f}")
