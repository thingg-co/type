#!/usr/bin/env python3
"""Trains the next-word network and exports the app asset.

Model: Bengio-style feed-forward LM. Last K=3 word ids -> tied 128-d embeddings ->
concat -> ReLU hidden (128) -> logits over the whole vocab through the tied embedding
matrix plus an output bias. Contexts are left-padded with BOS at sentence starts.

Export (en_nextword.bin, big-endian to match the app's DataInputStream):
  magic 'TNW1'
  int32 V(total incl. BOS+UNK), int32 K, int32 E
  emb:   V rows of E int8            (per-row scale = absmax/127)
  scale: V float32
  W1:    (K*E * E) float32  (row-major, out x in)
  b1:    E float32
  bout:  V float32
Inference: h = relu(W1 @ concat(emb[ctx]) + b1); q = int8(h / hs), hs = absmax(h)/127;
logit_v = scale[v] * hs * dot_int8(emb[v], q) + bout[v]. A golden.json with one input and
its top logits is exported for the Kotlin unit test.

usage: train.py data_dir out_dir [--steps N] [--dim E] [--batch B]
"""
import json
import math
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 3  # overridden by --k


def load_stream(path):
    a = np.fromfile(path, dtype="<u2").astype(np.int64)
    return a


def windows(stream, V):
    """(N, K) contexts and (N,) targets; BOS pads sentence starts, UNK targets dropped."""
    BOS, UNK, SEP = V - 2, V - 1, 0xFFFF
    ctxs, tgts = [], []
    sent = []
    for t in stream:
        if t == SEP:
            sent = []
            continue
        ctx = ([BOS] * K + sent)[-K:]
        if t != UNK:
            ctxs.append(ctx)
            tgts.append(t)
        sent.append(int(t))
    return np.array(ctxs, dtype=np.int64), np.array(tgts, dtype=np.int64)


class NextWord(nn.Module):
    def __init__(self, V, E, layers=1, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        dims = [K * E] + [hidden] * (layers - 1) + [E]
        mods = []
        for i in range(len(dims) - 1):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.trunk = nn.Sequential(*mods)
        self.bout = nn.Parameter(torch.zeros(V))
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ctx):
        e = self.emb(ctx).flatten(1)
        h = self.trunk(e)
        return h @ self.emb.weight.T + self.bout


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    global K
    K = int(args[args.index("--k") + 1]) if "--k" in args else K
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 30000
    E = int(args[args.index("--dim") + 1]) if "--dim" in args else 128
    B = int(args[args.index("--batch") + 1]) if "--batch" in args else 1024
    layers = int(args[args.index("--layers") + 1]) if "--layers" in args else 1
    hidden = int(args[args.index("--hidden") + 1]) if "--hidden" in args else 256
    negs = int(args[args.index("--negs") + 1]) if "--negs" in args else 8192

    n_words = sum(1 for _ in open("app/src/main/assets/en_words.txt", encoding="utf-8"))
    V = n_words + 2  # BOS, UNK
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"V={V} E={E} K={K} steps={steps} device={dev}")

    cache = f"{data_dir}/win_k{K}.npz"
    try:
        z = np.load(cache)
        tr_ctx, tr_tgt, va_ctx, va_tgt = z["tc"], z["tt"], z["vc"], z["vt"]
        print(f"windows loaded from cache {cache}")
    except Exception:
        tr_ctx, tr_tgt = windows(load_stream(f"{data_dir}/train.bin"), V)
        va_ctx, va_tgt = windows(load_stream(f"{data_dir}/val.bin"), V)
        np.savez(cache, tc=tr_ctx, tt=tr_tgt, vc=va_ctx, vt=va_tgt)
    print(f"train windows {len(tr_tgt)}, val {len(va_tgt)}")

    model = NextWord(V, E, layers, hidden).to(dev)
    print(f"layers={layers} hidden={hidden} params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-4)

    tr_ctx_t = torch.from_numpy(tr_ctx)
    tr_tgt_t = torch.from_numpy(tr_tgt)
    n = len(tr_tgt_t)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (B,))
        ctx = tr_ctx_t[idx].to(dev)
        tgt = tr_tgt_t[idx].to(dev)
        # Sampled softmax: score the batch targets plus shared random negatives instead of
        # the whole vocabulary. Validation below still uses the full softmax, so reported
        # metrics stay comparable across trainer versions.
        e = model.emb(ctx).flatten(1)
        h = model.trunk(e)
        # Zipfian negatives: ids are frequency-ordered, so log-uniform sampling yields the
        # hard, frequent negatives a uniform draw almost never picks.
        neg = (torch.rand(negs, device=dev) * math.log(float(V))).exp().long().clamp(1, V - 1) - 1
        cand = torch.cat([tgt, neg])
        logits = h @ model.emb.weight[cand].T + model.bout[cand]
        # logQ correction for the biased sampler (log-uniform: q ~ 1/((id+1) ln V)).
        logq = -torch.log((cand + 1).float()) - math.log(math.log(float(V)))
        logits = logits - logq.unsqueeze(0)
        # Accidental-hit masking: frequent targets appear many times per batch; without
        # this, duplicate columns compete with the true label and dilute the gradient.
        hits = tgt.unsqueeze(1) == cand.unsqueeze(0)
        hits[torch.arange(len(tgt), device=dev), torch.arange(len(tgt), device=dev)] = False
        logits = logits.masked_fill(hits, -1e9)
        loss = F.cross_entropy(logits, torch.arange(len(tgt), device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.3f} ({(time.time()-t0):.0f}s)", flush=True)

    # validation: perplexity and top-3 next-word hit rate
    model.eval()
    hits = top3 = total = 0
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(va_tgt), 8192):
            ctx = torch.from_numpy(va_ctx[i:i + 8192]).to(dev)
            tgt = torch.from_numpy(va_tgt[i:i + 8192]).to(dev)
            logits = model(ctx)
            nll += F.cross_entropy(logits, tgt, reduction="sum").item()
            top = logits.topk(3, dim=1).indices
            hits += (top[:, 0] == tgt).sum().item()
            top3 += (top == tgt.unsqueeze(1)).any(dim=1).sum().item()
            total += len(tgt)
    print(f"val ppl {math.exp(nll/total):.1f}  top1 {hits/total:.3f}  top3 {top3/total:.3f}")

    # ---- export ----------------------------------------------------------------------
    if layers != 1:
        print("multi-layer trunk: TNW1 export skipped (vetting run)")
        return
    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    w1 = model.trunk[0].weight.detach().cpu().numpy().astype(np.float32)
    b1 = model.trunk[0].bias.detach().cpu().numpy().astype(np.float32)
    bout = model.bout.detach().cpu().numpy().astype(np.float32)
    scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)

    out = f"{out_dir}/en_nextword.bin"
    with open(out, "wb") as f:
        f.write(b"TNW1")
        f.write(struct.pack(">iii", V, K, E))
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())
        f.write(w1.astype(">f4").tobytes())
        f.write(b1.astype(">f4").tobytes())
        f.write(bout.astype(">f4").tobytes())
    print(f"wrote {out}")

    # golden vector for the Kotlin test: context ids + expected top ids/logits (quantized path)
    ctx = va_ctx[0].tolist()
    e = q[ctx].astype(np.float32) * scale[ctx, None]
    h = np.maximum(w1 @ e.flatten() + b1, 0)
    hs = max(np.abs(h).max() / 127.0, 1e-8)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)
    logits = (q.astype(np.int32) @ hq) * scale * hs + bout
    top = np.argsort(-logits)[:5]
    json.dump(
        {"context": ctx, "top_ids": top.tolist(), "top_logits": logits[top].tolist()},
        open(f"{out_dir}/golden.json", "w"),
    )
    print("golden:", top.tolist())


if __name__ == "__main__":
    main()
