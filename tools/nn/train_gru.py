#!/usr/bin/env python3
"""Trains a recurrent next-word network and exports it as a TNW5 asset.

Model: tied E-d embeddings -> L-layer GRU (hidden H) -> linear back to E -> logits over the
vocabulary through the tied embedding matrix plus an output bias. The vocabulary product is
therefore the same int8 dot product the phone already runs; only the trunk changes, from a
fixed K-word window to a state carried along the sentence.

Data: one sequence per sentence, inputs [BOS, w1 .. wn], targets [w1 .. wn, BOS] (BOS as the
end-of-sentence target, as the feed-forward trainer does), cut at T tokens, UNK never a target.

Export (en_nextword.bin, big-endian):
  magic 'TNW5'
  int32 V, E, H, L
  emb:   V rows of E int8 (per-row scale = absmax/127), then V float32 scales
  per GRU layer, PyTorch gate order (r, z, n):
         W_ih (3H x in) float32 row-major, W_hh (3H x H), b_ih (3H), b_hh (3H)
  proj:  W (E x H) float32 row-major, b (E)
  bout:  V float32
Inference: h_t = GRU(x_t, h_{t-1}); v = W h_T + b; q = int8(v / vs), vs = absmax(v)/127;
logit_w = scale[w] * vs * dot_int8(emb[w], q) + bout[w].

usage: train_gru.py data_dir out_dir [--steps N] [--dim E] [--hidden H] [--layers L]
                    [--batch B] [--seq T] [--negs N] [--dropout p] [--wd w] [--lr r]
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEP = 0xFFFFFFFF


def sequences(stream, V, T):
    """(N, T) inputs and targets, int32, -1 where there is nothing to predict."""
    BOS, UNK = V - 2, V - 1
    xs, ys = [], []
    sent = []
    for t in stream:
        if t == SEP:
            if sent:
                seq = [BOS] + sent[: T - 1]
                tgt = sent[: T - 1] + [BOS]
                tgt = [-1 if w == UNK else w for w in tgt]
                x = np.full(T, BOS, dtype=np.int32)
                y = np.full(T, -1, dtype=np.int32)
                x[: len(seq)] = seq
                y[: len(tgt)] = tgt
                xs.append(x)
                ys.append(y)
            sent = []
            continue
        sent.append(int(t))
    return np.stack(xs), np.stack(ys)


class NextWordGRU(nn.Module):
    def __init__(self, V, E, H, layers=2, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.gru = nn.GRU(E, H, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(H, E)
        self.bout = nn.Parameter(torch.zeros(V))
        nn.init.normal_(self.emb.weight, std=0.02)

    def trunk(self, x):
        """(B, T) ids -> (B, T, E) vectors that face the vocabulary."""
        h, _ = self.gru(self.emb(x))
        return self.proj(h)

    def forward(self, x):
        return self.trunk(x) @ self.emb.weight.T + self.bout


def export_tnw5(model, out_path):
    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    V, E = emb.shape
    H = model.gru.hidden_size
    L = model.gru.num_layers
    scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)
    with open(out_path, "wb") as f:
        f.write(b"TNW5")
        f.write(struct.pack(">iiii", V, E, H, L))
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())
        for layer in range(L):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                w = getattr(model.gru, f"{name}_l{layer}").detach().cpu().numpy().astype(np.float32)
                f.write(w.astype(">f4").tobytes())
        f.write(model.proj.weight.detach().cpu().numpy().astype(np.float32).astype(">f4").tobytes())
        f.write(model.proj.bias.detach().cpu().numpy().astype(np.float32).astype(">f4").tobytes())
        f.write(model.bout.detach().cpu().numpy().astype(np.float32).astype(">f4").tobytes())
    return {"V": V, "E": E, "H": H, "L": L, "bytes": os.path.getsize(out_path)}


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]

    def opt_arg(name, default, cast):
        return cast(args[args.index(name) + 1]) if name in args else default

    steps = opt_arg("--steps", 30000, int)
    E = opt_arg("--dim", 192, int)
    H = opt_arg("--hidden", 512, int)
    layers = opt_arg("--layers", 2, int)
    B = opt_arg("--batch", 256, int)
    T = opt_arg("--seq", 41, int)
    negs = opt_arg("--negs", 8192, int)
    dropout = opt_arg("--dropout", 0.0, float)
    wd = opt_arg("--wd", 0.01, float)
    lr = opt_arg("--lr", 3e-3, float)

    n_words = sum(1 for _ in open("app/src/main/assets/en_words.txt", encoding="utf-8"))
    V = n_words + 2
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"V={V} E={E} H={H} layers={layers} T={T} steps={steps} device={dev}")

    cache = f"{data_dir}/seq_t{T}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        tr_x, tr_y, va_x, va_y = z["tx"], z["ty"], z["vx"], z["vy"]
        print(f"sequences loaded from cache {cache}")
    else:
        tr_x, tr_y = sequences(np.fromfile(f"{data_dir}/train.bin", dtype="<u4").astype(np.int64), V, T)
        va_x, va_y = sequences(np.fromfile(f"{data_dir}/val.bin", dtype="<u4").astype(np.int64), V, T)
        np.savez(cache, tx=tr_x, ty=tr_y, vx=va_x, vy=va_y)
    print(f"train sentences {len(tr_y)} ({int((tr_y >= 0).sum())} targets), val {len(va_y)}")

    model = NextWordGRU(V, E, H, layers, dropout).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters())} (trunk {sum(p.numel() for n, p in model.named_parameters() if not n.startswith('emb') and n != 'bout')})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)

    tr_x_t = torch.from_numpy(tr_x)
    tr_y_t = torch.from_numpy(tr_y)
    n = len(tr_y_t)
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (B,))
        x = tr_x_t[idx].to(dev).long()
        y = tr_y_t[idx].to(dev).long()
        h = model.trunk(x)                                  # (B, T, E)
        valid = y >= 0
        h = h[valid]                                        # (M, E)
        tgt = y[valid]                                      # (M,)
        # Sampled softmax with shared log-uniform negatives, as the feed-forward trainer:
        # score the true words plus frequent negatives instead of the whole vocabulary.
        neg = (torch.rand(negs, device=dev) * math.log(float(V))).exp().long().clamp(1, V - 1) - 1
        cand = torch.cat([tgt, neg])
        logits = h @ model.emb.weight[cand].T + model.bout[cand]
        logq = -torch.log((cand + 1).float()) - math.log(math.log(float(V)))
        logits = logits - logq.unsqueeze(0)
        hits = tgt.unsqueeze(1) == cand.unsqueeze(0)
        ar = torch.arange(len(tgt), device=dev)
        hits[ar, ar] = False
        logits = logits.masked_fill(hits, -1e9)
        loss = F.cross_entropy(logits, ar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.3f} ({(time.time() - t0):.0f}s)", flush=True)

    # validation: full softmax perplexity and top-1/top-3 over every target position
    model.eval()
    hits = top3 = total = 0
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(va_y), 512):
            x = torch.from_numpy(va_x[i:i + 512]).to(dev).long()
            y = torch.from_numpy(va_y[i:i + 512]).to(dev).long()
            valid = y >= 0
            logits = model.trunk(x)[valid] @ model.emb.weight.T + model.bout
            tgt = y[valid]
            nll += F.cross_entropy(logits, tgt, reduction="sum").item()
            top = logits.topk(3, dim=1).indices
            hits += (top[:, 0] == tgt).sum().item()
            top3 += (top == tgt.unsqueeze(1)).any(dim=1).sum().item()
            total += len(tgt)
    print(f"val ppl {math.exp(nll / total):.1f}  top1 {hits / total:.3f}  top3 {top3 / total:.3f}")

    # ---- export ----------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/en_nextword.bin"
    info = export_tnw5(model, out)
    print(f"wrote {out} ({info['bytes'] / 1e6:.1f} MB, GRU {layers}x{H})")

    # golden vector for the reader tests: a val prefix and the quantized-path top logits
    row = va_x[0]
    length = int((va_y[0] >= 0).sum())
    ctx = [int(w) for w in row[1:length]]          # words after BOS; the reader adds BOS itself
    try:
        import tnw
        net = tnw.read_tnw(out)
        logits = tnw.predict_logits(net, ctx)
    except Exception as e:  # noqa: BLE001 -- the reader may not know TNW5 yet; the asset is still written
        print(f"golden skipped: {type(e).__name__}: {e}")
        return
    top = np.argsort(-logits)[:5]
    json.dump({"context": ctx, "top_ids": top.tolist(), "top_logits": [float(v) for v in logits[top]]},
              open(f"{out_dir}/golden.json", "w"))
    print("golden:", top.tolist())


if __name__ == "__main__":
    main()
