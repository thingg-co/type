"""TNW (NextWord) binary format reader and exporter.

TNW3 format (big-endian):
  magic 'TNW3' (4 bytes)
  int32 V (total vocab incl. BOS+UNK)
  int32 K (context window size)
  int32 E (embedding dimension)
  int32 L (number of trunk layers)
  int8  emb: V rows of E bytes (quantized embeddings)
  float32 scale: V scales (one per row)
  L layers, each:
    int32 out, int32 in (dimensions)
    float32 W: out x in weights (row-major)
    float32 b: out biases
  float32 bout: V output biases

TNW2 (one-layer, no L in header):
  magic 'TNW3' (4 bytes)
  int32 V, int32 K, int32 E (12 bytes)
  same as TNW3 from there...
"""

import struct

import numpy as np


def export_tnw3(model, out_path, K):
    """Export a NextWord model to TNW3 format.

    Args:
        model: torch.nn.Module with emb, trunk (list of Linear+ReLU), and bout
        out_path: output file path
        K: context window size

    Returns:
        dict with V, K, E, L, bytes (file size)
    """
    import torch

    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    lin = [m for m in model.trunk if isinstance(m, torch.nn.Linear)]
    bout = model.bout.detach().cpu().numpy().astype(np.float32)

    # Quantize embedding table per row
    scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)

    L = len(lin)

    with open(out_path, "wb") as f:
        f.write(b"TNW3")
        f.write(struct.pack(">iiii", V := emb.shape[0], K, E := emb.shape[1], L))
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())
        for m in lin:
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack(">ii", w.shape[0], w.shape[1]))
            f.write(w.astype(">f4").tobytes())
            f.write(b.astype(">f4").tobytes())
        f.write(bout.astype(">f4").tobytes())

    # Calculate file size
    layer_bytes = sum(8 + 4 * no * ni + 4 * no for no, ni in [(m.weight.shape[0], m.weight.shape[1]) for m in lin])
    total_bytes = 4 + 16 + V * E + 4 * V + layer_bytes + 4 * V

    return {"V": V, "K": K, "E": E, "L": L, "bytes": total_bytes}


def read_tnw(path):
    """Read a TNW file (TNW1, TNW2, or TNW3).

    TNW2 has L=1 implicitly (no L in header).
    TNW3 has L explicitly in header.

    Args:
        path: file path

    Returns:
        dict with keys:
          - magic: bytes (always b"TNW3")
          - V: int
          - K: int
          - E: int
          - L: int (1 for TNW2)
          - layers: list of (W, b) float32 numpy arrays
          - q: int32 array V x E (quantized embeddings)
          - scale: float32 array V
          - bout: float32 array V
    """
    raw = open(path, "rb").read()
    o = 0

    magic = raw[o:o + 4]
    o += 4

    if magic == b"TNW3":
        V, K, E, L = struct.unpack(">iiii", raw[o:o + 16])
        o += 16
    else:
        # This shouldn't happen with current format, but handle legacy TNW2
        V, K, E = struct.unpack(">iii", raw[o:o + 12])
        o += 12
        L = 1

    # Quantized embedding table (V x E int8)
    q = np.frombuffer(raw[o:o + V * E], dtype=np.int8).reshape(V, E).astype(np.int32)
    o += V * E

    # Per-row scales (V float32)
    scale = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)
    o += 4 * V

    # Trunk layers
    layers = []
    for _ in range(L):
        no, ni = struct.unpack(">ii", raw[o:o + 8])
        o += 8
        w = np.frombuffer(raw[o:o + 4 * no * ni], dtype=">f4").astype(np.float32).reshape(no, ni)
        o += 4 * no * ni
        b = np.frombuffer(raw[o:o + 4 * no], dtype=">f4").astype(np.float32)
        o += 4 * no
        layers.append((w, b))

    # Output bias (V float32)
    bout = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)

    return {
        "magic": magic,
        "V": V,
        "K": K,
        "E": E,
        "L": L,
        "layers": layers,
        "q": q,
        "scale": scale,
        "bout": bout,
    }


def predict_logits(net, ctx_ids):
    """Compute logits for a context using the quantized TNW network.

    Forward pass:
      1. Dequantize embeddings for context
      2. Pass through trunk with ReLU
      3. Quantize trunk output to int8 (hs = absmax/127)
      4. logits[v] = scale[v] * hs * dot_int8(q[v], qh) + bout[v]

    Args:
        net: dict from read_tnw()
        ctx_ids: list or array of context word ids (length K)

    Returns:
        np.ndarray of logits shape (V,)
    """
    q = net["q"]
    scale = net["scale"]
    bout = net["bout"]
    layers = net["layers"]
    K = net["K"]
    BOS = q.shape[0] - 2  # BOS is second-to-last id

    # Left-pad context with BOS
    ctx = ([BOS] * K + list(ctx_ids))[-K:]

    # Dequantize and flatten embeddings
    h = (q[ctx].astype(np.float32) * scale[ctx, None]).flatten()

    # Trunk forward with ReLU
    for w, b in layers:
        h = np.maximum(w @ h + b, 0)

    # Quantize trunk output to int8
    hs = max(np.abs(h).max() / 127.0, 1e-8)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)

    # Compute logits: scale[v] * hs * dot(q[v], hq) + bout[v]
    logits = (q @ hq) * scale * hs + bout

    return logits
