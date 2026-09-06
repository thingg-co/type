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

TNW4 format (untied output embeddings, big-endian):
  magic 'TNW4' (4 bytes)
  int32 V (total vocab incl. BOS+UNK)
  int32 K (context window size)
  int32 E (embedding dimension)
  int32 L (number of trunk layers)
  input emb: V rows of E bytes (quantized embeddings)
  float32 scale: V scales (one per row)
  L layers, each:
    int32 out, int32 in (dimensions)
    float32 W: out x in weights (row-major)
    float32 b: out biases
  output emb: V rows of E bytes (quantized embeddings)
  float32 scale_out: V scales (one per row)
  float32 bout: V output biases

TNW2 (one-layer, no L in header):
  magic 'TNW3' (4 bytes)
  int32 V, int32 K, int32 E (12 bytes)
  same as TNW3 from there...
"""

import struct

import numpy as np


def _export_tnw(model, out_path, K, magic):
    """Internal function to export a NextWord model to TNW format.

    Args:
        model: torch.nn.Module with emb (and optionally out), trunk (list of Linear+ReLU), and bout
        out_path: output file path
        K: context window size
        magic: b"TNW3" for tied, b"TNW4" for untied

    Returns:
        dict with V, K, E, L, bytes (file size)
    """
    import torch

    emb = model.emb.weight.detach().cpu().numpy().astype(np.float32)
    out_emb = getattr(model, "out", None)
    out_emb = out_emb.weight.detach().cpu().numpy().astype(np.float32) if out_emb is not None else None
    lin = [m for m in model.trunk if isinstance(m, torch.nn.Linear)]
    bout = model.bout.detach().cpu().numpy().astype(np.float32)

    # Quantize input embedding table per row
    scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)

    if out_emb is not None:
        # Quantize output embedding table per row
        scale_out = np.maximum(np.abs(out_emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
        q_out = np.clip(np.round(out_emb / scale_out[:, None]), -127, 127).astype(np.int8)
    else:
        scale_out = None
        q_out = None

    L = len(lin)

    with open(out_path, "wb") as f:
        f.write(magic)
        if magic == b"TNW3":
            f.write(struct.pack(">iiii", V := emb.shape[0], K, E := emb.shape[1], L))
        else:  # TNW4
            f.write(struct.pack(">iiii", V := emb.shape[0], K, E := emb.shape[1], L))
        # Input embedding table
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())
        for m in lin:
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack(">ii", w.shape[0], w.shape[1]))
            f.write(w.astype(">f4").tobytes())
            f.write(b.astype(">f4").tobytes())
        # Output embedding table (TNW4 only)
        if out_emb is not None:
            f.write(q_out.tobytes())
            f.write(scale_out.astype(">f4").tobytes())
        f.write(bout.astype(">f4").tobytes())

    # Calculate file size
    layer_bytes = sum(8 + 4 * no * ni + 4 * no for no, ni in [(m.weight.shape[0], m.weight.shape[1]) for m in lin])
    if magic == b"TNW3":
        total_bytes = 4 + 16 + V * E + 4 * V + layer_bytes + 4 * V
    else:  # TNW4
        total_bytes = 4 + 16 + V * E + 4 * V + layer_bytes + V * E + 4 * V + 4 * V

    return {"V": V, "K": K, "E": E, "L": L, "bytes": total_bytes}


def export_tnw3(model, out_path, K):
    """Export a NextWord model to TNW3 format (tied embeddings).

    Args:
        model: torch.nn.Module with emb, trunk (list of Linear+ReLU), and bout
        out_path: output file path
        K: context window size

    Returns:
        dict with V, K, E, L, bytes (file size)
    """
    return _export_tnw(model, out_path, K, b"TNW3")


def export_tnw(model, out_path, K):
    """Export a NextWord model to TNW format.

    Writes TNW3 for tied models (no out attribute) and TNW4 for untied models.

    Args:
        model: torch.nn.Module with emb (and optionally out), trunk, bout
        out_path: output file path
        K: context window size

    Returns:
        dict with V, K, E, L, bytes (file size)
    """
    magic = b"TNW4" if hasattr(model, "out") and model.out is not None else b"TNW3"
    return _export_tnw(model, out_path, K, magic)


def read_tnw(path):
    """Read a TNW file (TNW2, TNW3, or TNW4).

    TNW2 has L=1 implicitly (no L in header).
    TNW3 has L explicitly in header and tied embeddings (q/scale used for vocab).
    TNW4 has L explicitly in header and untied embeddings (q/scale for input, q_out/scale_out for vocab).

    Args:
        path: file path

    Returns:
        dict with keys:
          - magic: bytes (b"TNW3" or b"TNW4")
          - V: int
          - K: int
          - E: int
          - L: int (1 for TNW2)
          - layers: list of (W, b) float32 numpy arrays
          - q: int32 array V x E (quantized input embeddings)
          - scale: float32 array V (input embedding scales)
          - q_out: int32 array V x E (quantized output embeddings, TNW4 only; same as q for tied)
          - scale_out: float32 array V (output embedding scales, TNW4 only; same as scale for tied)
          - bout: float32 array V
    """
    raw = open(path, "rb").read()
    o = 0

    magic = raw[o:o + 4]
    o += 4

    if magic == b"TNW3":
        # TNW3 format with 16-byte header: V, K, E, L (all 4 bytes each)
        V, K, E, L = struct.unpack(">iiii", raw[o:o + 16])
        o += 16
        has_output_emb = False
    elif magic == b"TNW4":
        # TNW4 format with separate output embedding table
        V, K, E, L = struct.unpack(">iiii", raw[o:o + 16])
        o += 16
        has_output_emb = True
    else:
        # This shouldn't happen with current format, but handle legacy TNW2
        V, K, E = struct.unpack(">iii", raw[o:o + 12])
        o += 12
        L = 1
        has_output_emb = False

    # Quantized input embedding table (V x E int8)
    q = np.frombuffer(raw[o:o + V * E], dtype=np.int8).reshape(V, E).astype(np.int32)
    o += V * E

    # Per-row input scales (V float32)
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

    # Output embedding table (TNW4 only, V x E int8)
    if has_output_emb:
        q_out = np.frombuffer(raw[o:o + V * E], dtype=np.int8).reshape(V, E).astype(np.int32)
        o += V * E
        scale_out = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)
        o += 4 * V
    else:
        # Tied model: q_out = q, scale_out = scale (same object for identity)
        q_out = q
        scale_out = scale

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
        "q_out": q_out,
        "scale_out": scale_out,
        "bout": bout,
    }


def predict_logits(net, ctx_ids):
    """Compute logits for a context using the quantized TNW network.

    Forward pass:
      1. Dequantize embeddings for context
      2. Pass through trunk with ReLU
      3. Quantize trunk output to int8 (hs = absmax/127)
      4. logits[v] = scale_out[v] * hs * dot_int8(q_out[v], qh) + bout[v]

    Args:
        net: dict from read_tnw()
        ctx_ids: list or array of context word ids (length K)

    Returns:
        np.ndarray of logits shape (V,)
    """
    q_out = net.get("q_out", net["q"])
    scale_out = net.get("scale_out", net["scale"])
    bout = net["bout"]
    layers = net["layers"]
    K = net["K"]
    BOS = q_out.shape[0] - 2  # BOS is second-to-last id

    # Left-pad context with BOS
    ctx = ([BOS] * K + list(ctx_ids))[-K:]

    # Dequantize and flatten embeddings
    h = (net["q"][ctx].astype(np.float32) * net["scale"][ctx, None]).flatten()

    # Trunk forward with ReLU
    for w, b in layers:
        h = np.maximum(w @ h + b, 0)

    # Quantize trunk output to int8
    hs = max(np.abs(h).max() / 127.0, 1e-8)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)

    # Compute logits: scale_out[v] * hs * dot(q_out[v], hq) + bout[v]
    logits = (q_out @ hq) * scale_out * hs + bout

    return logits


def predict_logits_batch(net, ctx_batch):
    """Compute logits for multiple contexts using the quantized TNW network.

    Batched version that computes trunk outputs for all contexts, then quantizes
    and computes logits via integer matmul.

    Forward pass:
      1. Dequantize embeddings for all contexts
      2. Pass through trunk with ReLU (vectorized)
      3. Quantize trunk output to int8 per row (hs = absmax/127)
      4. logits[v] = scale_out[v] * hs * dot_int8(q_out[v], qh[row]) + bout[v]

    Args:
        net: dict from read_tnw()
        ctx_batch: int array of shape (N, K) where N is batch size, K is context length

    Returns:
        np.ndarray of logits shape (N, V)
    """
    q = net["q"]
    scale = net["scale"]
    q_out = net.get("q_out", q)
    scale_out = net.get("scale_out", scale)
    bout = net["bout"]
    layers = net["layers"]
    K = net["K"]
    V = q.shape[0]
    BOS = V - 2  # BOS is second-to-last id

    N = ctx_batch.shape[0]

    # Left-pad contexts with BOS (N, K) -> (N, K) after padding and trimming
    # For each context, prepend K BOS ids and take last K
    bos_ctx = np.full((N, K), BOS, dtype=np.int64)
    ctx_ids = np.asarray(ctx_batch, dtype=np.int64)
    # Concatenate K BOS at start, then take last K
    padded = np.concatenate([bos_ctx, ctx_ids], axis=1)
    ctx = padded[:, -K:]

    # Dequantize embeddings for all contexts: (N, K, E) * (N, K, 1)
    q_ctx = q[ctx].astype(np.float32)  # (N, K, E)
    scale_ctx = scale[ctx][:, :, None]  # (N, K, 1)
    h = (q_ctx * scale_ctx).reshape(N, -1)  # (N, K*E) flattened

    # Trunk forward with ReLU (vectorized across batch)
    for w, b in layers:
        h = np.maximum(h @ w.T + b, 0)  # (N, out)

    # Quantize trunk output to int8 per row
    # hs[row] = max(abs(h[row]).max() / 127.0, 1e-8)
    abs_max = np.abs(h).max(axis=1, keepdims=True)  # (N, 1)
    hs = np.maximum(abs_max / 127.0, 1e-8)  # (N, 1)

    # hq[row] = clip(round(h[row] / hs[row]), -127, 127)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)  # (N, E)

    # Compute logits for all rows at once:
    # logits[row, v] = scale_out[v] * hs[row] * dot(q_out[v], hq[row]) + bout[v]
    # q_out is (V, E), hq is (N, E)
    # q_out @ hq.T = (V, N) where column row is dot(q_out[:, :], hq[row, :])
    # float32 BLAS instead of numpy's scalar int32 loops: every dot product is a sum of at most
    # E terms of |a*b| <= 127*127, far below 2^24, so the float32 result is the exact integer.
    qf = net.get("_qf_out")
    if qf is None:
        qf = net["_qf_out"] = q_out.astype(np.float32)
    dot_val = qf @ hq.astype(np.float32).T  # (V, N), exact
    logits = (dot_val * scale_out[:, None] * hs.T + bout[:, None]).T  # (N, V)

    return logits
