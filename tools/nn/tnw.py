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

TNW5 format (GRU recurrent trunk, big-endian):
  magic 'TNW5' (4 bytes)
  int32 V (total vocab incl. BOS+UNK)
  int32 E (embedding dimension)
  int32 H (hidden size per GRU layer)
  int32 L (number of GRU layers)
  input emb: V rows of E bytes (quantized embeddings)
  float32 scale: V scales (one per row)
  L GRU layers, each with PyTorch gate order (r, z, n):
    float32 W_ih: 3H x in weights (row-major)
    float32 W_hh: 3H x H weights (row-major)
    float32 b_ih: 3H biases
    float32 b_hh: 3H biases
    where in = E for layer 0, H for subsequent layers
  projection: W_proj: E x H weights, b_proj: E biases
  float32 bout: V output biases

TNW2 (one-layer, no L in header):
  magic 'TNW3' (4 bytes)
  int32 V, int32 K, int32 E (12 bytes)
  same as TNW3 from there...
"""

import struct

import numpy as np


def _gru_cell(x, h_prev, W_ih, W_hh, b_ih, b_hh):
    """Single GRU cell forward pass (PyTorch semantics).

    Args:
        x: input (N, E)
        h_prev: previous hidden state (N, H)
        W_ih: input-to-hidden weights (3H, E)
        W_hh: hidden-to-hidden weights (3H, H)
        b_ih: input bias (3H,)
        b_hh: hidden bias (3H,)

    Returns:
        h_new: new hidden state (N, H)
    """
    # Linear projections
    # x_proj = x @ W_ih.T + b_ih  # (N, 3H)
    # h_proj = h_prev @ W_hh.T + b_hh  # (N, 3H)
    # PyTorch uses row-major weight matrices, so we do W @ x (no transpose)
    x_proj = x @ W_ih.T + b_ih  # (N, 3H)
    h_proj = h_prev @ W_hh.T + b_hh  # (N, 3H)

    # Split into r, z, n gates (each H elements)
    r_x, z_x, n_x = np.split(x_proj, 3, axis=-1)  # each (N, H)
    r_h, z_h, n_h = np.split(h_proj, 3, axis=-1)  # each (N, H)

    # Gates with PyTorch semantics
    r = 1.0 / (1.0 + np.exp(-(r_x + r_h)))  # sigmoid
    z = 1.0 / (1.0 + np.exp(-(z_x + z_h)))  # sigmoid
    n = np.tanh(n_x + r * (n_h))  # tanh with reset applied to hidden

    # New hidden state
    h_new = (1.0 - z) * n + z * h_prev
    return h_new


def _gru_forward(x_seq, gru_layers):
    """Forward pass through GRU layers.

    Args:
        x_seq: input sequence (N, T, E) where T is sequence length
        gru_layers: list of dicts with w_ih, w_hh, b_ih, b_hh

    Returns:
        h_seq: output sequence (N, T, H) - hidden states for all timesteps
    """
    N, T, _ = x_seq.shape
    L = len(gru_layers)

    # Hidden states for all layers and timesteps
    # H is determined from w_hh shape (3H, H) since that's consistent across layers
    H = gru_layers[0]["w_hh"].shape[1]
    h = np.zeros((L + 1, N, T, H), dtype=np.float32)

    for l, layer in enumerate(gru_layers):
        w_ih = layer["w_ih"]  # (3H, in)
        w_hh = layer["w_hh"]  # (3H, H)
        b_ih = layer["b_ih"]  # (3H,)
        b_hh = layer["b_hh"]  # (3H,)

        in_size = w_ih.shape[1]
        if l == 0:
            x_l = x_seq  # First layer: input is the embedding sequence
        else:
            x_l = h[l - 1]  # Subsequent layers: input is previous layer's output

        for t in range(T):
            h[l, :, t, :] = _gru_cell(
                x_l[:, t, :],
                h[l, :, t - 1, :] if t > 0 else np.zeros((N, w_ih.shape[0] // 3), dtype=np.float32),
                w_ih, w_hh, b_ih, b_hh
            )

    return h[L - 1]  # Return top layer's hidden states


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
    """Read a TNW file (TNW2, TNW3, TNW4, or TNW5).

    TNW2 has L=1 implicitly (no L in header).
    TNW3 has L explicitly in header and tied embeddings (q/scale used for vocab).
    TNW4 has L explicitly in header and untied embeddings (q/scale for input, q_out/scale_out for vocab).
    TNW5 has GRU recurrent trunk (kind "gru" in output).

    Args:
        path: file path

    Returns:
        dict with keys:
          - magic: bytes (b"TNW3", b"TNW4", or b"TNW5")
          - kind: str ("ffn" for TNW2/3/4, "gru" for TNW5)
          - V: int
          - K: int (None for TNW5, not applicable)
          - E: int
          - H: int (hidden size, None for TNW2/3/4)
          - L: int (1 for TNW2)
          - layers: list of (W, b) float32 numpy arrays (FFN layers)
          - gru: list of dicts per layer (GRU layers, TNW5 only)
          - proj_w: E x H float32 array (projection, TNW5 only)
          - proj_b: E float32 array (projection bias, TNW5 only)
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
        kind = "ffn"
        H = None
    elif magic == b"TNW4":
        # TNW4 format with separate output embedding table
        V, K, E, L = struct.unpack(">iiii", raw[o:o + 16])
        o += 16
        has_output_emb = True
        kind = "ffn"
        H = None
    elif magic == b"TNW5":
        # TNW5 format: V, E, H, L (no K - GRU takes variable-length sequences)
        V, E, H, L = struct.unpack(">iiii", raw[o:o + 16])
        o += 16
        has_output_emb = False
        kind = "gru"
    else:
        # This shouldn't happen with current format, but handle legacy TNW2
        V, K, E = struct.unpack(">iii", raw[o:o + 12])
        o += 12
        L = 1
        has_output_emb = False
        kind = "ffn"
        H = None

    # Quantized input embedding table (V x E int8)
    q = np.frombuffer(raw[o:o + V * E], dtype=np.int8).reshape(V, E).astype(np.int32)
    o += V * E

    # Per-row input scales (V float32)
    scale = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)
    o += 4 * V

    if kind == "gru":
        # GRU layers with PyTorch gate order (r, z, n)
        gru = []
        for _ in range(L):
            in_size = E if len(gru) == 0 else H
            # W_ih: 3H x in_size
            w_ih = np.frombuffer(raw[o:o + 4 * 3 * H * in_size], dtype=">f4").astype(np.float32).reshape(3 * H, in_size)
            o += 4 * 3 * H * in_size
            # W_hh: 3H x H
            w_hh = np.frombuffer(raw[o:o + 4 * 3 * H * H], dtype=">f4").astype(np.float32).reshape(3 * H, H)
            o += 4 * 3 * H * H
            # b_ih: 3H
            b_ih = np.frombuffer(raw[o:o + 4 * 3 * H], dtype=">f4").astype(np.float32)
            o += 4 * 3 * H
            # b_hh: 3H
            b_hh = np.frombuffer(raw[o:o + 4 * 3 * H], dtype=">f4").astype(np.float32)
            o += 4 * 3 * H
            gru.append({
                "w_ih": w_ih,
                "w_hh": w_hh,
                "b_ih": b_ih,
                "b_hh": b_hh,
            })

        # Projection: W (E x H) and b (E)
        proj_w = np.frombuffer(raw[o:o + 4 * E * H], dtype=">f4").astype(np.float32).reshape(E, H)
        o += 4 * E * H
        proj_b = np.frombuffer(raw[o:o + 4 * E], dtype=">f4").astype(np.float32)
        o += 4 * E

        # Output bias (V float32)
        bout = np.frombuffer(raw[o:o + 4 * V], dtype=">f4").astype(np.float32)

        return {
            "magic": magic,
            "kind": kind,
            "V": V,
            "K": None,
            "E": E,
            "H": H,
            "L": L,
            "layers": [],  # FFN layers empty for GRU
            "gru": gru,
            "proj_w": proj_w,
            "proj_b": proj_b,
            "q": q,
            "scale": scale,
            "bout": bout,
        }
    else:
        # FFN layers (TNW2/3/4)
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
            "kind": kind,
            "V": V,
            "K": K,
            "E": E,
            "H": H,
            "L": L,
            "layers": layers,
            "gru": [],
            "proj_w": None,
            "proj_b": None,
            "q": q,
            "scale": scale,
            "q_out": q_out,
            "scale_out": scale_out,
            "bout": bout,
        }


def predict_logits(net, ctx_ids):
    """Compute logits for a context using the quantized TNW network.

    Forward pass for FFN:
      1. Dequantize embeddings for context
      2. Pass through trunk with ReLU
      3. Quantize trunk output to int8 (hs = absmax/127)
      4. logits[v] = scale_out[v] * hs * dot_int8(q_out[v], qh) + bout[v]

    Forward pass for GRU (TNW5):
      1. BOS is added internally (V - 2)
      2. Dequantize embeddings for context + BOS
      3. Run GRU layers with zero initial state
      4. Take last hidden state, project to E dims
      5. Compute quantized vocabulary product

    Args:
        net: dict from read_tnw()
        ctx_ids: list or array of context word ids (length K for FFN, any for GRU)

    Returns:
        np.ndarray of logits shape (V,)
    """
    if net["kind"] == "gru":
        return _predict_logits_gru(net, ctx_ids)
    else:
        return _predict_logits_ffn(net, ctx_ids)


def _predict_logits_ffn(net, ctx_ids):
    """Compute logits for FFN network."""
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


def _predict_logits_gru(net, ctx_ids):
    """Compute logits for GRU network (TNW5)."""
    q = net["q"]
    scale = net["scale"]
    bout = net["bout"]
    gru = net["gru"]
    proj_w = net["proj_w"]
    proj_b = net["proj_b"]
    E = net["E"]
    BOS = q.shape[0] - 2  # BOS is second-to-last id

    # Build input sequence: [BOS] + ctx_ids
    ctx_ids = list(ctx_ids)
    x_ids = [BOS] + ctx_ids

    # Dequantize embeddings for the sequence
    x_emb = q[x_ids].astype(np.float32) * scale[x_ids, None]  # (T, E)

    # Run GRU layers
    # gru_forward expects (N, T, E) where N is batch size
    h_seq = _gru_forward(x_emb[np.newaxis, :, :], gru)  # (1, T, H)

    # Take last hidden state
    h_last = h_seq[0, -1, :]  # (H,)

    # Project to E dimensions: v = W_proj @ h + b_proj
    v = h_last @ proj_w.T + proj_b  # (E,)

    # Quantize the projected vector
    vs = max(np.abs(v).max() / 127.0, 1e-8)
    vq = np.clip(np.round(v / vs), -127, 127).astype(np.int32)

    # Compute logits: scale[v] * vs * dot(q[v], vq) + bout[v]
    logits = (q.astype(np.float32) @ vq) * scale * vs + bout

    return logits


def predict_logits_batch(net, ctx_batch, ctx_lengths=None):
    """Compute logits for multiple contexts using the quantized TNW network.

    Batched version that computes trunk outputs for all contexts, then quantizes
    and computes logits via integer matmul.

    Forward pass for FFN:
      1. Dequantize embeddings for all contexts
      2. Pass through trunk with ReLU (vectorized)
      3. Quantize trunk output to int8 per row (hs = absmax/127)
      4. logits[v] = scale_out[v] * hs * dot_int8(q_out[v], qh[row]) + bout[v]

    Forward pass for GRU (TNW5):
      1. Group contexts by length for efficient batching
      2. Run GRU layers for each group with zero initial state
      3. Project last hidden state and compute quantized vocabulary product

    Args:
        net: dict from read_tnw()
        ctx_batch: int array of shape (N, K) where N is batch size, K is context length
                   For GRU, K is the maximum context length; shorter contexts are padded.
        ctx_lengths: optional list of actual context lengths (before padding).
                     For GRU, if None, assumes full row is context (no padding).

    Returns:
        np.ndarray of logits shape (N, V)
    """
    if net["kind"] == "gru":
        return _predict_logits_batch_gru(net, ctx_batch, ctx_lengths)
    else:
        return _predict_logits_batch_ffn(net, ctx_batch)


def _predict_logits_batch_ffn(net, ctx_batch):
    """Compute logits for FFN network in batch."""
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


def _predict_logits_batch_gru(net, ctx_batch, ctx_lengths=None):
    """Compute logits for GRU network in batch.

    Groups contexts by length to run efficient batched GRU forward passes.

    Args:
        net: dict from read_tnw()
        ctx_batch: int array of shape (N, K) where N is batch size, K is max context length
        ctx_lengths: optional list of actual context lengths (before padding).
                     If None, assumed to be the full row length (no padding).
    """
    q = net["q"]
    scale = net["scale"]
    bout = net["bout"]
    gru = net["gru"]
    proj_w = net["proj_w"]
    proj_b = net["proj_b"]
    V = q.shape[0]
    BOS = V - 2  # BOS is second-to-last id
    H = net["H"]

    N = ctx_batch.shape[0]
    K = ctx_batch.shape[1]

    # Determine actual context lengths for each row
    # If ctx_lengths is provided, use it; otherwise assume full row is context
    if ctx_lengths is None:
        ctx_lengths = [K] * N

    # For each context, we need [BOS] + context (variable length)
    # Group by actual sequence length (including BOS)
    groups = {}  # length -> list of (row_idx, ctx_ids)
    for i in range(N):
        actual_len = ctx_lengths[i]
        ctx = list(ctx_batch[i, :actual_len])
        # Actual sequence length is 1 (BOS) + len(ctx)
        seq_len = 1 + len(ctx)
        if seq_len not in groups:
            groups[seq_len] = []
        groups[seq_len].append((i, ctx))

    # Initialize output
    logits_out = np.zeros((N, V), dtype=np.float32)

    # Process each group
    for seq_len, items in groups.items():
        batch_size = len(items)

        # Build input: [BOS] + ctx for each item
        x_ids = np.full((batch_size, seq_len), BOS, dtype=np.int64)
        for j, (row_idx, ctx) in enumerate(items):
            if len(ctx) > 0:
                x_ids[j, 1:] = ctx

        # Dequantize embeddings
        x_emb = q[x_ids].astype(np.float32) * scale[x_ids, None]  # (B, T, E)

        # Run GRU
        h_seq = _gru_forward(x_emb, gru)  # (B, T, H)

        # Take last hidden state
        h_last = h_seq[:, -1, :]  # (B, H)

        # Project: v = W_proj @ h + b_proj
        v = h_last @ proj_w.T + proj_b  # (B, E)

        # Quantize each row
        vs = np.maximum(np.abs(v).max(axis=1, keepdims=True) / 127.0, 1e-8)  # (B, 1)
        vq = np.clip(np.round(v / vs), -127, 127).astype(np.int32)  # (B, E)

        # Compute logits for each row
        for j, (row_idx, _) in enumerate(items):
            vq_row = vq[j]
            vs_row = vs[j, 0]
            logits_row = (q.astype(np.float32) @ vq_row) * scale * vs_row + bout
            logits_out[row_idx] = logits_row

    return logits_out
