"""Tests for TNW format reader, writer, and predictor.

These tests ensure the binary format lives in one place (tnw.py) and can be:
- Exported correctly from a tiny PyTorch model
- Read back identically
- Produce logits matching the torch model's forward pass (within quantization tolerance)
- Handle TNW2 (one-layer) files correctly
"""

import os
import struct
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnw


class TinyNextWord(nn.Module):
    """Tiny NextWord model for testing: K=3, E=8, layers=2, hidden=16, V=50."""

    def __init__(self, V=50, E=8, layers=2, hidden=16, K=3):
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


def build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0):
    """Build and return a TinyNextWord model with given seed."""
    torch.manual_seed(seed)
    model = TinyNextWord(V=V, E=E, layers=layers, hidden=hidden, K=K)
    return model


def test_export_read_tnw3_header_and_shapes(tmp_path):
    """Export a tiny model, read it back, verify header and layer shapes."""
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)
    out_path = tmp_path / "test_tnw3.bin"

    # Export
    meta = tnw.export_tnw3(model, str(out_path), K=3)

    # Verify export metadata
    assert meta["V"] == 50
    assert meta["K"] == 3
    assert meta["E"] == 8
    assert meta["L"] == 2

    # Verify file size calculation
    # 4 (magic) + 16 (header V,K,E,L) + V*E (emb) + 4*V (scale) +
    # sum over layers of (8 + 4*out*in + 4*out) + 4*V (bout)
    # Layer 0: 16 x 24 (out=16, in=K*E=24)
    # Layer 1: 8 x 16 (out=E=8, in=hidden=16)
    expected_size = 4 + 16 + 50 * 8 + 4 * 50 + (8 + 4 * 16 * 24 + 4 * 16) + (8 + 4 * 8 * 16 + 4 * 8) + 4 * 50
    assert meta["bytes"] == expected_size

    # Verify actual file size
    actual_size = out_path.stat().st_size
    assert actual_size == expected_size

    # Read back
    net = tnw.read_tnw(str(out_path))

    # Verify header fields
    assert net["magic"] == b"TNW3"
    assert net["V"] == 50
    assert net["K"] == 3
    assert net["E"] == 8
    assert net["L"] == 2

    # Verify layer shapes
    assert len(net["layers"]) == 2
    # Layer 0: 16 x 24, bias 16
    assert net["layers"][0][0].shape == (16, 24)
    assert net["layers"][0][1].shape == (16,)
    # Layer 1: 8 x 16, bias 8
    assert net["layers"][1][0].shape == (8, 16)
    assert net["layers"][1][1].shape == (8,)

    # Verify quantized embedding table
    assert net["q"].shape == (50, 8)
    assert net["q"].dtype == np.int32

    # Verify scale vector
    assert net["scale"].shape == (50,)
    assert net["scale"].dtype == np.float32

    # Verify bout vector
    assert net["bout"].shape == (50,)
    assert net["bout"].dtype == np.float32


def test_predict_logits_matches_torch(tmp_path):
    """Verify predict_logits produces results matching torch model within tolerance."""
    torch.manual_seed(0)
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    # Export
    out_path = tmp_path / "test_logits.bin"
    tnw.export_tnw3(model, str(out_path), K=3)

    # Read back
    net = tnw.read_tnw(str(out_path))

    # Test on 10 random contexts (more samples = more likely to catch quantization issues)
    torch.manual_seed(123)
    matches = 0
    tol_ok = 0

    for _ in range(10):
        ctx = torch.randint(0, 50, (3,)).tolist()

        # Torch forward pass
        with torch.no_grad():
            ctx_t = torch.tensor([ctx])
            torch_logits = model(ctx_t).squeeze(0).numpy()

        # TNW predict_logits
        tnw_logits = tnw.predict_logits(net, ctx)

        # Check argmax match
        torch_top1 = int(np.argmax(torch_logits))
        tnw_top1 = int(np.argmax(tnw_logits))
        if torch_top1 == tnw_top1:
            matches += 1

        # Check top logit within 5% relative tolerance
        torch_top = torch_logits.max()
        tnw_top = tnw_logits.max()
        if torch_top > 0:
            rel_err = abs(tnw_top - torch_top) / torch_top
            if rel_err <= 0.05:
                tol_ok += 1
        elif tnw_top == 0:
            tol_ok += 1  # both zero is fine

    # Assert at least 8 of 10 argmax matches (more lenient for tiny model with close logits)
    assert matches >= 8, f"Only {matches}/10 argmax matches"

    # Assert top logit within 5% for all 10
    assert tol_ok == 10, f"Only {tol_ok}/10 top logits within 5% tolerance"


def test_file_size_formula(tmp_path):
    """Verify file size formula for various model configurations."""
    torch.manual_seed(0)

    # Test case 1: 2-layer model
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)
    out_path = tmp_path / "test_size1.bin"
    meta = tnw.export_tnw3(model, str(out_path), K=3)

    # Manual calculation
    # Layer 0: Linear(K*E, hidden) = Linear(24, 16)
    # Layer 1: Linear(hidden, E) = Linear(16, 8)
    V, K, E, L = 50, 3, 8, 2
    layer0_bytes = 8 + 4 * 16 * 24 + 4 * 16  # out=16, in=K*E=24
    layer1_bytes = 8 + 4 * 8 * 16 + 4 * 8     # out=E=8, in=hidden=16
    expected = 4 + 16 + V * E + 4 * V + layer0_bytes + layer1_bytes + 4 * V
    assert meta["bytes"] == expected

    # Test case 2: 1-layer model (TNW2 style)
    # For layers=1: dims = [K*E, E], so layer is Linear(K*E, E)
    model1 = build_tiny_model(V=40, E=10, layers=1, hidden=16, K=3, seed=0)
    out_path1 = tmp_path / "test_size2.bin"
    meta1 = tnw.export_tnw3(model1, str(out_path1), K=3)

    # Layer 0: Linear(K*E, E) = Linear(30, 10)
    V1, K1, E1, L1 = 40, 3, 10, 1
    layer_bytes1 = 8 + 4 * 10 * 30 + 4 * 10  # out=E=10, in=K*E=30
    expected1 = 4 + 16 + V1 * E1 + 4 * V1 + layer_bytes1 + 4 * V1
    assert meta1["bytes"] == expected1


def test_tnw2_one_layer_still_tnw3_magic(tmp_path):
    """A one-layer export should start with b"TNW3" and have L==1."""
    torch.manual_seed(0)
    model = build_tiny_model(V=50, E=8, layers=1, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_tnw2.bin"
    meta = tnw.export_tnw3(model, str(out_path), K=3)

    # Should be TNW3 magic
    assert meta["L"] == 1

    # Read back and verify L=1
    net = tnw.read_tnw(str(out_path))
    assert net["magic"] == b"TNW3"
    assert net["L"] == 1

    # Verify layer count
    assert len(net["layers"]) == 1


def test_read_tnw2_hand_built(tmp_path):
    """Test read_tnw can parse a hand-built TNW2 file (struct-written).

    TNW2 is just TNW3 with L=1, so we write V,K,E,L (16 bytes) but set L=1.
    """
    V, K, E, L = 20, 3, 4, 1

    # TNW2 format: magic + V,K,E,L (16 bytes), then emb, scale, W,b, bout
    # TNW2 is identical to TNW3 with L=1
    out_path = tmp_path / "hand_tnw2.bin"
    with open(out_path, "wb") as f:
        f.write(b"TNW3")
        f.write(struct.pack(">iiii", V, K, E, L))  # 16-byte header with L=1

        # Build quantized embedding table (20 x 4)
        torch.manual_seed(0)
        emb = torch.randn(V, E).numpy().astype(np.float32)
        scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
        q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())

        # One layer: out=E=4, in=K*E=12
        out_dim, in_dim = E, K * E
        W = torch.randn(out_dim, in_dim).numpy().astype(np.float32)
        b = torch.randn(out_dim).numpy().astype(np.float32)
        f.write(struct.pack(">ii", out_dim, in_dim))
        f.write(W.astype(">f4").tobytes())
        f.write(b.astype(">f4").tobytes())

        # bout
        bout = torch.zeros(V).numpy().astype(np.float32)
        f.write(bout.astype(">f4").tobytes())

    # Read it back
    net = tnw.read_tnw(str(out_path))

    assert net["magic"] == b"TNW3"
    assert net["V"] == V
    assert net["K"] == K
    assert net["E"] == E
    assert net["L"] == L  # read_tnw reads L from header

    assert len(net["layers"]) == 1
    assert net["layers"][0][0].shape == (E, K * E)
    assert net["layers"][0][1].shape == (E,)

    assert net["q"].shape == (V, E)
    assert net["scale"].shape == (V,)
    assert net["bout"].shape == (V,)


def test_tnw3_header_size_formula(tmp_path):
    """Verify the file size formula: 4 + 16 + V*E + 4*V + sum(layer_bytes) + 4*V."""
    torch.manual_seed(0)

    # Create a model with specific dimensions
    V, E, K, L = 100, 16, 5, 3
    hidden_dims = [32, 24, E]  # 3 layers with these outputs

    class CustomNextWord(nn.Module):
        def __init__(self, V, E, K, layers, hidden_dims):
            super().__init__()
            self.emb = nn.Embedding(V, E)
            dims = [K * E] + hidden_dims
            mods = []
            for i in range(len(dims) - 1):
                mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            self.trunk = nn.Sequential(*mods)
            self.bout = nn.Parameter(torch.zeros(V))
            nn.init.normal_(self.emb.weight, std=0.02)

        def forward(self, ctx):
            e = self.emb(ctx).flatten(1)
            return self.trunk(e) @ self.emb.weight.T + self.bout

    model = CustomNextWord(V, E, K, L, hidden_dims)
    torch.manual_seed(42)
    nn.init.normal_(model.emb.weight, std=0.02)

    out_path = tmp_path / "test_var.bin"
    meta = tnw.export_tnw3(model, str(out_path), K)

    # Calculate expected size
    # 4 (magic) + 16 (V,K,E,L) + V*E (emb) + 4*V (scale) + sum(8 + 4*out*in + 4*out) + 4*V (bout)
    expected = 4 + 16 + V * E + 4 * V + 4 * V  # base
    in_size = K * E
    for out_size in hidden_dims:
        expected += 8 + 4 * out_size * in_size + 4 * out_size
        in_size = out_size

    assert meta["bytes"] == expected
    assert meta["bytes"] == out_path.stat().st_size


def test_predict_logits_api_consistency(tmp_path):
    """Verify predict_logits accepts list or numpy array."""
    torch.manual_seed(0)
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_api.bin"
    tnw.export_tnw3(model, str(out_path), K=3)
    net = tnw.read_tnw(str(out_path))

    ctx_list = [1, 2, 3]
    ctx_array = np.array([1, 2, 3])

    logits_list = tnw.predict_logits(net, ctx_list)
    logits_array = tnw.predict_logits(net, ctx_array)

    assert logits_list.shape == (50,)
    assert logits_array.shape == (50,)
    np.testing.assert_array_almost_equal(logits_list, logits_array)


def test_predict_logits_batch_random_contexts(tmp_path):
    """Test predict_logits_batch on 16 random contexts equals per-row loop."""
    torch.manual_seed(0)
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_batch.bin"
    tnw.export_tnw3(model, str(out_path), K=3)
    net = tnw.read_tnw(str(out_path))

    # Generate 16 random contexts
    torch.manual_seed(123)
    ctxs = [torch.randint(0, 50, (3,)).tolist() for _ in range(16)]
    ctx_batch = np.array(ctxs, dtype=np.int64)

    # Per-row loop
    expected = np.stack([tnw.predict_logits(net, c) for c in ctxs])

    # Batched version
    actual = tnw.predict_logits_batch(net, ctx_batch)

    # Check allclose
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    # Check argmax per row
    for i in range(16):
        assert np.argmax(actual[i]) == np.argmax(expected[i]), f"Row {i} argmax mismatch"


def test_predict_logits_batch_single_row(tmp_path):
    """Test predict_logits_batch with batch size 1 equals predict_logits."""
    torch.manual_seed(0)
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_batch1.bin"
    tnw.export_tnw3(model, str(out_path), K=3)
    net = tnw.read_tnw(str(out_path))

    ctx = [1, 2, 3]

    # Single context via predict_logits
    expected = tnw.predict_logits(net, ctx)

    # Single context via predict_logits_batch
    actual = tnw.predict_logits_batch(net, np.array([ctx], dtype=np.int64))

    assert actual.shape == (1, 50)
    np.testing.assert_array_almost_equal(actual[0], expected)


class TinyNextWordUntied(nn.Module):
    """Tiny NextWord model with untied output embeddings for testing."""

    def __init__(self, V=50, E=8, layers=2, hidden=16, K=3):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.out = nn.Embedding(V, E)  # Untied output embeddings
        dims = [K * E] + [hidden] * (layers - 1) + [E]
        mods = []
        for i in range(len(dims) - 1):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.trunk = nn.Sequential(*mods)
        self.bout = nn.Parameter(torch.zeros(V))
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, ctx):
        e = self.emb(ctx).flatten(1)
        h = self.trunk(e)
        return h @ self.out.weight.T + self.bout


def build_tiny_untied_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0):
    """Build and return a TinyNextWordUntied model with given seed."""
    torch.manual_seed(seed)
    model = TinyNextWordUntied(V=V, E=E, layers=layers, hidden=hidden, K=K)
    return model


def test_untied_exports_tnw4(tmp_path):
    """An untied model exports as TNW4 and has q_out different from q."""
    model = build_tiny_untied_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)
    out_path = tmp_path / "test_untied.bin"

    # Export using export_tnw (should auto-detect and use TNW4)
    meta = tnw.export_tnw(model, str(out_path), K=3)

    # Verify it's TNW4
    assert meta["bytes"] > 0

    # Read back and verify q_out is different from q
    net = tnw.read_tnw(str(out_path))

    # Check magic is TNW4
    assert net["magic"] == b"TNW4"

    # q and q_out should be different arrays (different weights)
    assert net["q"].shape == (50, 8)
    assert net["q_out"].shape == (50, 8)
    # They should not be identical (different initializations)
    # Note: with different seeds, they definitely differ; using same seed but
    # different nn.Embedding instances means different values
    assert not np.array_equal(net["q"], net["q_out"])


def test_untied_predict_logits_matches_torch(tmp_path):
    """Verify predict_logits on untied model matches torch model."""
    torch.manual_seed(0)
    model = build_tiny_untied_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    # Export
    out_path = tmp_path / "test_untied_logits.bin"
    tnw.export_tnw(model, str(out_path), K=3)

    # Read back
    net = tnw.read_tnw(str(out_path))

    # Test on 10 random contexts
    torch.manual_seed(123)
    matches = 0

    for _ in range(10):
        ctx = torch.randint(0, 50, (3,)).tolist()

        # Torch forward pass
        with torch.no_grad():
            ctx_t = torch.tensor([ctx])
            torch_logits = model(ctx_t).squeeze(0).numpy()

        # TNW predict_logits
        tnw_logits = tnw.predict_logits(net, ctx)

        # Check argmax match
        torch_top1 = int(np.argmax(torch_logits))
        tnw_top1 = int(np.argmax(tnw_logits))
        if torch_top1 == tnw_top1:
            matches += 1

    # Assert at least 8 of 10 argmax matches
    assert matches >= 8, f"Only {matches}/10 argmax matches"


def test_tied_tnw3_compatibility(tmp_path):
    """A tied model exports identically via export_tnw3 and export_tnw."""
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path1 = tmp_path / "test_tnw3.bin"
    out_path2 = tmp_path / "test_tnw.bin"

    # Export with both methods
    meta1 = tnw.export_tnw3(model, str(out_path1), K=3)
    meta2 = tnw.export_tnw(model, str(out_path2), K=3)

    # Files should be identical
    with open(out_path1, "rb") as f1, open(out_path2, "rb") as f2:
        assert f1.read() == f2.read()


def test_read_tnw3_returns_q_out_equals_q(tmp_path):
    """read_tnw on a TNW3 file returns q_out is q (same object) and scale_out equals scale."""
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_tied.bin"
    tnw.export_tnw3(model, str(out_path), K=3)

    net = tnw.read_tnw(str(out_path))

    # For tied models, q_out should be the same object as q
    assert net["q_out"] is net["q"]
    assert net["scale_out"] is net["scale"]


def test_untied_predict_logits_batch_equals_rowwise(tmp_path):
    """Test predict_logits_batch on an untied net equals per-row predict_logits."""
    torch.manual_seed(0)
    model = build_tiny_untied_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0)

    out_path = tmp_path / "test_untied_batch.bin"
    tnw.export_tnw(model, str(out_path), K=3)
    net = tnw.read_tnw(str(out_path))

    # Generate 16 random contexts
    torch.manual_seed(123)
    ctxs = [torch.randint(0, 50, (3,)).tolist() for _ in range(16)]
    ctx_batch = np.array(ctxs, dtype=np.int64)

    # Per-row loop
    expected = np.stack([tnw.predict_logits(net, c) for c in ctxs])

    # Batched version
    actual = tnw.predict_logits_batch(net, ctx_batch)

    # Check allclose
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    # Check argmax per row
    for i in range(16):
        assert np.argmax(actual[i]) == np.argmax(expected[i]), f"Row {i} argmax mismatch"


def _build_tnw5(V, E, H, L, seed=0, path=None):
    """Build a TNW5 format file by hand using struct.

    TNW5 layout (big-endian):
      magic "TNW5"; ">iiii" V, E, H, L
      input table: V*E int8 rows (per-row scale = absmax/127), then V float32 scales
      per GRU layer l = 0..L-1, PyTorch gate order (r, z, n):
        W_ih (3H x in) float32 row-major, W_hh (3H x H), b_ih (3H), b_hh (3H)
      proj: W (E x H) float32 row-major, b (E)
      bout: V float32
    """
    import torch
    import tempfile
    torch.manual_seed(seed)

    if path is None:
        path = tempfile.mktemp(suffix=".bin")
    out_path = path

    with open(out_path, "wb") as f:
        f.write(b"TNW5")
        f.write(struct.pack(">iiii", V, E, H, L))

        # Input embedding table
        torch.manual_seed(seed + 1)
        emb = torch.randn(V, E).numpy().astype(np.float32)
        scale = np.maximum(np.abs(emb).max(axis=1) / 127.0, 1e-8).astype(np.float32)
        q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)
        f.write(q.tobytes())
        f.write(scale.astype(">f4").tobytes())

        # GRU layers
        torch.manual_seed(seed + 2)
        in_size = E
        for l in range(L):
            # W_ih: 3H x in
            W_ih = torch.randn(3 * H, in_size).numpy().astype(np.float32)
            # W_hh: 3H x H
            W_hh = torch.randn(3 * H, H).numpy().astype(np.float32)
            # b_ih: 3H
            b_ih = torch.randn(3 * H).numpy().astype(np.float32)
            # b_hh: 3H
            b_hh = torch.randn(3 * H).numpy().astype(np.float32)

            f.write(W_ih.astype(">f4").tobytes())
            f.write(W_hh.astype(">f4").tobytes())
            f.write(b_ih.astype(">f4").tobytes())
            f.write(b_hh.astype(">f4").tobytes())

            in_size = H

        # Projection: E x H
        torch.manual_seed(seed + 3)
        proj_w = torch.randn(E, H).numpy().astype(np.float32)
        proj_b = torch.randn(E).numpy().astype(np.float32)
        f.write(proj_w.astype(">f4").tobytes())
        f.write(proj_b.astype(">f4").tobytes())

        # bout: V
        bout = torch.zeros(V).numpy().astype(np.float32)
        f.write(bout.astype(">f4").tobytes())

    return out_path, emb, scale, q, W_ih, W_hh, b_ih, b_hh, proj_w, proj_b


def test_read_tnw5_header_and_shapes(tmp_path):
    """Read a hand-built TNW5 file and verify header and GRU layer shapes."""
    V, E, H, L = 50, 8, 6, 2
    out_path, _, _, _, _, _, _, _, _, _ = _build_tnw5(V, E, H, L, seed=0)

    net = tnw.read_tnw(str(out_path))

    # Verify magic and kind
    assert net["magic"] == b"TNW5"
    assert net["kind"] == "gru"

    # Verify header fields
    assert net["V"] == V
    assert net["K"] is None  # GRU doesn't use K
    assert net["E"] == E
    assert net["H"] == H
    assert net["L"] == L

    # Verify input embedding table
    assert net["q"].shape == (V, E)
    assert net["q"].dtype == np.int32
    assert net["scale"].shape == (V,)
    assert net["scale"].dtype == np.float32

    # Verify GRU layers
    assert len(net["gru"]) == L
    # Layer 0: in_size = E = 8, out = H = 6
    assert net["gru"][0]["w_ih"].shape == (3 * H, E)  # (18, 8)
    assert net["gru"][0]["w_hh"].shape == (3 * H, H)  # (18, 6)
    assert net["gru"][0]["b_ih"].shape == (3 * H,)    # (18,)
    assert net["gru"][0]["b_hh"].shape == (3 * H,)    # (18,)
    # Layer 1: in_size = H = 6, out = H = 6
    assert net["gru"][1]["w_ih"].shape == (3 * H, H)  # (18, 6)
    assert net["gru"][1]["w_hh"].shape == (3 * H, H)  # (18, 6)
    assert net["gru"][1]["b_ih"].shape == (3 * H,)    # (18,)
    assert net["gru"][1]["b_hh"].shape == (3 * H,)    # (18,)

    # Verify projection
    assert net["proj_w"].shape == (E, H)  # (8, 6)
    assert net["proj_b"].shape == (E,)    # (8,)

    # Verify bout
    assert net["bout"].shape == (V,)
    assert net["bout"].dtype == np.float32


def test_predict_logits_gru_matches_torch(tmp_path):
    """Verify predict_logits on GRU net matches torch.nn.GRU within tolerance."""
    V, E, H, L = 50, 8, 6, 2
    out_path, emb, scale, q, W_ih, W_hh, b_ih, b_hh, proj_w, proj_b = _build_tnw5(V, E, H, L, seed=0)

    net = tnw.read_tnw(str(out_path))

    # Build equivalent torch GRU
    torch.manual_seed(0)
    gru = torch.nn.GRU(E, H, L, batch_first=True)

    # Copy weights (PyTorch uses layer-specific parameter names weight_ih_l{l}, weight_hh_l{l})
    with torch.no_grad():
        for l in range(L):
            w_ih = torch.tensor(net["gru"][l]["w_ih"], dtype=torch.float32)
            w_hh = torch.tensor(net["gru"][l]["w_hh"], dtype=torch.float32)
            b_ih = torch.tensor(net["gru"][l]["b_ih"], dtype=torch.float32)
            b_hh = torch.tensor(net["gru"][l]["b_hh"], dtype=torch.float32)

            setattr(gru, f"weight_ih_l{l}", torch.nn.Parameter(w_ih))
            setattr(gru, f"weight_hh_l{l}", torch.nn.Parameter(w_hh))
            setattr(gru, f"bias_ih_l{l}", torch.nn.Parameter(b_ih))
            setattr(gru, f"bias_hh_l{l}", torch.nn.Parameter(b_hh))

    # Project layer
    proj = torch.nn.Linear(H, E, bias=True)
    with torch.no_grad():
        proj.weight.data.copy_(torch.tensor(proj_w, dtype=torch.float32))
        proj.bias.data.copy_(torch.tensor(proj_b, dtype=torch.float32))

    # Test on multiple contexts
    torch.manual_seed(123)
    matches = 0
    tol_ok = 0

    for _ in range(10):
        ctx = torch.randint(0, V - 2, (3,)).tolist()

        # Torch forward pass
        with torch.no_grad():
            x_ids = torch.tensor([V - 2] + ctx, dtype=torch.long)
            emb_layer = torch.nn.Embedding(V, E)
            emb_layer.weight.data.copy_(torch.tensor(emb, dtype=torch.float32))
            x = emb_layer(x_ids).unsqueeze(0)
            h_t, _ = gru(x)
            h_last = h_t[:, -1, :]
            v = proj(h_last).squeeze(0)

        # TNW predict_logits
        tnw_logits = tnw.predict_logits(net, ctx)

        # Torch logits: v @ emb.T + bout
        torch_logits = v @ torch.tensor(emb, dtype=torch.float32).T

        # Check argmax match
        torch_top1 = int(np.argmax(torch_logits.numpy()))
        tnw_top1 = int(np.argmax(tnw_logits))
        if torch_top1 == tnw_top1:
            matches += 1

        # Check logits allclose
        # GRU with quantized int8 embeddings has ~0.1-0.4 quantization error in logits
        tnw_logits_f32 = torch.tensor(tnw_logits, dtype=torch.float32)
        if torch.allclose(torch_logits, tnw_logits_f32, atol=0.5):
            tol_ok += 1

    assert matches >= 8, f"Only {matches}/10 argmax matches"
    assert tol_ok >= 8, f"Only {tol_ok}/10 logits allclose"


def test_predict_logits_batch_gru_equals_rowwise(tmp_path):
    """Test predict_logits_batch on GRU net equals per-row predict_logits."""
    V, E, H, L = 50, 8, 6, 2
    out_path, _, _, _, _, _, _, _, _, _ = _build_tnw5(V, E, H, L, seed=0)

    net = tnw.read_tnw(str(out_path))

    # Generate 12 contexts of mixed lengths (0 to 5 ids)
    torch.manual_seed(123)
    ctxs = [torch.randint(0, V - 2, (np.random.randint(0, 6),)).tolist() for _ in range(12)]

    # Per-row loop
    expected = np.stack([tnw.predict_logits(net, c) for c in ctxs])

    # Batched version
    max_len = max(len(c) for c in ctxs)
    ctx_batch = np.zeros((12, max_len), dtype=np.int64)
    ctx_lengths = []
    for i, c in enumerate(ctxs):
        if len(c) > 0:
            ctx_batch[i, :len(c)] = c
        ctx_lengths.append(len(c))

    actual = tnw.predict_logits_batch(net, ctx_batch, ctx_lengths)

    # Check allclose
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    # Check argmax per row
    for i in range(12):
        assert np.argmax(actual[i]) == np.argmax(expected[i]), f"Row {i} argmax mismatch"


def test_tnw5_file_size_formula(tmp_path):
    """Verify TNW5 file size formula.

    Size = 4 + 16 + V*E + 4*V +
           sum over layers of (4*(3H*in + 3H*H + 6H)) +
           4*(E*H + E) + 4*V
    """
    V, E, H, L = 50, 8, 6, 2
    out_path, _, _, _, _, _, _, _, _, _ = _build_tnw5(V, E, H, L, seed=0)

    expected = 4 + 16
    expected += V * E
    expected += 4 * V

    in_size = E
    for l in range(L):
        expected += 4 * (3 * H * in_size + 3 * H * H + 6 * H)
        in_size = H

    expected += 4 * (E * H + E)
    expected += 4 * V

    actual = os.path.getsize(out_path)
    assert actual == expected, f"Expected {expected}, got {actual}"
