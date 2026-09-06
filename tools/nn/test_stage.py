"""Tests for stage.py - promoting a trained run into the app."""

import json
import os
import shutil
import sys

import numpy as np
import pytest

# Add tools/nn to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnw


def build_tiny_model(V=50, E=8, layers=2, hidden=16, K=3, seed=0):
    """Build a tiny NextWord model for testing."""
    import torch
    import torch.nn as nn

    class TinyNextWord(nn.Module):
        def __init__(self, V, E, layers, hidden, K):
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

    torch.manual_seed(seed)
    return TinyNextWord(V=V, E=E, layers=layers, hidden=hidden, K=K)


def export_run_dir(tmp_path, K=3, seed=0):
    """Export a tiny model to a run directory with en_nextword.bin and golden.json."""
    model = build_tiny_model(V=50, E=8, layers=2, hidden=16, K=K, seed=seed)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Export the model
    bin_path = run_dir / "en_nextword.bin"
    tnw.export_tnw3(model, str(bin_path), K)

    # Build golden.json the same way train.py does
    net = tnw.read_tnw(str(bin_path))
    # Use a sample context (first context from validation would be ideal,
    # but we'll use a fixed one for determinism)
    ctx = [1, 2, 3]
    h = (net["q"][ctx].astype(np.float32) * net["scale"][ctx, None]).flatten()
    for w, b in net["layers"]:
        h = np.maximum(w @ h + b, 0)
    hs = max(np.abs(h).max() / 127.0, 1e-8)
    hq = np.clip(np.round(h / hs), -127, 127).astype(np.int32)
    q_out = net.get("q_out", net["q"])
    scale_out = net.get("scale_out", net["scale"])
    logits = (q_out @ hq) * scale_out * hs + net["bout"]
    top = np.argsort(-logits)[:5]
    golden = {
        "context": ctx,
        "top_ids": top.tolist(),
        "top_logits": logits[top].tolist(),
    }
    golden_path = run_dir / "golden.json"
    golden_path.write_text(json.dumps(golden))

    return run_dir, golden, ctx


def test_stage_golden_check_passes(tmp_path, monkeypatch):
    """Test that a valid run passes the golden check."""
    # Create a run dir with valid golden
    run_dir, golden, ctx = export_run_dir(tmp_path)

    # Create a words file with matching V (50 = 48 + 2)
    words_file = tmp_path / "words.txt"
    words_file.write_text("\n".join(f"word{i}" for i in range(48)))

    # Mock shutil.copy2 to track calls
    copy_calls = []
    original_copy2 = shutil.copy2

    def mock_copy2(src, dst):
        copy_calls.append((src, dst))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        original_copy2(src, dst)

    monkeypatch.setattr(shutil, "copy2", mock_copy2)

    # Run stage.py with --words pointing to our test words file
    from tools.nn import stage

    monkeypatch.setattr("sys.argv", ["stage.py", str(run_dir), f"--words={words_file}", f"--app-dir={tmp_path / 'app'}", f"--out-dir={tmp_path / 'out'}"])
    stage.main()

    # Verify golden was accepted (printed "golden ok")
    # Verify files were copied (3 copies: 2 for bin, 1 for golden)
    assert len(copy_calls) == 3


def test_stage_tampered_golden_exits_2(tmp_path, monkeypatch):
    """Test that a tampered golden.json causes exit code 2."""
    run_dir, golden, ctx = export_run_dir(tmp_path)

    # Tamper with top_ids (reverse them)
    golden["top_ids"] = golden["top_ids"][::-1]
    (run_dir / "golden.json").write_text(json.dumps(golden))

    # Create a words file with matching V
    words_file = tmp_path / "words.txt"
    words_file.write_text("\n".join(f"word{i}" for i in range(48)))

    # Run stage.py - should exit 2
    from tools.nn import stage

    monkeypatch.setattr("sys.argv", ["stage.py", str(run_dir), f"--words={words_file}", f"--app-dir={tmp_path / 'app'}", f"--out-dir={tmp_path / 'out'}"])
    with pytest.raises(SystemExit) as exc_info:
        stage.main()
    assert exc_info.value.code == 2


def test_stage_dry_run_copies_nothing(tmp_path, monkeypatch):
    """Test that --dry-run doesn't copy any files."""
    run_dir, golden, ctx = export_run_dir(tmp_path)

    # Create a words file with matching V
    words_file = tmp_path / "words.txt"
    words_file.write_text("\n".join(f"word{i}" for i in range(48)))

    # Mock shutil.copy2 to track calls
    copy_calls = []

    def mock_copy2(src, dst):
        copy_calls.append((src, dst))
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    monkeypatch.setattr(shutil, "copy2", mock_copy2)

    # Run stage.py with --dry-run
    from tools.nn import stage

    monkeypatch.setattr("sys.argv", ["stage.py", str(run_dir), f"--words={words_file}", "--dry-run", f"--app-dir={tmp_path / 'app'}", f"--out-dir={tmp_path / 'out'}"])
    stage.main()

    # Verify no files were copied
    assert len(copy_calls) == 0


def test_stage_vocab_mismatch_exits_3(tmp_path, monkeypatch):
    """Test that vocabulary mismatch causes exit code 3."""
    run_dir, golden, ctx = export_run_dir(tmp_path)

    # Create a words file with a different line count than expected
    # Our model has V=50, so we need 48 words. Let's use 10.
    words_file = tmp_path / "words.txt"
    words_file.write_text("\n".join(f"word{i}" for i in range(10)))

    # Run stage.py - should exit 3
    from tools.nn import stage

    monkeypatch.setattr("sys.argv", ["stage.py", str(run_dir), f"--words={words_file}", f"--app-dir={tmp_path / 'app'}", f"--out-dir={tmp_path / 'out'}"])
    with pytest.raises(SystemExit) as exc_info:
        stage.main()
    assert exc_info.value.code == 3


def test_stage_vocab_mismatch_with_override(tmp_path, monkeypatch):
    """Test vocabulary check with --words override."""
    run_dir, golden, ctx = export_run_dir(tmp_path)

    # Create a words file that would pass (48 words for V=50)
    words_file = tmp_path / "words.txt"
    words_file.write_text("\n".join(f"word{i}" for i in range(48)))

    # Create a words file with different count
    override_words_file = tmp_path / "override_words.txt"
    override_words_file.write_text("\n".join(f"word{i}" for i in range(10)))

    # Run stage.py with --words pointing to the mismatched file
    from tools.nn import stage

    monkeypatch.setattr("sys.argv", [
        "stage.py", str(run_dir),
        f"--words={override_words_file}"
    ])
    with pytest.raises(SystemExit) as exc_info:
        stage.main()
    assert exc_info.value.code == 3
