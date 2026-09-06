#!/usr/bin/env python3
"""Promote a trained run into the app.

Usage: stage.py RUN_DIR [--dry-run] [--words PATH]

1. Load RUN_DIR/en_nextword.bin and golden.json; verify golden logits match.
2. Print a one-line summary of the model.
3. Copy assets to app/src/main/assets/ and tools/nn/out/ (unless --dry-run).
4. Verify V matches the dictionary line count (V = n_words + 2).
"""

import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Promote a trained run into the app"
    )
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy files")
    parser.add_argument("--words", help="Path to en_words.txt (default: app/src/main/assets/en_words.txt)")
    parser.add_argument("--app-dir", help="asset directory to stage into (default: app/src/main/assets)")
    parser.add_argument("--out-dir", help="copy directory (default: tools/nn/out)")
    args = parser.parse_args()

    run_dir = args.run_dir

    # Import tnw after arg parsing so errors show usage
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tnw

    # Paths
    bin_path = os.path.join(run_dir, "en_nextword.bin")
    golden_path = os.path.join(run_dir, "golden.json")

    # Load model and golden
    net = tnw.read_tnw(bin_path)
    with open(golden_path, "r") as f:
        golden = json.load(f)

    # Verify golden
    context = golden["context"]
    top_ids = golden["top_ids"]
    top_logits = golden["top_logits"]

    # Recompute logits
    computed_logits = tnw.predict_logits(net, context)

    # Check argsort matches
    computed_top = computed_logits.argsort()[::-1][: len(top_ids)].tolist()
    if computed_top != top_ids:
        print(f"golden mismatch: top_ids differ", file=sys.stderr)
        print(f"  expected: {top_ids}", file=sys.stderr)
        print(f"  got:      {computed_top}", file=sys.stderr)
        sys.exit(2)

    # Check logit values within 1e-3
    for i, (got, expected) in enumerate(zip(computed_logits[top_ids], top_logits)):
        if abs(got - expected) > 1e-3:
            print(f"golden mismatch: logit[{i}] for id {top_ids[i]}", file=sys.stderr)
            print(f"  expected: {expected}", file=sys.stderr)
            print(f"  got:      {got}", file=sys.stderr)
            sys.exit(2)

    print("golden ok")

    # Print summary
    V = net["V"]
    K = net["K"]
    E = net["E"]
    L = net["L"]
    file_size = os.path.getsize(bin_path) / 1e6
    layer_shapes = ", ".join(f"{w.shape[0]}x{w.shape[1]}" for w, _ in net["layers"])

    print(f"{net['magic'].decode()} V={V} K={K} E={E} L={L} layers=[{layer_shapes}] {file_size:.2f}MB")

    if args.dry_run:
        print("[dry-run] no files copied")
        return

    # Verify V matches dictionary
    if args.words:
        words_path = args.words
    else:
        words_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "app/src/main/assets/en_words.txt"
        )
    if os.path.exists(words_path):
        with open(words_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        expected_v = line_count + 2  # BOS + UNK
        if V != expected_v:
            print(f"V mismatch: asset has {V}, dictionary has {line_count} lines (expected V={expected_v})", file=sys.stderr)
            sys.exit(3)

    # Copy files
    # Destination paths
    app_assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "app/src/main/assets"
    )
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    if args.app_dir:
        app_assets_dir = args.app_dir
    if args.out_dir:
        out_dir = args.out_dir

    os.makedirs(app_assets_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Copy to app assets
    app_bin_path = os.path.join(app_assets_dir, "en_nextword.bin")
    shutil.copy2(bin_path, app_bin_path)
    print(f"copied {bin_path} -> {app_bin_path}")

    # Copy to tools/nn/out
    out_bin_path = os.path.join(out_dir, "en_nextword.bin")
    shutil.copy2(bin_path, out_bin_path)
    print(f"copied {bin_path} -> {out_bin_path}")

    # Copy golden.json to tools/nn/out
    out_golden_path = os.path.join(out_dir, "golden.json")
    shutil.copy2(golden_path, out_golden_path)
    print(f"copied {golden_path} -> {out_golden_path}")


if __name__ == "__main__":
    main()
