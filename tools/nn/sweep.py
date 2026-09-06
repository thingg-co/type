#!/usr/bin/env python3
"""Sweep runner for next-word training experiments.

Usage: sweep.py CONFIG.json    # run all runs in the config
       sweep.py --table CONFIG.json  # only rebuild the markdown table
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def expand_paths(config):
    """Expand ~ in all path fields of a config dict."""
    config = dict(config)
    config["data"] = os.path.expanduser(config["data"])
    config["val"] = os.path.expanduser(config["val"])
    config["out"] = os.path.expanduser(config["out"])
    config["python"] = os.path.expanduser(config["python"])
    for run in config["runs"]:
        run["name"] = run["name"]
        run["args"] = list(run["args"])
    return config


def plan(config):
    """Return list of run names that should be executed (skipping already completed)."""
    out_dir = config["out"]
    completed = set()
    for run in config["runs"]:
        name = run["name"]
        eval_log = os.path.join(out_dir, name, "eval.log")
        if os.path.exists(eval_log):
            with open(eval_log, "r") as f:
                content = f.read()
                if "EVAL_EXIT 0" in content:
                    completed.add(name)
    return [run["name"] for run in config["runs"] if run["name"] not in completed]


def parse_train_log(text):
    """Parse train.py output to extract metrics.

    Returns dict with keys: val_ppl, top1, top3, exit_code (or None if not found).
    """
    result = {}
    for line in text.split("\n"):
        # Look for line like: "val ppl 272.5  top1 0.189  top3 0.297"
        if "val ppl" in line:
            # Extract val_ppl
            parts = line.split()
            # Find "val" and "ppl" positions
            try:
                ppl_idx = parts.index("ppl")
                result["val_ppl"] = float(parts[ppl_idx + 1])
            except (ValueError, IndexError):
                pass
            # Find top1
            try:
                top1_idx = parts.index("top1")
                result["top1"] = float(parts[top1_idx + 1])
            except (ValueError, IndexError):
                pass
            # Find top3
            try:
                top3_idx = parts.index("top3")
                result["top3"] = float(parts[top3_idx + 1])
            except (ValueError, IndexError):
                pass
        # Look for TRAIN_EXIT line
        if line.startswith("TRAIN_EXIT "):
            result["exit_code"] = int(line.split()[-1])
    return result


def parse_eval_log(text):
    """Parse eval_nextword.py output to extract metrics.

    Returns dict with keys: app_top1, app_top3, completion (or None if not found).
    """
    result = {}
    for line in text.split("\n"):
        # Next-word line: "next-word (N cases): ... network top1 A top3 B"
        if line.startswith("next-word ("):
            # Extract network top1 and top3 (skip bigram values)
            parts = line.split()
            # Find "network" and then look for top1/top3 after it
            try:
                network_idx = parts.index("network")
                top1_idx = parts.index("top1", network_idx)
                result["app_top1"] = float(parts[top1_idx + 1])
                top3_idx = parts.index("top3", network_idx)
                result["app_top3"] = float(parts[top3_idx + 1])
            except (ValueError, IndexError):
                pass
        # 2-letter completion line: "2-letter completion (M cases): ... network top1 C"
        elif line.startswith("2-letter completion ("):
            # Extract network top1 for completion (skip bigram)
            parts = line.split()
            try:
                network_idx = parts.index("network")
                top1_idx = parts.index("top1", network_idx)
                result["completion"] = float(parts[top1_idx + 1])
            except (ValueError, IndexError):
                pass
        # Look for EVAL_EXIT line
        if line.startswith("EVAL_EXIT "):
            result["exit_code"] = int(line.split()[-1])
    return result


def render_table(rows):
    """Render table rows as markdown string.

    rows: list of dicts with keys matching the headers (with underscores).
    """
    # Map from display headers to dict keys (with underscores)
    header_key_map = [
        ("name", "name"),
        ("args", "args"),
        ("val ppl", "val_ppl"),
        ("top1", "top1"),
        ("top3", "top3"),
        ("app top1", "app_top1"),
        ("app top3", "app_top3"),
        ("completion", "completion"),
        ("MB", "mb"),
        ("minutes", "minutes"),
    ]
    lines = []

    # Header row
    headers = [h for h, _ in header_key_map]
    lines.append("| " + " | ".join(headers) + " |")

    # Separator row
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")

    # Data rows
    for row in rows:
        fmt_row = []
        for header, key in header_key_map:
            val = row.get(key)
            if val is None or val == "":
                val = "-"
            else:
                val = str(val)
            fmt_row.append(val)
        lines.append("| " + " | ".join(fmt_row) + " |")

    return "\n".join(lines)


def run_train(config, run, out_dir, log_dir, python, repo_root):
    """Run train.py for a single run, return (train_log, eval_log, elapsed_minutes, bin_size)."""
    name = run["name"]
    args = run["args"]

    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    train_log_path = os.path.join(log_dir, "train.log")
    eval_log_path = os.path.join(log_dir, "eval.log")

    # Build command for train.py
    train_cmd = [python, "tools/nn/train.py", config["data"], run_dir] + args

    # Run train.py from repo root
    t0 = time.time()
    result = subprocess.run(
        train_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - t0

    # Write train log
    with open(train_log_path, "w") as f:
        f.write(result.stdout + result.stderr)
        f.write(f"\nTRAIN_EXIT {result.returncode}\n")

    if result.returncode != 0:
        return None, None, elapsed, None

    # Run eval_nextword.py
    bin_path = os.path.join(run_dir, "en_nextword.bin")
    eval_cmd = [python, "tools/nn/eval_nextword.py", config["val"], "--bin", bin_path]

    result2 = subprocess.run(
        eval_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True
    )

    # Write eval log
    with open(eval_log_path, "w") as f:
        f.write(result2.stdout + result2.stderr)
        f.write(f"\nEVAL_EXIT {result2.returncode}\n")

    # Get file size
    bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else None

    # Read logs for parsing
    with open(train_log_path, "r") as f:
        train_text = f.read()
    with open(eval_log_path, "r") as f:
        eval_text = f.read()

    return train_text, eval_text, elapsed, bin_size


def main():
    if len(sys.argv) < 2:
        print("Usage: sweep.py CONFIG.json [--table]", file=sys.stderr)
        sys.exit(1)

    # Determine if we're just rebuilding the table
    if sys.argv[1] == "--table":
        # Second argument should be the config file
        config_path = sys.argv[2]
        config_name = Path(config_path).stem
    else:
        config_path = sys.argv[1]
        config_name = Path(config_path).stem

    with open(config_path, "r") as f:
        config = json.load(f)

    config = expand_paths(config)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if sys.argv[1] == "--table":
        # Only rebuild table from existing logs
        rows = []
        for run in config["runs"]:
            name = run["name"]
            run_dir = os.path.join(config["out"], name)
            train_log_path = os.path.join(run_dir, "train.log")

            # Parse existing logs if they exist
            train_text = None
            eval_text = None
            train_metrics = {}
            eval_metrics = {}
            bin_size = None
            elapsed = None

            if os.path.exists(train_log_path):
                with open(train_log_path, "r") as f:
                    train_text = f.read()
                train_metrics = parse_train_log(train_text)

            eval_log_path = os.path.join(run_dir, "eval.log")
            if os.path.exists(eval_log_path):
                with open(eval_log_path, "r") as f:
                    eval_text = f.read()
                eval_metrics = parse_eval_log(eval_text)

            # Get bin size
            bin_path = os.path.join(run_dir, "en_nextword.bin")
            if os.path.exists(bin_path):
                bin_size = os.path.getsize(bin_path)

            # Get elapsed time from train log (parse TRAIN_EXIT and estimate)
            # We'll use "-" for elapsed if not available
            elapsed = None

            # Build row
            row = {
                "name": name,
                "args": " ".join(run["args"]),
                "val_ppl": train_metrics.get("val_ppl"),
                "top1": train_metrics.get("top1"),
                "top3": train_metrics.get("top3"),
                "app_top1": eval_metrics.get("app_top1"),
                "app_top3": eval_metrics.get("app_top3"),
                "completion": eval_metrics.get("completion"),
                "mb": round(bin_size / 1e6, 1) if bin_size else None,
                "minutes": elapsed,
            }
            rows.append(row)

        table = render_table(rows)
        print(table)
        sys.exit(0)

    # Normal run: execute all runs
    out_dir = config["out"]
    python = config["python"]

    # Determine which runs to execute
    runs_to_run = plan(config)

    print(f"Runs to execute: {runs_to_run}")
    print(f"Total: {len(runs_to_run)} runs")

    all_rows = []
    completed_names = set()

    for run in config["runs"]:
        name = run["name"]
        args = run["args"]

        # Check if we should skip this run
        if name not in runs_to_run:
            print(f"Skipping {name} (already completed)")
            # Still read existing logs for the table
            run_dir = os.path.join(out_dir, name)
            train_metrics = {}
            eval_metrics = {}
            bin_size = None
            elapsed = None

            train_log_path = os.path.join(run_dir, "train.log")
            eval_log_path = os.path.join(run_dir, "eval.log")

            if os.path.exists(train_log_path):
                with open(train_log_path, "r") as f:
                    train_text = f.read()
                train_metrics = parse_train_log(train_text)

            if os.path.exists(eval_log_path):
                with open(eval_log_path, "r") as f:
                    eval_text = f.read()
                eval_metrics = parse_eval_log(eval_text)

            bin_path = os.path.join(run_dir, "en_nextword.bin")
            if os.path.exists(bin_path):
                bin_size = os.path.getsize(bin_path)

            row = {
                "name": name,
                "args": " ".join(args),
                "val_ppl": train_metrics.get("val_ppl"),
                "top1": train_metrics.get("top1"),
                "top3": train_metrics.get("top3"),
                "app_top1": eval_metrics.get("app_top1"),
                "app_top3": eval_metrics.get("app_top3"),
                "completion": eval_metrics.get("completion"),
                "mb": round(bin_size / 1e6, 1) if bin_size else None,
                "minutes": elapsed,
            }
            all_rows.append(row)
            continue

        print(f"\nRunning {name}...")
        print(f"  args: {' '.join(args)}")

        train_text, eval_text, elapsed, bin_size = run_train(
            config, run, out_dir, out_dir, python, repo_root
        )

        if train_text is None:
            print(f"  FAILED train")
            row = {
                "name": name,
                "args": " ".join(args),
                "val_ppl": None,
                "top1": None,
                "top3": None,
                "app_top1": None,
                "app_top3": None,
                "completion": None,
                "mb": None,
                "minutes": None,
            }
            all_rows.append(row)
            # Rebuild table after failure
            table = render_table(all_rows)
            sweep_md_path = os.path.join(out_dir, "sweep_" + config_name + ".md")
            with open(sweep_md_path, "w") as f:
                f.write("# Sweep Results\n\n")
                f.write(table)
                f.write("\n")
            print(f"\nTable written to {sweep_md_path}")
            continue

        train_metrics = parse_train_log(train_text)
        eval_metrics = parse_eval_log(eval_text)

        row = {
            "name": name,
            "args": " ".join(args),
            "val_ppl": train_metrics.get("val_ppl"),
            "top1": train_metrics.get("top1"),
            "top3": train_metrics.get("top3"),
            "app_top1": eval_metrics.get("app_top1"),
            "app_top3": eval_metrics.get("app_top3"),
            "completion": eval_metrics.get("completion"),
            "mb": round(bin_size / 1e6, 1) if bin_size else None,
            "minutes": round(elapsed / 60, 1) if elapsed else None,
        }
        all_rows.append(row)
        completed_names.add(name)

        # Rebuild table after each run
        table = render_table(all_rows)
        sweep_md_path = os.path.join(out_dir, "sweep_" + config_name + ".md")
        with open(sweep_md_path, "w") as f:
            f.write("# Sweep Results\n\n")
            f.write(table)
            f.write("\n")
        print(f"\nTable written to {sweep_md_path}")

    print("\n" + "=" * 60)
    print("Sweep complete!")
    print(table)


if __name__ == "__main__":
    main()
