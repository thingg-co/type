"""Tests for sweep.py parsing and planning functions."""
import os
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep


def test_parse_train_log():
    """Test parse_train_log on a sample containing metrics and exit code."""
    text = """step 1 loss 4.234 (0s)
step 1000 loss 2.876 (12s)
val ppl 272.5  top1 0.189  top3 0.297
golden: [1234, 5678, 9012]
TRAIN_EXIT 0
"""
    result = sweep.parse_train_log(text)
    assert result["val_ppl"] == 272.5
    assert result["top1"] == 0.189
    assert result["top3"] == 0.297
    assert result["exit_code"] == 0


def test_parse_train_log_missing_exit():
    """Test parse_train_log when TRAIN_EXIT line is missing."""
    text = """step 1 loss 4.234 (0s)
val ppl 272.5  top1 0.189  top3 0.297
"""
    result = sweep.parse_train_log(text)
    assert result["val_ppl"] == 272.5
    assert result["top1"] == 0.189
    assert result["top3"] == 0.297
    assert "exit_code" not in result


def test_parse_eval_log():
    """Test parse_eval_log on the two eval output lines."""
    text = """next-word (50000 cases):  bigram top1 0.321 top3 0.456   network top1 0.512 top3 0.678
2-letter completion (12345 cases):  bigram top1 0.432   network top1 0.567
EVAL_EXIT 0
"""
    result = sweep.parse_eval_log(text)
    assert result["app_top1"] == 0.512
    assert result["app_top3"] == 0.678
    assert result["completion"] == 0.567
    assert result["exit_code"] == 0


def test_parse_eval_log_missing_exit():
    """Test parse_eval_log when EVAL_EXIT line is missing."""
    text = """next-word (50000 cases):  bigram top1 0.321 top3 0.456   network top1 0.512 top3 0.678
2-letter completion (12345 cases):  bigram top1 0.432   network top1 0.567
"""
    result = sweep.parse_eval_log(text)
    assert result["app_top1"] == 0.512
    assert result["app_top3"] == 0.678
    assert result["completion"] == 0.567
    assert "exit_code" not in result


def test_render_table():
    """Test render_table produces header and rows."""
    rows = [
        {"name": "run1", "args": "--k 8 --layers 2", "val_ppl": 272.5, "top1": 0.189, "top3": 0.297,
         "app_top1": 0.512, "app_top3": 0.678, "completion": 0.567, "mb": 45.3, "minutes": 12.5},
        {"name": "run2", "args": "--k 10 --layers 3", "val_ppl": None, "top1": None, "top3": None,
         "app_top1": 0.499, "app_top3": 0.655, "completion": 0.543, "mb": 67.8, "minutes": 18.2},
    ]
    result = sweep.render_table(rows)

    # Check header
    assert "| name | args | val ppl | top1 | top3 | app top1 | app top3 | completion | MB | minutes |" in result

    # Check separator
    assert "|---" in result

    # Check run1 row
    assert "| run1 | --k 8 --layers 2 | 272.5 | 0.189 | 0.297 | 0.512 | 0.678 | 0.567 | 45.3 | 12.5 |" in result

    # Check run2 row with "-" for missing val_ppl
    assert "| run2 | --k 10 --layers 3 | - | - | - | 0.499 | 0.655 | 0.543 | 67.8 | 18.2 |" in result


def test_plan_skips_completed(tmp_path):
    """Test that plan() skips runs with EVAL_EXIT 0 in eval.log."""
    out_dir = str(tmp_dir := tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    # Create config with two runs
    config = {
        "data": "~/data",
        "val": "~/val.bin",
        "out": out_dir,
        "python": "~/python",
        "runs": [
            {"name": "completed_run", "args": ["--k", "8"]},
            {"name": "pending_run", "args": ["--k", "10"]},
        ],
    }

    # Create run directories
    completed_dir = os.path.join(out_dir, "completed_run")
    pending_dir = os.path.join(out_dir, "pending_run")
    os.makedirs(completed_dir, exist_ok=True)
    os.makedirs(pending_dir, exist_ok=True)

    # Create eval.log with EVAL_EXIT 0 for completed run
    with open(os.path.join(completed_dir, "eval.log"), "w") as f:
        f.write("next-word (100 cases): network top1 0.5 top3 0.7\n")
        f.write("EVAL_EXIT 0\n")

    # No eval.log for pending run

    # Plan should only return pending_run
    result = sweep.plan(config)
    assert result == ["pending_run"]


def test_plan_completes_all(tmp_path):
    """Test that plan() returns empty list when all runs are completed."""
    out_dir = str(tmp_dir := tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "data": "~/data",
        "val": "~/val.bin",
        "out": out_dir,
        "python": "~/python",
        "runs": [
            {"name": "run1", "args": ["--k", "8"]},
            {"name": "run2", "args": ["--k", "10"]},
        ],
    }

    # Create eval.log with EVAL_EXIT 0 for both runs
    for name in ["run1", "run2"]:
        run_dir = os.path.join(out_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "eval.log"), "w") as f:
            f.write("EVAL_EXIT 0\n")

    # Plan should return empty list
    result = sweep.plan(config)
    assert result == []


def test_plan_no_completion_file(tmp_path):
    """Test that plan() includes runs without eval.log."""
    out_dir = str(tmp_dir := tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "data": "~/data",
        "val": "~/val.bin",
        "out": out_dir,
        "python": "~/python",
        "runs": [
            {"name": "no_file", "args": ["--k", "8"]},
        ],
    }

    # No eval.log file

    result = sweep.plan(config)
    assert result == ["no_file"]


def test_plan_respects_nonzero_exit(tmp_path):
    """Test that plan() includes runs where eval.log has non-zero exit."""
    out_dir = str(tmp_dir := tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "data": "~/data",
        "val": "~/val.bin",
        "out": out_dir,
        "python": "~/python",
        "runs": [
            {"name": "failed_run", "args": ["--k", "8"]},
        ],
    }

    # Create eval.log with EVAL_EXIT 1 (not 0)
    run_dir = os.path.join(out_dir, "failed_run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "eval.log"), "w") as f:
        f.write("EVAL_EXIT 1\n")

    # Plan should include the run since EVAL_EXIT 0 is not present
    result = sweep.plan(config)
    assert result == ["failed_run"]


def test_expand_paths():
    """Test that expand_paths expands ~ in all path fields."""
    config = {
        "data": "~/type-data/data_mix",
        "val": "~/type-data/data126k/val.bin",
        "out": "~/type-out",
        "python": "~/lab/.venv/bin/python",
        "runs": [
            {"name": "test", "args": ["--k", "8"]},
        ],
    }

    expanded = sweep.expand_paths(config)

    # Paths should be expanded (not contain ~)
    assert expanded["data"].startswith("/")
    assert expanded["val"].startswith("/")
    assert expanded["out"].startswith("/")
    assert expanded["python"].startswith("/")

    # Run names should be preserved
    assert expanded["runs"][0]["name"] == "test"


def test_plan_with_empty_config():
    """Test that plan() returns empty list for config with no runs."""
    config = {
        "data": "~/data",
        "val": "~/val.bin",
        "out": "/tmp/out",
        "python": "~/python",
        "runs": [],
    }

    result = sweep.plan(config)
    assert result == []
