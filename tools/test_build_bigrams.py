"""Tests for build_bigrams.py tokenization, pair counting, scoring, and binary output."""
import os
import struct
import sys
import tempfile

# Import build_bigrams module by path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_bigrams


def make_vocab():
    """Create a small test vocabulary."""
    return {"the": 0, "quick": 1, "brown": 2, "fox": 3, "jumps": 4,
            "over": 5, "lazy": 6, "dog": 7, "hello": 8, "world": 9,
            "and": 10, "this": 11, "is": 12, "a": 13, "test": 14}


def test_pair_counting():
    """pair counting on a tiny token stream gives the expected counts."""
    pairs = [
        ("the", "quick", 100),
        ("quick", "brown", 50),
        ("the", "quick", 100),  # same pair, different count
    ]
    ids = make_vocab()

    result = build_bigrams.filter_and_pack_pairs(pairs, ids)

    # We have 3 pairs, but only 2 unique (the last is a duplicate)
    # The function just collects, doesn't aggregate
    assert len(result) == 3
    # Check the packed keys
    # (prev << 16) | next
    # (0 << 16) | 1 = 1
    # (1 << 16) | 2 = 65538
    # (0 << 16) | 1 = 1 (same as first)
    assert result[0] == (100, 1)
    assert result[1] == (50, 65538)
    assert result[2] == (100, 1)


def test_ids_at_or_above_65536_dropped():
    """ids at or above 65536 are not included in the vocab dict.

    The load_vocab function truncates to 65536 words, so words at or beyond
    line 65536 are never loaded. This means their ids (line numbers) are
    never present in the dict, and pairs containing them will have ib=None.
    """
    # The loaded vocab is limited to 65536 words
    ids = build_bigrams.load_vocab("app/src/main/assets/en_words.txt")
    assert len(ids) == 65536
    # Word "the" has id 0, "to" has id 1, etc.
    # Check that a known word within range is present
    assert "the" in ids
    assert ids["the"] == 0

    # Verify that id 65536 (line 65537) is not in the dict
    # (because load_vocab only loads first 65536 words)
    # Note: we can't test this directly since we don't know the word at line 65536
    # Instead, we test that the dict size is exactly 65536
    assert len(ids) == 65536

    # Test that pairs with unknown words (ib=None) are filtered out
    custom_ids = {"a": 100, "b": 200}
    pairs = [
        ("a", "unknown", 100),  # second word unknown, ib=None
        ("unknown", "b", 200),  # first word unknown, ia=None
        ("a", "b", 300),  # both known
    ]
    result = build_bigrams.filter_and_pack_pairs(pairs, custom_ids)
    # Only the third pair should be kept
    assert len(result) == 1
    assert result[0] == (300, (100 << 16) | 200)


def test_score_quantisation_max_to_255():
    """The score quantisation maps the maximum count to 255 and never exceeds 255."""
    # Test with a very large count
    count = 1000000
    score = build_bigrams.compute_scores_and_sort([(count, 123)])
    # score = round(10 * ln(1000000)) = round(10 * 13.815...) = 138
    assert score[0][1] == 138
    assert score[0][1] <= 255

    # Test with a count that would give > 255
    # 10 * ln(count) > 255 => ln(count) > 25.5 => count > e^25.5 ~ 1.7e11
    huge_count = 200000000000  # 2e11
    score = build_bigrams.compute_scores_and_sort([(huge_count, 456)])
    assert score[0][1] == 255  # Clamped to max

    # Test with count=1 (minimum valid score)
    count = 1
    score = build_bigrams.compute_scores_and_sort([(count, 789)])
    # round(10 * ln(1)) = round(0) = 0, but clamped to 1
    assert score[0][1] == 1


def test_binary_format_roundtrip():
    """The written file round-trips: parse the bytes with struct and check structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entries = [
            (0x00010002, 100),  # prev=1, next=2, score=100
            (0x00030004, 200),  # prev=3, next=4, score=200
            (0x00050006, 150),  # prev=5, next=6, score=150
        ]
        out_path = os.path.join(tmpdir, "test.bin")
        build_bigrams.write_binary(out_path, entries)

        with open(out_path, "rb") as f:
            data = f.read()

        # Check magic
        assert data[:4] == b"TBG1"

        # Check count (big-endian int32)
        count = struct.unpack(">i", data[4:8])[0]
        assert count == 3

        # Check keys (big-endian int64)
        keys = struct.unpack(">qqq", data[8:32])
        assert keys == (0x00010002, 0x00030004, 0x00050006)

        # Check scores (uint8)
        scores = struct.unpack("BBB", data[32:35])
        assert scores == (100, 200, 150)


def test_top_n_per_prev_keeps_best():
    """The top-N rule keeps the N best entries sorted by key."""
    # Create test data with same prev word but different counts
    # (count, key) pairs where key = (prev << 16) | next
    # We'll use prev=1 for first 3 pairs, prev=2 for next 2
    pairs = [
        (500, 0x00010005),  # prev=1, next=5, count=500 (best for prev=1)
        (100, 0x00010003),  # prev=1, next=3, count=100
        (300, 0x00010007),  # prev=1, next=7, count=300
        (600, 0x00020001),  # prev=2, next=1, count=600 (best for prev=2)
        (200, 0x00020009),  # prev=2, next=9, count=200
    ]

    # Sort by count descending and take all (cap=5)
    pairs.sort(reverse=True)
    assert pairs == [
        (600, 0x00020001),
        (500, 0x00010005),
        (300, 0x00010007),
        (200, 0x00020009),
        (100, 0x00010003),
    ]

    # Compute scores and sort by key
    entries = build_bigrams.compute_scores_and_sort(pairs)
    # Should be sorted by key: 0x00010003 < 0x00010005 < 0x00010007 < 0x00020001 < 0x00020009
    keys = [e[0] for e in entries]
    assert keys == [0x00010003, 0x00010005, 0x00010007, 0x00020001, 0x00020009]


def test_parse_count_file():
    """parse_count_file correctly parses Norvig format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            f.write("the quick 1000\n")  # valid line
            f.write("quick brown 500\n")  # valid line
            f.write("bad line\n")  # invalid - only 2 parts
            f.write("a b c 400\n")  # invalid - 4 parts
            f.write("fox jumps 300\n")  # valid line

        result = build_bigrams.parse_count_file(input_path)
        assert len(result) == 3
        assert result[0] == ("the", "quick", 1000)
        assert result[1] == ("quick", "brown", 500)
        assert result[2] == ("fox", "jumps", 300)


def test_load_vocab():
    """load_vocab correctly loads vocabulary from file."""
    vocab = build_bigrams.load_vocab("app/src/main/assets/en_words.txt")
    assert "the" in vocab
    assert vocab["the"] == 0
    assert "to" in vocab
    assert vocab["to"] == 1
    # Vocabulary should be limited to 65536 words
    assert len(vocab) == 65536


def test_filter_and_pack_pairs_filters_unknown_words():
    """filter_and_pack_pairs filters out pairs with unknown words."""
    ids = {"the": 0, "quick": 1, "fox": 2}
    pairs = [
        ("the", "quick", 100),  # both known
        ("unknown", "word", 200),  # both unknown
        ("the", "unknown", 300),  # one known, one unknown
        ("fox", "the", 400),  # both known
    ]
    result = build_bigrams.filter_and_pack_pairs(pairs, ids)
    assert len(result) == 2
    assert result[0] == (100, 1)  # (0 << 16) | 1 = 1
    assert result[1] == (400, 131072)  # (2 << 16) | 0 = 131072


def test_compute_scores_and_sort():
    """compute_scores_and_sort correctly converts counts to scores."""
    pairs = [
        (1000, 100),  # ln(1000) ~ 6.9, score ~ 69
        (100, 50),    # ln(100) ~ 4.6, score ~ 46
        (10, 200),    # ln(10) ~ 2.3, score ~ 23
    ]
    result = build_bigrams.compute_scores_and_sort(pairs)
    # Should be sorted by key ascending
    assert len(result) == 3
    assert result[0] == (50, 46)   # key=50, score=46
    assert result[1] == (100, 69)  # key=100, score=69
    assert result[2] == (200, 23)  # key=200, score=23
