"""Tests for build_caps.py canonical capitalization builder."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_caps


def test_load_vocab():
    """load_vocab correctly loads vocabulary from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("word1\n")
            f.write("word2\n")
            f.write("word3\n")

        result = build_caps.load_vocab(vocab_path)
        assert result == {"word1", "word2", "word3"}


def test_tokenize():
    """tokenize correctly extracts words from text."""
    text = "Hello world! How are you?"
    result = build_caps.tokenize(text)
    assert result == ["Hello", "world", "How", "are", "you"]

    # Also handles curly apostrophes
    text2 = "It's a test"
    result2 = build_caps.tokenize(text2)
    assert "It's" in result2


def test_count_forms():
    """count_forms correctly counts capitalized forms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")
            f.write("world\n")
            f.write("john\n")

        # Create test input file
        # Sentence-initial tokens are ignored
        # "hello" and "Hello" at start of line 1-3 are ignored
        # "hello" and "Hello" at start of line 4-5 are ignored
        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            f.write("hello world\n")  # "hello" SI, "world" counted
            f.write("Hello world\n")  # "Hello" SI, "world" counted
            f.write("HELLO world\n")  # "HELLO" SI, "world" counted
            f.write("hello John\n")  # "hello" SI, "John" counted (as "john")
            f.write("Hello John\n")  # "Hello" SI, "John" counted (as "john")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # "hello" is never counted - it only appears at sentence-initial positions
        assert "hello" not in forms

        # "world" has 5 occurrences (sentence-initial ones ignored): 3 lowercase, 2 uppercase
        assert "world" in forms
        assert forms["world"]["world"] == 3

        # "john" has 2 occurrences (both uppercase, since "John" follows sentence-initial)
        assert "john" in forms
        assert forms["john"]["John"] == 2


def test_build_entries_basic():
    """build_entries correctly filters by MIN_SEEN and MIN_RATIO."""
    # Create forms dict with test data
    forms = {}

    # Word with enough occurrences but low ratio (not capitalized enough)
    c1 = build_caps.Counter()
    c1["hello"] = 10
    c1["Hello"] = 10
    forms["hello"] = c1

    # Word with enough occurrences and high ratio (mostly capitalized)
    c2 = build_caps.Counter()
    c2["john"] = 5
    c2["John"] = 20
    c2["JOHN"] = 1
    forms["john"] = c2

    # Word with too few occurrences
    c3 = build_caps.Counter()
    c3["test"] = 10
    forms["test"] = c3

    entries = build_caps.build_entries(forms)

    # hello should not be included (only 50% capitalized, need 90%)
    assert ("hello", "Hello") not in entries

    # john: 20/26 = 76.9% < 90%, should not be included
    assert ("john", "John") not in entries

    # test should not be included (only 10 occurrences, need 25)
    assert ("test", "Test") not in entries


def test_build_entries_with_enough_data():
    """build_entries correctly includes words with sufficient data."""
    forms = {}

    # Word with 25+ occurrences and 90%+ capitalized
    c = build_caps.Counter()
    c["john"] = 2
    c["John"] = 23  # 23/25 = 0.92 >= 0.9
    forms["john"] = c

    entries = build_caps.build_entries(forms)

    assert ("john", "John") in entries


def test_build_entries_sorted():
    """build_entries returns results sorted alphabetically."""
    forms = {}
    c1 = build_caps.Counter()
    c1["zebra"] = 2
    c1["Zebra"] = 23
    forms["zebra"] = c1

    c2 = build_caps.Counter()
    c2["apple"] = 2
    c2["Apple"] = 23
    forms["apple"] = c2

    c3 = build_caps.Counter()
    c3["mango"] = 2
    c3["Mango"] = 23
    forms["mango"] = c3

    entries = build_caps.build_entries(forms)

    # Should be sorted: apple, mango, zebra
    assert entries[0] == ("apple", "Apple")
    assert entries[1] == ("mango", "Mango")
    assert entries[2] == ("zebra", "Zebra")


def test_word_not_in_vocab_ignored():
    """Words not in vocab are not counted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")

        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            # "hello" is sentence-initial (ignored), "world" not in vocab
            f.write("hello world\n")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # No words counted (sentence-initial hello ignored, world not in vocab)
        assert len(forms) == 0


def test_sentence_initial_token_ignored():
    """First token of each line (sentence-initial) is ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")
            f.write("world\n")

        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            # "Hello" is sentence-initial, ignored
            # "hello" is not sentence-initial, counted
            f.write("Hello hello\n")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # Only the second hello is counted
        assert forms["hello"]["hello"] == 1
        assert "Hello" not in forms["hello"]


def test_tsv_format_handling():
    """count_forms correctly handles TSV input format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")

        input_path = os.path.join(tmpdir, "input.tsv")
        with open(input_path, "w") as f:
            # Tatoeba format: id\tlang\ttext
            # "Hello" is sentence-initial, ignored
            f.write("1\teng\tHello world\n")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # "hello" is sentence-initial, ignored; "world" not in vocab
        # So actually no words should be counted
        assert len(forms) == 0


def test_non_eng_tsv_format():
    """count_forms falls back to plain text for non-English TSV lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")

        input_path = os.path.join(tmpdir, "input.tsv")
        with open(input_path, "w") as f:
            # Non-English line should use plain text parsing (full line)
            f.write("1\tspa\tHello world\n")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # For non-eng, text = full line. Tokenization gives ['1', 'spa', 'Hello', 'world']
        # First token '1' is sentence-initial, ignored
        # 'spa' is not in vocab
        # 'Hello' (lowercased to 'hello') is in vocab, counted as 'Hello'
        # 'world' is not in vocab
        assert "hello" in forms
        assert forms["hello"]["Hello"] == 1


def test_mixed_case_in_input():
    """count_forms correctly counts different case variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("hello\n")

        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            # All occurrences are sentence-initial, so none are counted
            f.write("Hello\n")
            f.write("hello\n")
            f.write("HELLO\n")

        vocab = build_caps.load_vocab(vocab_path)
        forms = build_caps.count_forms(input_path, vocab)

        # All are sentence-initial, so nothing is counted
        assert len(forms) == 0
