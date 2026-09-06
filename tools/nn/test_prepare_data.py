"""Tests for prepare_data.py tokenization, ID mapping, and output generation."""
import os
import struct
import random
import sys
import tempfile

# Import prepare_data module by path (like test_tnw.py does)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_data


def make_vocab():
    """Create a small test vocabulary."""
    return {"the": 0, "quick": 1, "brown": 2, "fox": 3, "jumps": 4,
            "over": 5, "lazy": 6, "dog": 7, "hello": 8, "world": 9,
            "and": 10, "this": 11, "is": 12, "a": 13, "test": 14}


def test_tokenize_splits_on_punctuation_and_newlines():
    """tokenize splits on . ! ? and newlines, lowercases, keeps contractions."""
    vocab = make_vocab()
    text = "Hello world is great! This is a test. How are you?\nI am fine."
    result = prepare_data.tokenize(text)
    assert len(result) == 4
    assert result[0] == ["hello", "world", "is", "great"]
    assert result[1] == ["this", "is", "a", "test"]
    assert result[2] == ["how", "are", "you"]
    assert result[3] == ["i", "am", "fine"]


def test_tokenize_keeps_apostrophe_contractions():
    """tokenize keeps apostrophe contractions as one token."""
    text = "don't stop believing. can't be serious"
    result = prepare_data.tokenize(text)
    assert result == [["don't", "stop", "believing"], ["can't", "be", "serious"]]


def test_tokenize_converts_curly_apostrophes():
    """tokenize converts curly apostrophes to straight ones."""
    text = "don't stop here. it is fine"
    result = prepare_data.tokenize(text)
    assert result == [["don't", "stop", "here"], ["it", "is", "fine"]]


def test_tokenize_drops_short_runs():
    """tokenize drops runs shorter than 3 words."""
    text = "hi\nhello world there\ntest"
    result = prepare_data.tokenize(text)
    # "hi" (1 word) and "test" (1 word) are dropped
    assert result == [["hello", "world", "there"]]


def test_tokenize_drops_long_runs():
    """tokenize drops runs longer than 40 words."""
    # Create a line with 41 words
    long_text = " ".join(["word"] * 41)
    result = prepare_data.tokenize(long_text)
    assert result == []
    # 40 words should be kept
    forty_text = " ".join(["word"] * 40)
    result = prepare_data.tokenize(forty_text)
    assert len(result) == 1
    assert len(result[0]) == 40


def test_to_ids_maps_unknown_words_to_unk():
    """to_ids maps unknown words to UNK."""
    vocab = make_vocab()
    UNK = 999
    words = ["the", "unknown", "fox", "also_unknown"]
    ids = prepare_data.to_ids(words, vocab, UNK)
    assert ids == [0, UNK, 3, UNK]


def test_keep_sentence_rejects_high_unk_rate():
    """keep_sentence rejects sentences with unk rate above 0.2."""
    ids = [0, 1, 2, 3, 999]  # 1/5 = 0.2 exactly
    assert prepare_data.keep_sentence(ids, 999, max_unk_rate=0.2) is True

    ids = [0, 1, 2, 999, 999]  # 2/5 = 0.4 > 0.2
    assert prepare_data.keep_sentence(ids, 999, max_unk_rate=0.2) is False

    ids = [0, 1, 2, 3, 4]  # 0/5 = 0.0 <= 0.2
    assert prepare_data.keep_sentence(ids, 999, max_unk_rate=0.2) is True


def test_keep_sentence_accepts_exactly_0_2():
    """keep_sentence accepts exactly 0.2 unk rate."""
    # 1 UNK out of 5 = 0.2
    ids = [0, 1, 2, 3, 999]
    assert prepare_data.keep_sentence(ids, 999, max_unk_rate=0.2) is True


def test_sniff_line_tatoeba_eng():
    """sniff_line takes column 3 of an 'eng' TSV line."""
    line = "1\teng\thello world"
    assert prepare_data.sniff_line(line) == "hello world"


def test_sniff_line_tatoeba_non_eng():
    """sniff_line returns whole line for non-eng TSV."""
    line = "1\tfra\tbonjour le monde"
    assert prepare_data.sniff_line(line) == "1\tfra\tbonjour le monde"


def test_sniff_line_plain_line():
    """sniff_line returns whole line for plain text."""
    line = "hello world"
    assert prepare_data.sniff_line(line) == "hello world"


def test_sniff_line_not_three_columns():
    """sniff_line returns whole line if not exactly 3 columns."""
    line = "1\thello\tworld\textra"
    assert prepare_data.sniff_line(line) == "1\thello\tworld\textra"


def test_writer_packs_sentences_correctly():
    """Writer with seeded RNG packs sentences correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = random.Random(42)
        writer = prepare_data.Writer(tmpdir, rng)

        # Write a few sentences
        writer.emit([0, 1, 2])
        writer.emit([3, 4])
        writer.emit([5, 6, 7, 8])

        writer.close()

        # Read back and verify
        with open(f"{tmpdir}/train.bin", "rb") as f:
            train_data = f.read()

        # Check the structure: each sentence is len+1 uint32s followed by SEP
        # First sentence: 3 ids + 1 SEP = 4 * 4 = 16 bytes
        # Second sentence: 2 ids + 1 SEP = 3 * 4 = 12 bytes
        # Third sentence: 4 ids + 1 SEP = 5 * 4 = 20 bytes
        assert len(train_data) == 16 + 12 + 20

        # Verify first sentence
        val = struct.unpack("<4I", train_data[0:16])
        assert val == (0, 1, 2, 0xFFFFFFFF)

        # Verify second sentence
        val = struct.unpack("<3I", train_data[16:28])
        assert val == (3, 4, 0xFFFFFFFF)

        # Verify third sentence
        val = struct.unpack("<5I", train_data[28:48])
        assert val == (5, 6, 7, 8, 0xFFFFFFFF)


def test_writer_routes_to_val_based_on_rng():
    """Writer routes ~1% of sentences to val with seeded RNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a fixed seed for deterministic behavior
        rng = random.Random(7)
        writer = prepare_data.Writer(tmpdir, rng)

        # Write 2000 sentences to get enough samples
        for _ in range(2000):
            writer.emit([0, 1, 2])

        writer.close()

        # Count files written (train and val should both exist)
        import os
        assert os.path.exists(f"{tmpdir}/train.bin")
        assert os.path.exists(f"{tmpdir}/val.bin")

        # With 1% rate over 2000 sentences, expect ~20 in val
        # With seed 7, verify the count is in a reasonable range
        with open(f"{tmpdir}/val.bin", "rb") as f:
            val_data = f.read()

        # Each sentence is 4 bytes (3 ids + 1 sep)
        val_count = len(val_data) // 16  # 4 uint32s per sentence
        assert 5 <= val_count <= 40


def test_main_produces_same_bytes_as_hand_call():
    """main() on a 2-sentence file produces same bytes as calling pieces by hand."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple input file
        input_path = os.path.join(tmpdir, "input.txt")
        with open(input_path, "w") as f:
            f.write("hello world\n")
            f.write("this is a test\n")

        # Run main
        prepare_data.main([tmpdir, input_path])

        # Now do it by hand
        vocab = prepare_data.load_vocab("app/src/main/assets/en_words.txt")
        V = len(vocab)
        UNK = V + 1

        with tempfile.TemporaryDirectory() as tmpdir2:
            rng = random.Random(7)
            writer = prepare_data.Writer(tmpdir2, rng)

            # Same input, processed manually
            with open(input_path, "r") as f:
                for line in f:
                    text = prepare_data.sniff_line(line)
                    for words in prepare_data.tokenize(text):
                        ids = prepare_data.to_ids(words, vocab, UNK)
                        if prepare_data.keep_sentence(ids, UNK):
                            writer.emit(ids)

            writer.close()

            # Compare train.bin
            with open(f"{tmpdir}/train.bin", "rb") as f1, open(f"{tmpdir2}/train.bin", "rb") as f2:
                assert f1.read() == f2.read()

            # Compare val.bin
            with open(f"{tmpdir}/val.bin", "rb") as f1, open(f"{tmpdir2}/val.bin", "rb") as f2:
                assert f1.read() == f2.read()


def test_tokenize_curly_apostrophe_handling():
    """Verify curly apostrophe conversion happens before regex matching."""
    text = "don't stop here. it is fine"
    result = prepare_data.tokenize(text)
    # All should be detected as valid contractions
    assert len(result) == 2
    assert len(result[0]) == 3  # don't, stop, here
    assert len(result[1]) == 3  # it, is, fine
