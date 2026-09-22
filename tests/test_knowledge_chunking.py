import pytest

from core.knowledge.chunking import chunk_text


def test_chunk_text_creates_overlapping_chunks():
    text = "a" * 2500

    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    assert len(chunks) == 4
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2, 3]

    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == 1000

    assert chunks[1]["start_char"] == 800
    assert chunks[1]["end_char"] == 1800


def test_chunk_text_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Snowball", chunk_size=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("Snowball", overlap=-1)


def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Snowball", chunk_size=100, overlap=100)


def test_chunk_text_preserves_short_document():
    chunks = chunk_text("Snowball remembers Kraken.")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Snowball remembers Kraken."
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len("Snowball remembers Kraken.")