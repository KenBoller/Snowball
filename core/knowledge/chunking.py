from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split text into overlapping chunks with source offsets.

    This is Snowball's baseline character-based chunker. More specialized
    chunkers can later handle conversations, structured documents, source
    code, and other content types without changing the ingestion interface.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[dict] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunk_index += 1

        start += step

    return chunks