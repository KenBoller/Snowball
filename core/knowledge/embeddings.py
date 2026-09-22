from __future__ import annotations

import os

import ollama


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding_model() -> str:
    return os.getenv(
        "SNOWBALL_EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )


def create_embedding(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    model = get_embedding_model()

    response = ollama.embed(
        model=model,
        input=text,
    )

    embeddings = response.embeddings

    if not embeddings:
        raise RuntimeError(
            f"Ollama returned no embedding for model '{model}'."
        )

    return list(embeddings[0])


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Embedding inputs cannot contain empty text.")

    model = get_embedding_model()

    response = ollama.embed(
        model=model,
        input=texts,
    )

    embeddings = response.embeddings

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected number of embeddings."
        )

    return [list(embedding) for embedding in embeddings]