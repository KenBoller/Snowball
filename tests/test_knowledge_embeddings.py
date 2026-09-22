import pytest

from core.knowledge.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    create_embedding,
    create_embeddings,
    get_embedding_model,
)


def test_default_embedding_model(monkeypatch):
    monkeypatch.delenv(
        "SNOWBALL_EMBEDDING_MODEL",
        raising=False,
    )

    assert get_embedding_model() == DEFAULT_EMBEDDING_MODEL


def test_embedding_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv(
        "SNOWBALL_EMBEDDING_MODEL",
        "custom-model",
    )

    assert get_embedding_model() == "custom-model"


def test_create_embedding_returns_vector():
    embedding = create_embedding(
        "Kraken is my Neptune 4 printer."
    )

    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(value, float) for value in embedding)


def test_create_embeddings_returns_one_vector_per_input():
    texts = [
        "Kraken is my Neptune 4 printer.",
        "Ferris is my Ender 3 S1 Pro printer.",
    ]

    embeddings = create_embeddings(texts)

    assert len(embeddings) == 2
    assert all(len(vector) == 768 for vector in embeddings)


def test_create_embedding_rejects_empty_text():
    with pytest.raises(ValueError):
        create_embedding("   ")


def test_create_embeddings_empty_list_returns_empty_list():
    assert create_embeddings([]) == []


def test_create_embeddings_rejects_empty_items():
    with pytest.raises(ValueError):
        create_embeddings(
            [
                "Snowball",
                "",
            ]
        )