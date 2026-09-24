import pytest

import core.memory.conversational_extractor as extractor


def test_extractor_rejects_blank_message():
    with pytest.raises(
        ValueError,
        match="user_message cannot be blank",
    ):
        extractor.extract_memories("   ")


def test_extractor_parses_model_response():
    def fake_chat(**kwargs):
        return {
            "message": {
                "content": """
                {
                    "entities": [
                        {
                            "reference": "kraken",
                            "entity_type": "device",
                            "name": "Kraken"
                        }
                    ],
                    "facts": [
                        {
                            "subject_reference": "kraken",
                            "predicate": "model",
                            "value": "Neptune 4"
                        }
                    ],
                    "relationships": []
                }
                """
            }
        }

    extraction = extractor.extract_memories(
        "My Neptune 4 printer is named Kraken.",
        chat_fn=fake_chat,
    )

    assert extraction.entities[0].name == "Kraken"
    assert extraction.facts[0].predicate == "model"
    assert extraction.facts[0].value == "Neptune 4"


def test_extractor_accepts_empty_model_extraction():
    def fake_chat(**kwargs):
        return {
            "message": {
                "content": """
                {
                    "entities": [],
                    "facts": [],
                    "relationships": []
                }
                """
            }
        }

    extraction = extractor.extract_memories(
        "What kind of printer is Kraken?",
        chat_fn=fake_chat,
    )

    assert extraction.entities == ()
    assert extraction.facts == ()
    assert extraction.relationships == ()


def test_extractor_rejects_invalid_model_json():
    def fake_chat(**kwargs):
        return {
            "message": {
                "content": "Kraken is a printer."
            }
        }

    with pytest.raises(
        ValueError,
        match="Memory extraction is not valid JSON",
    ):
        extractor.extract_memories(
            "Kraken is my printer.",
            chat_fn=fake_chat,
        )


def test_extractor_rejects_invalid_model_references():
    def fake_chat(**kwargs):
        return {
            "message": {
                "content": """
                {
                    "entities": [],
                    "facts": [
                        {
                            "subject_reference": "kraken",
                            "predicate": "model",
                            "value": "Neptune 4"
                        }
                    ],
                    "relationships": []
                }
                """
            }
        }

    with pytest.raises(
        ValueError,
        match="Unresolved fact subject reference",
    ):
        extractor.extract_memories(
            "Kraken is my Neptune 4.",
            chat_fn=fake_chat,
        )