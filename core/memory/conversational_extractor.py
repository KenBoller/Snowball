import ollama

from core.memory.extraction import (
    MemoryExtraction,
    parse_memory_extraction_json,
)


EXTRACTION_SYSTEM_PROMPT = """
You extract durable structured memories from a user's message.

Return ONLY valid JSON with exactly these top-level fields:
{
  "entities": [],
  "facts": [],
  "relationships": []
}

Entity format:
{
  "reference": "temporary_reference",
  "entity_type": "type",
  "name": "human-readable name"
}

Fact format:
{
  "subject_reference": "temporary_reference",
  "predicate": "predicate",
  "value": "value"
}

Relationship format:
{
  "source_reference": "temporary_reference",
  "relationship": "relationship",
  "target_reference": "temporary_reference"
}

Rules:
- Extract only information explicitly stated by the user.
- Do not infer unstated facts.
- Do not extract information from questions as if it were true.
- Do not extract hypothetical information as fact.
- Do not treat quoted claims as the user's own claims.
- Prefer durable information over temporary conversational state.
- Every fact subject_reference must refer to an entity in entities.
- Every relationship source_reference and target_reference must refer
  to entities in entities.
- References are temporary identifiers used only inside this extraction.
- Do not invent permanent IDs.
- Do not include timestamps, provenance, confidence, status, or
  supersession information.
- If nothing should be remembered, return empty arrays.
- Do not include Markdown fences.
- Do not include explanation or commentary.
- An entity represents a real person, place, device, project, pet,
  organization, object, or other identifiable thing.
- A name is not an entity type.
- When the user gives a thing a proper name, use that proper name as
  the entity name.
- Product models, descriptions, attributes, and properties should
  normally be facts about the named entity rather than separate
  entities.
- Prefer general stable entity types such as person, device, pet,
  project, place, organization, vehicle, or object.
- Hypothetical, conditional, imagined, planned, possible, or uncertain
  situations are not established facts. Do not extract them.
- Statements about temporary feelings, moods, conditions, or immediate
  conversational state should normally not be remembered.
- Do not create an entity unless it participates in at least one
  extracted fact or relationship, except when the user's statement
  explicitly establishes the durable identity or existence of that
  entity, such as naming a pet.
- When the user refers to themselves with "I", "me", "my", or "myself"
  and a relationship requires representing the user, create one person
  entity with reference "user" and name "User".
- A statement such as "Jess is my girlfriend" should represent a
  relationship between Jess and User. Never reference "me" without
  defining an entity.
- When information is explicitly attributed to another person, do not
  extract it for now. Example: "Jess said her favorite color is green"
  should produce an empty extraction. Snowball does not yet have a
  provenance model for attributed conversational claims.
- Use short normalized snake_case predicates.
- Prefer stable predicates that describe the underlying property rather
  than wording from the sentence.
- For nozzle diameter or nozzle size, use the predicate "nozzle_size".

Example:

User:
My Neptune 4 printer is named Kraken.

Output:
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
""".strip()


def extract_memories(
    user_message: str,
    model: str = "qwen2.5:7b-instruct",
    chat_fn=ollama.chat,
) -> MemoryExtraction:
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError(
            "user_message cannot be blank."
        )

    response = chat_fn(
        model=model,
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        format="json",
        options={
            "temperature": 0,
        },
    )

    content = response["message"]["content"]

    return parse_memory_extraction_json(content)