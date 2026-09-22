from pathlib import Path

from core.knowledge.vector_store import VectorStore
from core.memory.manager import MemoryManager


def test_memory_manager_ingests_and_retrieves_knowledge(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    document = tmp_path / "snowball-facts.txt"

    document.write_text(
        (
            "Kraken is my Neptune 4 3D printer. "
            "Kraken is used to manufacture physical printed objects.\n\n"
            "Monty is my augmented snake plant project.\n\n"
            "Brain is PeenQi's clockwork mouse companion."
        ),
        encoding="utf-8",
    )

    result = manager.ingest_document(
        document,
        document_id="snowball-facts",
        chunk_size=100,
        overlap=20,
    )

    assert result["document_id"] == "snowball-facts"
    assert result["chunk_count"] > 0

    context = manager.get_knowledge_context(
        "Which machine can manufacture physical objects?",
        result_count=1,
    )

    assert "Kraken" in context
    assert "snowball-facts.txt" in context


def test_memory_manager_empty_knowledge_returns_empty_context(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    context = manager.get_knowledge_context(
        "What do you know about my printers?"
    )

    assert context == ""

def test_memory_manager_builds_unified_context(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    document = tmp_path / "printer-facts.txt"

    document.write_text(
        "Kraken is my Neptune 4 3D printer.",
        encoding="utf-8",
    )

    manager.ingest_document(
        document,
        document_id="printer-facts",
    )

    context = manager.get_context(
        "Which machine do I use for 3D printing?",
        knowledge_result_count=1,
    )

    assert isinstance(context, dict)
    assert "knowledge" in context
    assert "Kraken" in context["knowledge"]


def test_memory_manager_unified_context_without_knowledge(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    context = manager.get_context(
        "What do you remember?"
    )

    assert context == {
        "episodic": "",
        "knowledge": "",
    }

class FakeEpisodicMemory:
    def __init__(self, memory=None):
        self.memory = memory

    def retrieve(self, query):
        return self.memory


def test_memory_manager_includes_episodic_memory(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    episodic = FakeEpisodicMemory(
        {
            "user_input": "My Neptune 4 printer is named Kraken.",
            "ai_response": "Got it. Kraken is your Neptune 4 printer.",
        }
    )

    manager = MemoryManager(
        store,
        episodic_memory=episodic,
    )

    context = manager.get_context(
        "What is Kraken?"
    )

    assert "Kraken" in context["episodic"]
    assert "My Neptune 4 printer is named Kraken." in context["episodic"]


def test_memory_manager_without_episodic_memory_returns_empty(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    context = manager.get_episodic_context(
        "What is Kraken?"
    )

    assert context == ""


def test_memory_manager_combines_episodic_and_knowledge(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    episodic = FakeEpisodicMemory(
        {
            "user_input": "My Neptune 4 printer is named Kraken.",
            "ai_response": "Kraken is your Neptune 4 printer.",
        }
    )

    manager = MemoryManager(
        store,
        episodic_memory=episodic,
    )

    document = tmp_path / "printer-notes.txt"

    document.write_text(
        "The Neptune 4 is a fused-filament 3D printer.",
        encoding="utf-8",
    )

    manager.ingest_document(
        document,
        document_id="printer-notes",
    )

    context = manager.get_context(
        "Tell me about my Neptune 4 printer.",
        knowledge_result_count=1,
    )

    assert "Kraken" in context["episodic"]
    assert "Neptune 4" in context["knowledge"]

def test_structured_context_preserves_user_provenance(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    episodic = FakeEpisodicMemory(
        {
            "id": "kraken-memory",
            "timestamp": "2026-09-22T12:00:00",
            "user_input": "My Neptune 4 printer is named Kraken.",
            "ai_response": "Got it. Kraken is your Neptune 4 printer.",
        }
    )

    manager = MemoryManager(
        store,
        episodic_memory=episodic,
    )

    context = manager.get_structured_context(
        "What is Kraken?"
    )

    entry = context["episodic"][0]

    assert entry.text == "My Neptune 4 printer is named Kraken."
    assert entry.source.memory_type == "episodic"
    assert entry.source.source_type == "user_statement"
    assert entry.source.authority == "user"
    assert entry.source.source_id == "kraken-memory"


def test_structured_context_preserves_knowledge_provenance(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")
    manager = MemoryManager(store)

    document = tmp_path / "printer-notes.txt"

    document.write_text(
        "The Neptune 4 is a fused-filament 3D printer.",
        encoding="utf-8",
    )

    manager.ingest_document(
        document,
        document_id="printer-notes",
    )

    context = manager.get_structured_context(
        "What machine is used for 3D printing?",
        knowledge_result_count=1,
    )

    entry = context["knowledge"][0]

    assert "Neptune 4" in entry.text
    assert entry.source.memory_type == "semantic"
    assert entry.source.source_type == "knowledge_document"
    assert entry.source.authority == "reference"
    assert entry.source.source_id == "printer-notes"


def test_structured_context_keeps_memory_sources_separate(
    tmp_path: Path,
):
    store = VectorStore(tmp_path / "vectors")

    episodic = FakeEpisodicMemory(
        {
            "user_input": "My Neptune 4 printer is named Kraken.",
            "ai_response": "Kraken is your Neptune 4 printer.",
        }
    )

    manager = MemoryManager(
        store,
        episodic_memory=episodic,
    )

    document = tmp_path / "printer-notes.txt"

    document.write_text(
        "The Neptune 4 is a fused-filament 3D printer.",
        encoding="utf-8",
    )

    manager.ingest_document(
        document,
        document_id="printer-notes",
    )

    context = manager.get_structured_context(
        "Tell me about my Neptune 4 printer.",
        knowledge_result_count=1,
    )

    assert len(context["episodic"]) == 1
    assert len(context["knowledge"]) == 1

    assert context["episodic"][0].source.authority == "user"
    assert context["knowledge"][0].source.authority == "reference"