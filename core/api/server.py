from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.api.chat_api import get_agent, send_message
from core.memory.semantic import MemorySource
from core.memory.structured import Entity, Fact, Relationship

app = FastAPI(
    title="Snowball AI/OS API",
    version="0.1.0",
)


class MemoryEntityRequest(BaseModel):
    entity_id: str
    entity_type: str
    name: str
    metadata: dict[str, object] | None = None

    @field_validator(
        "entity_id",
        "entity_type",
        "name",
    )
    @classmethod
    def fields_cannot_be_blank(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("field cannot be blank.")

        return value


class MemoryFactRequest(BaseModel):
    fact_id: str
    subject_id: str
    predicate: str
    value: object
    source_type: str
    authority: str
    learned_at: str
    metadata: dict[str, object] | None = None

    @field_validator(
        "fact_id",
        "subject_id",
        "predicate",
        "source_type",
        "authority",
        "learned_at",
    )
    @classmethod
    def fields_cannot_be_blank(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("field cannot be blank.")

        return value


class MemoryRelationshipRequest(BaseModel):
    relationship_id: str
    source_entity_id: str
    relationship: str
    target_entity_id: str
    source_type: str
    authority: str
    learned_at: str
    metadata: dict[str, object] | None = None

    @field_validator(
        "relationship_id",
        "source_entity_id",
        "relationship",
        "target_entity_id",
        "source_type",
        "authority",
        "learned_at",
    )
    @classmethod
    def fields_cannot_be_blank(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("field cannot be blank.")

        return value


class MemoryFactSupersedeRequest(BaseModel):
    fact_id: str
    value: object
    source_type: str
    authority: str
    learned_at: str
    metadata: dict[str, object] | None = None

    @field_validator(
        "fact_id",
        "source_type",
        "authority",
        "learned_at",
    )
    @classmethod
    def fields_cannot_be_blank(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("field cannot be blank.")

        return value


class MemoryRelationshipSupersedeRequest(BaseModel):
    relationship_id: str
    target_entity_id: str
    source_type: str
    authority: str
    learned_at: str
    metadata: dict[str, object] | None = None

    @field_validator(
        "relationship_id",
        "target_entity_id",
        "source_type",
        "authority",
        "learned_at",
    )
    @classmethod
    def fields_cannot_be_blank(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("field cannot be blank.")

        return value


class KnowledgeIngestRequest(BaseModel):
    path: str
    document_id: str | None = None
    metadata: dict[str, object] | None = None

    @field_validator("path")
    @classmethod
    def path_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path cannot be blank.")

        return value

    @field_validator("document_id")
    @classmethod
    def document_id_cannot_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("document_id cannot be blank.")

        return value


class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, gt=0)

    @field_validator("query")
    @classmethod
    def query_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank.")

        return value


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Snowball AI/OS",
    }


@app.get("/status")
def status() -> dict[str, object]:
    return {
        "service": "Snowball AI/OS",
        "version": "0.1.0",
        "status": "ok",
        "capabilities": [
            "chat",
            "persistent_memory",
            "semantic_knowledge",
            "knowledge_provenance",
            "structured_memory",
        ],
    }


@app.post("/memory/entities")
def create_memory_entity(
    request: MemoryEntityRequest,
) -> dict:
    agent = get_agent()

    entity = Entity(
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        name=request.name,
        metadata=request.metadata,
    )

    agent.memory_manager.save_entity(entity)

    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "name": entity.name,
        "metadata": entity.metadata,
    }


@app.post("/memory/relationships")
def create_memory_relationship(
    request: MemoryRelationshipRequest,
) -> dict:
    agent = get_agent()

    source = MemorySource(
        memory_type="structured",
        source_type=request.source_type,
        authority=request.authority,
    )

    relationship = Relationship(
        relationship_id=request.relationship_id,
        source_entity_id=request.source_entity_id,
        relationship=request.relationship,
        target_entity_id=request.target_entity_id,
        source=source,
        learned_at=request.learned_at,
        metadata=request.metadata,
    )

    agent.memory_manager.save_relationship(
        relationship
    )

    return {
        "relationship_id": relationship.relationship_id,
        "source_entity_id": relationship.source_entity_id,
        "relationship": relationship.relationship,
        "target_entity_id": relationship.target_entity_id,
        "source": {
            "memory_type": relationship.source.memory_type,
            "source_type": relationship.source.source_type,
            "authority": relationship.source.authority,
        },
        "learned_at": relationship.learned_at,
        "status": relationship.status,
        "supersedes": relationship.supersedes,
        "metadata": relationship.metadata,
    }


@app.post("/memory/relationships/{relationship_id}/supersede")
def supersede_memory_relationship(
    relationship_id: str,
    request: MemoryRelationshipSupersedeRequest,
) -> dict:
    agent = get_agent()

    old_relationship = agent.memory_manager.get_relationship(
        relationship_id
    )

    if old_relationship is None:
        raise HTTPException(
            status_code=404,
            detail=f"Relationship not found: {relationship_id}",
        )

    source = MemorySource(
        memory_type="structured",
        source_type=request.source_type,
        authority=request.authority,
    )

    replacement = Relationship(
        relationship_id=request.relationship_id,
        source_entity_id=old_relationship.source_entity_id,
        relationship=old_relationship.relationship,
        target_entity_id=request.target_entity_id,
        source=source,
        learned_at=request.learned_at,
        metadata=request.metadata,
    )

    new_relationship = (
        agent.memory_manager.supersede_relationship(
            old_relationship,
            replacement,
        )
    )

    return {
        "relationship_id": new_relationship.relationship_id,
        "source_entity_id": new_relationship.source_entity_id,
        "relationship": new_relationship.relationship,
        "target_entity_id": new_relationship.target_entity_id,
        "source": {
            "memory_type": new_relationship.source.memory_type,
            "source_type": new_relationship.source.source_type,
            "authority": new_relationship.source.authority,
        },
        "learned_at": new_relationship.learned_at,
        "status": new_relationship.status,
        "supersedes": new_relationship.supersedes,
        "metadata": new_relationship.metadata,
    }


@app.post("/memory/facts")
def create_memory_fact(
    request: MemoryFactRequest,
) -> dict:
    agent = get_agent()

    source = MemorySource(
        memory_type="structured",
        source_type=request.source_type,
        authority=request.authority,
    )

    fact = Fact(
        fact_id=request.fact_id,
        subject_id=request.subject_id,
        predicate=request.predicate,
        value=request.value,
        source=source,
        learned_at=request.learned_at,
        metadata=request.metadata,
    )

    agent.memory_manager.save_fact(fact)

    return {
        "fact_id": fact.fact_id,
        "subject_id": fact.subject_id,
        "predicate": fact.predicate,
        "value": fact.value,
        "source": {
            "memory_type": fact.source.memory_type,
            "source_type": fact.source.source_type,
            "authority": fact.source.authority,
        },
        "learned_at": fact.learned_at,
        "status": fact.status,
        "supersedes": fact.supersedes,
        "metadata": fact.metadata,
    }


@app.post("/memory/facts/{fact_id}/supersede")
def supersede_memory_fact(
    fact_id: str,
    request: MemoryFactSupersedeRequest,
) -> dict:
    agent = get_agent()

    old_fact = agent.memory_manager.get_fact(fact_id)

    if old_fact is None:
        raise HTTPException(
            status_code=404,
            detail=f"Fact not found: {fact_id}",
        )

    source = MemorySource(
        memory_type="structured",
        source_type=request.source_type,
        authority=request.authority,
    )

    replacement = Fact(
        fact_id=request.fact_id,
        subject_id=old_fact.subject_id,
        predicate=old_fact.predicate,
        value=request.value,
        source=source,
        learned_at=request.learned_at,
        metadata=request.metadata,
    )

    new_fact = agent.memory_manager.supersede_fact(
        old_fact,
        replacement,
    )

    return {
        "fact_id": new_fact.fact_id,
        "subject_id": new_fact.subject_id,
        "predicate": new_fact.predicate,
        "value": new_fact.value,
        "source": {
            "memory_type": new_fact.source.memory_type,
            "source_type": new_fact.source.source_type,
            "authority": new_fact.source.authority,
        },
        "learned_at": new_fact.learned_at,
        "status": new_fact.status,
        "supersedes": new_fact.supersedes,
        "metadata": new_fact.metadata,
    }


@app.get("/knowledge")
def knowledge() -> dict[str, object]:
    agent = get_agent()
    documents = agent.memory_manager.list_knowledge_documents()

    return {
        "documents": documents,
        "document_count": len(documents),
        "chunk_count": sum(
            document["chunk_count"]
            for document in documents
        ),
    }


@app.post("/knowledge")
def ingest_knowledge(request: KnowledgeIngestRequest) -> dict:
    agent = get_agent()

    try:
        return agent.memory_manager.ingest_document(
            request.path,
            document_id=request.document_id,
            metadata=request.metadata,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        if str(exc).startswith("Unsupported document type:"):
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        raise


@app.post("/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest) -> dict:
    agent = get_agent()

    results = agent.memory_manager.search_knowledge(
        request.query,
        result_count=request.limit,
    )

    return {
        "query": request.query,
        "results": results,
        "result_count": len(results),
    }


@app.get("/knowledge/{document_id}")
def knowledge_document(document_id: str) -> dict:
    agent = get_agent()
    document = agent.memory_manager.get_knowledge_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge document not found.",
        )

    return document


@app.delete("/knowledge/{document_id}")
def delete_knowledge_document(document_id: str) -> dict:
    agent = get_agent()

    deleted = agent.memory_manager.delete_knowledge_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Knowledge document not found.",
        )

    return {
        "document_id": document_id,
        "deleted": True,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    response = send_message(request.message)
    return ChatResponse(response=response)
