from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.api.chat_api import get_agent, send_message


app = FastAPI(
    title="Snowball AI/OS API",
    version="0.1.0",
)

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
        ],
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
