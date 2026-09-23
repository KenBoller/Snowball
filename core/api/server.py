from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.api.chat_api import get_agent, send_message


app = FastAPI(
    title="Snowball AI/OS API",
    version="0.1.0",
)


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