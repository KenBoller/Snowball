from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from core.api.chat_api import send_message


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
        ],
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    response = send_message(request.message)
    return ChatResponse(response=response)