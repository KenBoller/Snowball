# ☃️ Snowball AI/OS

**Snowball AI/OS** is a modular personal AI system built around persistent memory, semantic knowledge, local AI models, and stable interfaces that can connect one Snowball identity across desktop, mobile, automation systems, games, and eventually robotics.

Rather than building Snowball as one enormous AI script, the project is evolving into a small **AI operating layer**.

The core idea is simple:

> One Snowball. Many models, tools, devices, and interfaces.

The language model is not Snowball itself. Models are replaceable reasoning engines. Snowball is the surrounding system: memory, context, identity, routing, permissions, projects, history, tools, knowledge, and automation.

---

## 🚧 Project Status

Snowball is under active development.

The current focus is deliberately narrow: grow outward from a small, tested core without repeatedly rebuilding Snowball from scratch.

### Working now

- Python-based Snowball AI core
- Local Ollama model integration
- Persistent episodic memory
- Memory retrieval across process restarts
- Unified episodic and semantic memory coordination
- Semantic document ingestion and retrieval
- PDF, TXT, Markdown, and DOCX knowledge ingestion
- DOCX paragraph and table preservation
- Ollama embeddings using `nomic-embed-text`
- Persistent embedded ChromaDB vector storage
- Provenance-aware knowledge retrieval
- Historical-vs-current knowledge reasoning
- Safe document reingestion using stable document IDs
- Decision and request routing
- Internal Python chat API
- FastAPI HTTP interface
- Health and capability endpoints
- n8n chat gateway
- Browser-based web/mobile chat client
- Verified Android local-network access
- Automated regression, knowledge, memory, and API contract tests

### Planned / future layers

- Voice interaction
- Broader file and application integrations
- Calendar and messaging integrations
- System and hardware monitoring
- Game-playing environments
- Cross-game strategy learning
- Opt-in vision
- Cloud synchronization
- Smart-device control
- Robotics / InMoov integration

These systems will be added around the stable core rather than folded into one monolithic application.

---

## 🧠 Architecture

Snowball is being developed as a collection of small layers with clear responsibilities.

```text
                ┌─────────────────────────┐
                │    Web / Mobile Client  │
                │    Future Bots / Apps   │
                └────────────┬────────────┘
                             │
                         Webhook / HTTP
                             │
                ┌────────────▼────────────┐
                │          n8n            │
                │ Integration / Automation│
                └────────────┬────────────┘
                             │
                         HTTP / JSON
                             │
                ┌────────────▼────────────┐
                │        FastAPI          │
                │        core/api/        │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │    Snowball AI Core     │
                │       core/ai/          │
                └──────┬─────────┬────────┘
                       │         │
             ┌─────────▼───┐ ┌───▼─────────┐
             │   Memory    │ │   Ollama    │
             │   Manager   │ │   Models    │
             └──────┬──────┘ └─────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
    ┌─────▼──────┐      ┌──────▼────────┐
    │ Episodic   │      │   Semantic    │
    │ JSONL      │      │   Knowledge   │
    └────────────┘      └──────┬────────┘
                               │
                        ┌──────▼────────┐
                        │ Embedded      │
                        │ ChromaDB      │
                        └───────────────┘
```

The working local-network chat path is:

```text
Web / Android Browser
        │
        ▼
      n8n
        │
        ▼
     FastAPI
        │
        ▼
 Snowball AI Core
        │
   ┌────┴────┐
   ▼         ▼
Memory    Ollama
        │
        ▼
    Response
```

Interfaces and integrations communicate with Snowball through stable boundaries rather than reaching directly into the AI core.

n8n acts as an integration or "nervous system" layer. Snowball's identity, reasoning, memory, and knowledge remain responsibilities of the Snowball core.

---

## 📁 Current Project Structure

```text
Snowball/
├── core/
│   ├── ai/
│   │   ├── chat.py
│   │   ├── decision_maker.py
│   │   ├── memory.py
│   │   ├── orchestrator.py
│   │   └── router.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat_api.py
│   │   └── server.py
│   │
│   ├── knowledge/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   ├── rag.py
│   │   └── vector_store.py
│   │
│   ├── memory/
│   │   ├── episodic.py
│   │   ├── manager.py
│   │   ├── retrieval.py
│   │   └── semantic.py
│   │
│   └── system/
│       ├── advanced_logger.py
│       ├── config_loader.py
│       └── logger.py
│
├── docs/
│   ├── CURRENT_STATE.md
│   └── SECURITY.md
│
├── integrations/
│   └── n8n/
│       ├── README.md
│       └── workflows/
│           └── snowball-chat-gateway.json
│
├── storage/
│   ├── knowledge/
│   ├── memory/
│   │   └── local_memory.jsonl
│   └── vectors/
│
├── tests/
├── tools/
│   └── importers/
│
├── web/
│   └── client/
│       └── index.html
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
└── run_snowball.py
```

Runtime memory, vector databases, caches, and other generated storage are intentionally excluded from Git where appropriate.

`docs/CURRENT_STATE.md` is the canonical human-readable description of Snowball's currently implemented and verified state.

---

## 💾 Memory and Knowledge

Memory is one of Snowball's core architectural requirements.

Snowball currently has two complementary persistent information systems.

### Episodic memory

Conversational interactions are persisted locally in:

```text
storage/memory/local_memory.jsonl
```

Persisted interactions survive Snowball process shutdown and restart and can be retrieved as context for later conversations.

For example:

```text
User:
My Neptune 4 printer is named Kraken.

Snowball:
Got it. Kraken is your Neptune 4 printer.
```

After Snowball is stopped and restarted:

```text
User:
What is Kraken?

Snowball:
Kraken is your Neptune 4 printer.
```

This restart behavior is protected by an automated regression test.

### Semantic knowledge

Snowball also has persistent semantic document knowledge.

Supported formats currently include:

- PDF
- TXT
- Markdown
- DOCX

Documents are extracted, divided into chunks, embedded through Ollama using `nomic-embed-text`, and stored in an embedded ChromaDB vector store under:

```text
storage/vectors/
```

When Snowball receives a question, it can semantically retrieve relevant document chunks and supply them to the conversational model as context.

`MemoryManager` in:

```text
core/memory/manager.py
```

provides a unified coordination layer for episodic memory and semantic knowledge.

---

## 🏷️ Knowledge Provenance

Semantic knowledge can carry provenance metadata including:

```text
SOURCE
TYPE
AUTHORITY
DATE
```

This allows Snowball to distinguish between different kinds of retrieved knowledge.

For example:

```text
historical_reference
```

identifies material useful for understanding Snowball's history, earlier designs, experiments, and plans.

```text
current_reference
```

identifies material intended to describe the current implementation.

When retrieved sources differ, Snowball is instructed to consider which authority and time period best match the question rather than assuming every retrieved document describes the present.

This allows historical Snowball documentation to remain useful without automatically becoming current implementation truth.

---

## 📚 Knowledge Reingestion

Knowledge documents use stable document identifiers.

Reingesting a document with the same document ID replaces its previous vectors instead of continually creating duplicate copies.

For non-empty documents, replacement currently occurs in this order:

1. Extract the document.
2. Create chunks.
3. Generate replacement embeddings successfully.
4. Remove the previous vectors for that document.
5. Store the replacement vectors.

This prevents an embedding failure from deleting the previously working version.

The replacement operation is not yet fully transactional if a failure occurs after old vectors are removed but during the final vector-store write.

---

## 🤖 Local Models

Snowball currently supports local models through **Ollama**.

The AI architecture allows models to act as interchangeable reasoning engines rather than defining Snowball's identity.

Different models can be used for responsibilities such as:

- conversation
- planning
- reasoning
- criticism
- summarization
- specialized tool use

The current semantic knowledge system also uses Ollama for local embeddings.

This allows models to evolve without rebuilding Snowball's identity, memory, knowledge, or surrounding architecture around a particular provider.

---

## 🔌 HTTP API

Snowball exposes a FastAPI service through:

```text
core/api/server.py
```

Start it from the project root:

```bash
python -m uvicorn core.api.server:app --host 127.0.0.1 --port 8000
```

For LAN development, the bind address can be changed deliberately when another local device or Docker-hosted integration needs to reach the service.

Do not expose the development API directly to the public internet without authentication and encrypted transport.

Once running, FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

and an OpenAPI schema at:

```text
http://127.0.0.1:8000/openapi.json
```

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "Snowball AI/OS"
}
```

This is a lightweight service liveness check.

### Status

```http
GET /status
```

Example response:

```json
{
  "service": "Snowball AI/OS",
  "version": "0.1.0",
  "status": "ok",
  "capabilities": [
    "chat",
    "persistent_memory",
    "semantic_knowledge",
    "knowledge_provenance"
  ]
}
```

This endpoint describes capabilities exposed by the Snowball service itself.

External integrations such as n8n and the browser client are intentionally not represented as API-service capabilities.

### Chat

```http
POST /chat
```

Request:

```json
{
  "message": "What is Kraken?"
}
```

Response:

```json
{
  "response": "Kraken is your Neptune 4 printer."
}
```

The `/chat` endpoint passes the message through Snowball's internal chat API and AI core.

It uses the same Snowball memory and knowledge systems rather than creating a separate AI identity for HTTP clients.

---

## ⚡ n8n Integration

n8n currently acts as Snowball's external integration and automation layer.

The current development environment uses Docker Desktop with:

```text
Container: snowball-n8n
Volume:    snowball_n8n_data
```

A version-controlled chat gateway workflow exists under:

```text
integrations/n8n/workflows/
```

The current path is:

```text
External Client
      │
      ▼
 n8n Webhook
      │
      ▼
FastAPI /chat
      │
      ▼
 Snowball
```

The Docker-hosted n8n instance can reach the Windows-hosted Snowball API using `host.docker.internal`.

The current n8n environment is intended for trusted local development.

---

## 📱 Web and Mobile Access

Snowball currently includes a browser-based chat client at:

```text
web/client/index.html
```

The client sends messages through the n8n Snowball Chat Gateway and displays the returned response.

The complete path:

```text
Browser
   │
   ▼
  n8n
   │
   ▼
FastAPI
   │
   ▼
Snowball
   │
   ▼
Memory / Knowledge / Ollama
   │
   ▼
Response
```

has been successfully exercised from an Android browser over the local network.

The current web client is a local-network development interface rather than a public authenticated service.

---

## ⚙️ Installation

### Requirements

- Python 3.10+
- Ollama
- Git

Docker Desktop is additionally required for the current n8n development integration.

Clone the repository:

```bash
git clone https://github.com/KenBoller/Snowball.git
cd Snowball
```

Create and activate a virtual environment, then install the runtime dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The current runtime includes components for:

- FastAPI / Uvicorn
- HTTP requests and configuration support
- Ollama integration
- embedded ChromaDB vector storage
- PDF ingestion
- DOCX ingestion

Additional systems such as voice, games, robotics, and cloud integrations can use separate dependency groups as they are introduced.

---

## 🧪 Tests

Run the complete test suite from the project root:

```bash
pytest -v
```

As of **2026-09-23**, the complete suite contains:

```text
80 passed, 2 warnings
```

The current suite protects behavior including:

- persistent episodic memory
- memory recovery across reinitialization
- restart-safe memory retrieval
- unified memory coordination
- semantic memory structures
- knowledge chunking
- local embeddings
- document ingestion
- DOCX ingestion
- DOCX table ordering
- vector storage
- semantic retrieval
- RAG context generation
- provenance propagation
- safe knowledge reingestion
- semantic context reaching the Snowball model prompt
- decision-making behavior
- logging
- operation when Ollama is unavailable
- internal Python chat behavior
- FastAPI service behavior
- `/status` API contract
- `/chat` HTTP contract

The two current warnings are known dependency deprecation warnings involving Starlette/httpx TestClient behavior and the AnyIO BlockingPortal alias.

The goal is to make Snowball easier to evolve without silently breaking functionality that already works.

---

## 🧭 Development Direction

Snowball previously grew across many experiments at once: desktop interfaces, games, voice, cloud services, mobile integration, Minecraft, robotics, memory systems, and other prototypes.

The current development strategy is intentionally different.

### Build outward from the tested Snowball core.

The architecture has progressed through:

```text
Python AI Core
      │
      ▼
Persistent Episodic Memory
      │
      ▼
FastAPI
      │
      ▼
n8n Gateway
      │
      ▼
Web / Mobile Client
```

and:

```text
Episodic Memory
       │
       ├─────────────┐
       │             │
       ▼             ▼
 Conversation    Semantic Knowledge
                     │
                     ▼
                  ChromaDB
                     │
                     ▼
               Provenance-aware
                  Retrieval
```

Future integrations can now grow around these tested boundaries.

Only components that make Snowball uniquely Snowball should require custom implementation.

Commodity integration work can be delegated to established tools and services.

---

## 🧩 Design Principles

### One Snowball

Desktop, mobile, games, automation systems, and future robotics should not create separate Snowball identities.

They should be different interfaces to the same underlying system.

### Memory is infrastructure

Memory is not an optional chat feature.

Persistent episodic context and semantic knowledge are part of Snowball's foundation.

### Models are replaceable

Snowball should not depend on one LLM vendor, model family, or inference environment.

### Retrieval relevance is not authority

A semantically relevant result is not automatically the most trustworthy answer.

Source, authority, provenance, and time period matter.

### Stable boundaries

External systems should interact with defined APIs rather than becoming tightly coupled to Snowball's internal implementation.

### Build the unique parts

Do not build everything Snowball can use.

Build the part that makes it Snowball.

---

## 🗺️ Long-Term Vision

Snowball's current episodic and semantic systems establish the beginning of a broader long-term memory architecture.

Future layers may include:

- richer working memory
- structured personal facts
- project memory
- preference memory
- procedural memory
- automation history
- game strategy memory
- broader file understanding
- system and hardware monitoring
- mature voice interaction
- opt-in vision
- calendar and communication integrations
- smart-device control
- autonomous game-playing environments
- learned cross-game strategy transfer
- cloud synchronization
- InMoov robotics

The eventual goal is not a collection of disconnected AI applications.

It is one persistent intelligence layer capable of interacting through many environments.

---

## 🔐 Security

Snowball's current network services are intended for trusted local development.

FastAPI and n8n should not be exposed directly to the public internet without appropriate authentication and encrypted transport.

Snowball currently uses ChromaDB through its embedded local `PersistentClient` rather than intentionally exposing a ChromaDB HTTP service.

The currently installed ChromaDB dependency has known published security advisories. The local embedded architecture reduces exposure to several server-oriented attack paths, but this remains documented dependency risk rather than a claim that the system is vulnerability-free.

Current security notes are maintained in:

```text
docs/SECURITY.md
```

As Snowball gains access to files, devices, accounts, and automation systems, permissions will become increasingly important.

Future integrations should distinguish actions such as:

```text
READ
WRITE
EXECUTE
DELETE
EXTERNAL
PURCHASE
DEVICE
SENSITIVE
```

High-consequence actions should require explicit authorization.

Snowball should gain capabilities deliberately rather than receiving unrestricted access simply because an integration exists.

---

## 📖 Sources of Truth

Snowball contains both current and historical project documentation.

For questions about the current implementation, evidence should generally be considered in this order:

1. Current executable code and automated tests
2. `docs/CURRENT_STATE.md`
3. Current technical documentation
4. Historical project documentation for historical context

Historical documentation remains valuable for understanding Snowball's evolution, earlier experiments, and long-term vision, but should not override verified current implementation behavior.

`docs/CURRENT_STATE.md` is maintained as a living reference and should describe implemented reality rather than future intent.

---

## 👤 Author

**Kenneth Lloyd Boller**

AI Systems Builder • Automation Engineer • Creator of Snowball AI/OS

---

## ☃️ Current Mission

Snowball does not need every planned feature at once.

It needs a core that can reliably:

```text
Remember
   ↓
Retrieve
   ↓
Reason
   ↓
Respond
   ↓
Expose stable interfaces
   ↓
Connect to other systems
   ↓
Grow without being rebuilt
```

Build the foundation first.

Then let the snowball roll.