# ☃️ Snowball AI/OS

**Snowball AI/OS** is a modular personal AI system built around persistent memory, local AI models, and a stable API that can eventually connect one Snowball identity across desktop, mobile, automation systems, games, and robotics.

Rather than building Snowball as one enormous AI script, the project is evolving into a small **AI operating layer**.

The core idea is simple:

> One Snowball. Many models, tools, devices, and interfaces.

The language model is not Snowball itself. Models are replaceable reasoning engines. Snowball is the surrounding system: memory, context, identity, routing, permissions, projects, history, tools, and eventually automation.

---

## 🚧 Project Status

Snowball is under active development.

The current focus is deliberately narrow: establish a small, reliable core before reconnecting the larger systems developed during earlier versions of the project.

### Working now

- Python-based Snowball AI core
- Local Ollama model integration
- Persistent local memory
- Memory retrieval across process restarts
- Decision and request routing
- Internal Python chat API
- FastAPI HTTP interface
- Health and capability endpoints
- Automated regression and API contract tests

### Planned

- n8n automation and event orchestration
- Mobile access
- Voice interaction
- Structured and semantic memory
- File and application integrations
- System monitoring
- Game environments
- Vision
- Cloud synchronization
- Robotics / InMoov integration

These systems will be added around the stable core rather than folded into one monolithic application.

---

## 🧠 Architecture

Snowball is being designed as a collection of small layers with clear responsibilities.

```text
                   ┌─────────────────────┐
                   │   Future Clients    │
                   │ Mobile / Web / Bots │
                   └──────────┬──────────┘
                              │
                         HTTP / JSON
                              │
                   ┌──────────▼──────────┐
                   │       FastAPI       │
                   │     core/api/       │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Snowball AI Core  │
                   │      core/ai/       │
                   └──────┬───────┬──────┘
                          │       │
                    ┌─────▼───┐ ┌─▼──────────┐
                    │ Memory  │ │   Ollama   │
                    │ JSONL   │ │   Models   │
                    └─────────┘ └────────────┘

                 Future integration layer:
                          n8n
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
         Calendar        Files         Devices
         Messages       Webhooks       Services
```

The long-term goal is for interfaces and integrations to communicate with Snowball through stable boundaries rather than reaching directly into the AI core.

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
│   └── system/
│       ├── advanced_logger.py
│       ├── config_loader.py
│       └── logger.py
│
├── storage/
│   └── memory/
│       └── local_memory.jsonl
│
├── tests/
├── requirements.txt
├── pytest.ini
└── README.md
```

Runtime memory and other generated storage files are intentionally excluded from Git.

---

## 💾 Persistent Memory

Persistent memory is one of Snowball's core architectural requirements.

Snowball can store conversational interactions locally in:

```text
storage/memory/local_memory.jsonl
```

When Snowball starts again, persisted interactions are loaded back into memory and can be retrieved as context for future conversations.

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

The current memory system uses lightweight local retrieval. More advanced semantic, episodic, project, preference, and structured memory systems are planned as later layers.

---

## 🤖 Local Models

Snowball currently supports local models through **Ollama**.

The AI architecture allows models to act as interchangeable reasoning engines rather than defining Snowball's identity.

Different models may eventually be selected for tasks such as:

- conversation
- planning
- reasoning
- criticism
- summarization
- specialized tool use

This allows the underlying models to evolve without rebuilding Snowball around a particular provider.

---

## 🔌 HTTP API

Snowball exposes a FastAPI service.

Start it from the project root:

```bash
python -m uvicorn core.api.server:app --host 127.0.0.1 --port 8000
```

The API is currently bound to localhost intentionally.

Once running, FastAPI also provides interactive API documentation at:

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
    "persistent_memory"
  ]
}
```

This endpoint describes the currently exposed Snowball service and its core capabilities.

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

---

## ⚙️ Installation

### Requirements

- Python 3.10+
- Ollama for local model-backed conversation
- Git

Clone the repository:

```bash
git clone https://github.com/KenBoller/Snowball.git
cd Snowball
```

Install the core runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Current core dependencies are intentionally kept small:

- FastAPI
- Uvicorn
- Pydantic
- Requests
- Cachetools

Large optional systems such as voice, games, robotics, cloud integrations, and embeddings will use separate dependency groups as they are reintroduced.

---

## 🧪 Tests

Run the complete test suite from the project root:

```bash
pytest -v
```

The suite currently protects several important architectural boundaries:

- persistent memory storage
- memory recovery after reinitialization
- decision-making behavior
- logging
- operation without Ollama
- internal Python chat API
- FastAPI service behavior
- `/status` API contract
- `/chat` HTTP contract

The goal is to make Snowball easier to evolve without silently breaking functionality that already works.

---

## 🧭 Development Direction

Snowball previously grew across many experiments at once: desktop interfaces, games, voice, cloud services, mobile integration, Minecraft, robotics, memory systems, and other prototypes.

The current development strategy is intentionally different.

### Build the Snowball core first.

The near-term architecture is:

```text
Existing Python AI
        │
        ▼
Persistent Memory
        │
        ▼
FastAPI
        │
        ▼
n8n
        │
        ├── Mobile
        ├── Messages
        ├── Calendar
        ├── Reminders
        ├── Files
        └── Webhooks
```

Only components that make Snowball uniquely Snowball should require custom implementation.

Commodity integration work can be delegated to established tools and services.

---

## 🧩 Design Principles

### One Snowball

Desktop, mobile, games, automation systems, and future robotics should not create separate Snowball instances with separate identities.

They should be different interfaces to the same underlying system.

### Memory is infrastructure

Memory is not an optional chat feature. Persistent context is part of Snowball's foundation.

### Models are replaceable

Snowball should not depend on one LLM vendor, model family, or inference environment.

### Stable boundaries

External systems should interact with defined APIs rather than becoming tightly coupled to Snowball's internal implementation.

### Build the unique parts

Do not build everything Snowball can use.

Build the part that makes it Snowball.

---

## 🗺️ Long-Term Vision

Future Snowball layers may include:

- working memory
- episodic memory
- semantic facts
- project memory
- preference memory
- procedural memory
- automation history
- game strategy memory
- file understanding
- system and hardware monitoring
- mobile interfaces
- voice interaction
- opt-in vision
- calendar and communication integrations
- smart-device control
- game-playing environments
- InMoov robotics

The eventual goal is not a collection of disconnected AI applications.

It is one persistent intelligence layer capable of interacting through many environments.

---

## 🔐 Security Philosophy

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

## 👤 Author

**Kenneth Lloyd Boller**

AI Systems Builder • Automation Engineer • Creator of Snowball AI/OS

---

## ☃️ Current Mission

Snowball does not need every planned feature at once.

It needs a small core that can reliably:

```text
Remember
   ↓
Think
   ↓
Respond
   ↓
Expose a stable interface
   ↓
Grow without being rebuilt
```

Build the foundation first.

Then let the snowball roll.