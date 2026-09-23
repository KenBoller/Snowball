# Snowball AI/OS Current State

**State date:** 2026-09-23  
**Document role:** Canonical current-state reference  
**Authority:** `current_reference`  
**Project:** Snowball AI/OS

---

## Purpose

This document describes the currently implemented and verified state of
Snowball AI/OS.

It is intended to answer questions about what Snowball is and what Snowball
can do now.

Historical project documents may describe earlier implementations,
experiments, abandoned approaches, future plans, or capabilities that are not
part of the current system. Those documents remain useful as historical
references, but they should not automatically be treated as descriptions of
the current implementation.

For questions about Snowball's present implementation, this current-state
reference should normally take precedence over historical project
documentation.

---

## Current Architecture

Snowball AI/OS is being developed as a modular personal AI system rather than
as one monolithic application.

The current architecture separates Snowball's identity and supporting systems
from the language models used for reasoning.

The core design principle is:

> One Snowball. Many models, tools, devices, and interfaces.

Snowball currently consists of several cooperating layers:

1. Python AI core
2. Persistent episodic memory
3. Semantic knowledge and retrieval
4. Local Ollama models
5. FastAPI HTTP interface
6. n8n automation gateway
7. Web/mobile chat client

External systems are intended to communicate with Snowball through stable
interfaces rather than becoming tightly coupled to the AI core.

---

## AI Core

The primary conversational implementation is located in:

`core/ai/chat.py`

The current AI layer includes:

- `SnowballAI`
- request routing
- decision-making components
- persistent memory integration
- semantic knowledge retrieval
- model fallback behavior
- deterministic context injection
- provenance-aware knowledge context

Snowball currently uses local Ollama models as reasoning engines.

The language model is not treated as Snowball's identity. Models can be
replaced while Snowball's memory, context, history, architecture, and other
supporting systems remain persistent.

---

## Episodic Memory

Snowball has persistent local conversational memory.

The primary local memory file is:

`storage/memory/local_memory.jsonl`

Interactions stored there survive process shutdown and restart.

This behavior has been verified using restart tests such as remembering that
Kraken is the user's Neptune 4 printer and recalling that fact after Snowball
is restarted.

The existing conversational memory implementation is wrapped by the newer
memory architecture rather than being replaced with an unrelated second
memory system.

The compatibility layer is located in:

`core/memory/episodic.py`

---

## Unified Memory Layer

Snowball has a unified memory coordination layer located in:

`core/memory/manager.py`

`MemoryManager` coordinates episodic conversational memory and semantic
knowledge retrieval.

The conversational AI can retrieve both types of context for a user request
and provide relevant information to the model before generating a response.

This allows Snowball to combine remembered interactions with information from
ingested project and knowledge documents.

---

## Semantic Knowledge System

Snowball currently has a persistent semantic knowledge system.

Its implementation is divided across:

- `core/knowledge/ingestion.py`
- `core/knowledge/chunking.py`
- `core/knowledge/embeddings.py`
- `core/knowledge/vector_store.py`
- `core/knowledge/rag.py`
- `core/memory/retrieval.py`
- `core/memory/semantic.py`

Supported knowledge document formats currently include:

- PDF
- TXT
- Markdown
- DOCX

DOCX ingestion preserves normal paragraph content and table content in document
order.

Documents are divided into chunks before embedding and storage.

Snowball currently generates embeddings through Ollama and uses
`nomic-embed-text` for semantic knowledge embeddings.

The persistent vector store uses ChromaDB through its embedded
`PersistentClient`.

The production vector store is located under:

`storage/vectors/`

Runtime vector storage is excluded from Git.

---

## Knowledge Retrieval

When a user asks a question, Snowball can create an embedding for the question
and search its semantic vector store for relevant document chunks.

Retrieved semantic knowledge can then be inserted into Snowball's model
context.

The RAG layer retrieves information but does not independently generate the
final answer. Snowball's conversational system remains responsible for
reasoning over the retrieved context and producing the response.

---

## Knowledge Provenance

Semantic knowledge supports provenance metadata.

Retrieved knowledge can expose information including:

- `SOURCE`
- `TYPE`
- `AUTHORITY`
- `DATE`

These fields are included in the context presented to the conversational
model.

Authority and time period are used to help distinguish different kinds of
knowledge.

For example, knowledge marked:

`historical_reference`

should be treated as evidence about Snowball's history, previous designs,
plans, or earlier states. It should not automatically be interpreted as a
description of Snowball's current implementation.

Knowledge marked as a current-state reference is intended to describe the
currently implemented system.

When sources conflict, Snowball is instructed to consider which authority and
time period best match the question being asked rather than blindly assuming
that every retrieved source describes the present.

---

## Knowledge Reingestion

Knowledge documents use stable document identifiers.

Reingesting a document with the same document ID replaces the previous vectors
for that document rather than creating duplicate copies indefinitely.

The replacement process currently:

1. Extracts the document.
2. Creates chunks.
3. Successfully creates replacement embeddings.
4. Removes the previous vectors for that document.
5. Stores the replacement vectors.

This ordering prevents an embedding failure from deleting the previously
working copy of the document.

Replacing a document with an empty document removes the previous vectors for
that document.

A failure occurring after old vectors have been removed but during the final
vector-store write is not currently guaranteed to be fully transactional.

---

## Historical Project Knowledge

Snowball currently contains an ingested historical archive synthesis describing
earlier Snowball development.

Its document ID is:

`snowball-archive-synthesis-2026-09-23`

The source synthesis covers archived material from approximately:

`2024-10-02` through `2025-03-06`

It is stored with historical-reference authority.

The archive synthesis is useful for understanding Snowball's origins, previous
architecture, earlier experiments, and long-term ideas.

It is not authoritative evidence that every capability described in those
historical materials exists in the current implementation.

---

## FastAPI Interface

Snowball exposes an HTTP API through:

`core/api/server.py`

The current API version identifies itself as:

`0.1.0`

Currently implemented endpoints include:

### `GET /health`

Provides a lightweight service-liveness response.

### `GET /status`

Reports basic Snowball service information and currently advertises the
capabilities:

- `chat`
- `persistent_memory`

This capability list does not yet describe every capability implemented
elsewhere in the current Snowball codebase.

### `POST /chat`

Accepts a user message and returns Snowball's response.

The HTTP chat path uses the same underlying Snowball AI system and persistent
memory rather than creating an independent Snowball identity.

---

## n8n Integration

n8n currently acts as Snowball's external automation and integration layer.

The current development installation runs through Docker Desktop.

The container is named:

`snowball-n8n`

Its persistent Docker volume is:

`snowball_n8n_data`

A version-controlled Snowball Chat Gateway workflow exists under:

`integrations/n8n/workflows/`

The working request path is:

External client  
→ n8n webhook  
→ Snowball FastAPI `/chat`  
→ Snowball AI core  
→ response through n8n

The Docker-hosted n8n instance reaches the Windows-hosted Snowball API through
`host.docker.internal`.

Persistent Snowball memory has been verified through this gateway.

n8n is treated as an integration or "nervous system" layer. Snowball's AI
behavior, identity, reasoning, and memory remain responsibilities of the
Snowball core.

---

## Web and Mobile Access

A web chat client currently exists at:

`web/client/index.html`

The client provides a browser-based Snowball chat interface.

It sends messages to the Snowball n8n chat webhook and displays returned
responses.

The interface includes responsive behavior for smaller screens.

The complete path:

Web/mobile browser  
→ n8n webhook  
→ FastAPI  
→ Snowball AI  
→ persistent memory and semantic knowledge  
→ response

has been successfully exercised over the local network from an Android device.

The current web client is a local-network development interface rather than a
public authenticated service.

---

## Security State

Snowball's current network interfaces are intended for trusted local
development.

The FastAPI and n8n services should not be directly exposed to the public
internet without appropriate authentication and encrypted transport.

ChromaDB is currently used only as an embedded local vector database through
`chromadb.PersistentClient`.

Snowball does not intentionally expose the ChromaDB HTTP server.

ChromaDB 1.5.9 currently has known published security advisories. Snowball's
embedded/local architecture reduces exposure to several affected server paths,
but this is documented residual dependency risk rather than a claim that the
dependency is vulnerability-free.

Additional details are maintained in:

`docs/SECURITY.md`

GitHub currently reports dependency vulnerabilities on the default branch.
These remain known security debt.

---

## Automated Tests

As of 2026-09-23, the complete automated test suite contains:

**80 passing tests**

The current full-suite result is:

`80 passed, 2 warnings`

The two current warnings are known dependency deprecation warnings involving
Starlette/httpx TestClient behavior and the AnyIO BlockingPortal alias.

They are warnings rather than test failures.

The test suite currently protects behavior including:

- persistent memory
- restart-safe memory retrieval
- episodic memory integration
- semantic memory structures
- knowledge chunking
- embeddings
- document ingestion
- DOCX ingestion
- DOCX table ordering
- vector storage
- semantic retrieval
- RAG context generation
- provenance propagation
- safe knowledge reingestion
- semantic context reaching the live Snowball model prompt
- FastAPI service behavior
- HTTP chat behavior
- operation when Ollama is unavailable
- decision-making behavior
- logging

---

## Verified Current Behaviors

The following behaviors have been directly exercised during current
development:

### Persistent conversational memory

Snowball can store a user fact, shut down, restart, and retrieve the fact in a
later conversation.

### Semantic project knowledge

Snowball can retrieve information from an ingested Snowball project document
using semantic similarity.

### Historical self-knowledge

Snowball can retrieve information about the origins and earlier development of
the Snowball project from its historical archive synthesis.

### Provenance-aware reasoning

Snowball receives provenance metadata and instructions explaining that
historical-reference material should not automatically be treated as current
implementation truth.

In a live test, Snowball correctly distinguished its historical project
documentation from its current implementation and stated that historical plans
may not have been realized.

This behavior represents provenance-aware model reasoning over supplied context.
It is not evidence of consciousness or subjective self-awareness.

### API persistence

Persistent conversational behavior has been exercised through Snowball's HTTP
API.

### n8n gateway

The n8n webhook can communicate with the Snowball FastAPI service and return
Snowball responses.

### Local-network mobile chat

The Snowball web client has successfully communicated with the complete
Snowball stack from an Android browser on the local network.

---

## Known Current Limitations

The current Snowball implementation is intentionally incomplete.

The following long-term capabilities should not currently be assumed to exist
merely because they appear in historical documents or roadmaps:

- general autonomous computer control
- broad file-system awareness
- automatic ingestion of all personal files
- continuous PC hardware monitoring
- mature voice interaction
- continuous vision
- autonomous game-playing environments
- learned cross-game strategy transfer
- full calendar and messaging integration
- unrestricted external service control
- cloud synchronization
- InMoov robotic embodiment
- unrestricted autonomous actions

Some earlier Snowball repositories or archived files may contain prototypes,
partial implementations, experiments, or plans related to these capabilities.
Their historical existence does not establish them as part of the current
working system.

---

## Current Development Principle

Snowball is no longer being developed as one giant creature-script.

The current strategy is:

Keep the existing Python conversational core working.  
Build reliable persistent and semantic memory around it.  
Expose the core through FastAPI.  
Use n8n for commodity integration and automation.  
Provide multiple interfaces to the same Snowball identity.  
Custom-build the components that make Snowball uniquely Snowball.

The goal is to grow outward from a tested core without repeatedly rebuilding
Snowball from scratch.

---

## Current Source of Truth

For questions about current implementation, use evidence in approximately this
order:

1. Current executable code and automated tests
2. This current-state reference
3. Current technical documentation
4. Historical project documentation for historical context

Historical documentation remains valuable, but should not override verified
current implementation behavior when answering questions about what Snowball
can do now.

---

## Maintenance

This is a living document.

It should be updated when Snowball gains, removes, or substantially changes a
verified capability.

Changes to this document should describe implemented reality rather than future
intent.

Future plans belong in roadmap documentation rather than being presented here
as current capabilities.