# Snowball AI/OS - n8n Integration

n8n acts as the external automation and integration layer for Snowball AI/OS.

Snowball's Python/FastAPI core remains responsible for AI behavior, memory,
reasoning, and identity. n8n connects external services and interfaces to that
core.

## Architecture

External Client / Service
        |
        v
       n8n
        |
        v
Snowball FastAPI
        |
        v
Snowball AI Core
   |          |
   v          v
 Memory     Ollama

n8n is the nervous system.

Snowball remains the brain.

## Local n8n Installation

n8n currently runs in Docker Desktop using the official n8n image.

Container:

    snowball-n8n

Persistent Docker volume:

    snowball_n8n_data

Local editor:

    http://localhost:5678

The n8n editor is intentionally bound to localhost.

## Snowball API

Snowball's FastAPI server currently runs on the Windows host:

    http://localhost:8000

Start it from the Snowball project root:

    .\.venv\Scripts\Activate.ps1

    python -m uvicorn core.api.server:app --host 0.0.0.0 --port 8000

Because n8n runs inside Docker, workflows reach the Windows-hosted Snowball API
through:

    http://host.docker.internal:8000

## Workflows

Version-controlled workflow exports are stored in:

    integrations/n8n/workflows/

### Snowball Chat Gateway

File:

    workflows/snowball-chat-gateway.json

Purpose:

    External caller
        |
        v
    n8n Webhook
        |
        v
    POST /chat
        |
        v
    Snowball AI
        |
        v
    n8n response
        |
        v
    External caller

Production webhook:

    POST http://localhost:5678/webhook/snowball-chat

Expected request body:

    {
      "message": "What is Kraken?"
    }

Example response:

    {
      "response": "Kraken is your Neptune 4 printer, as we discussed earlier."
    }

## Restoring a Workflow

Workflow JSON files in this directory are exports from n8n.

To restore one:

1. Start the `snowball-n8n` Docker container.
2. Open the n8n editor.
3. Import the desired JSON workflow.
4. Review its configuration.
5. Publish the workflow.
6. Test its production webhook.

The Docker volume contains the live n8n configuration and data.

The exported JSON files in this repository provide version-controlled,
reproducible workflow definitions.

## Security

Snowball's n8n editor and webhook are currently intended for local development.

Do not expose ports 5678 or 8000 directly to the public internet.

Remote access should be added through an authenticated and encrypted gateway
rather than direct port forwarding.

## Current Status

Working:

- Dockerized n8n instance
- Persistent n8n Docker volume
- n8n -> Snowball FastAPI communication
- Production chat webhook
- Persistent Snowball memory through the gateway
- Version-controlled workflow export

Next milestone:

Snowball Web Client v0.1 and local-network mobile access.