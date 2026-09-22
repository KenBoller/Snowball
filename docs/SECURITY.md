# Snowball AI/OS Security Notes

## ChromaDB

Snowball currently uses ChromaDB as an embedded local vector store through
`chromadb.PersistentClient`.

Snowball does not run or expose the ChromaDB HTTP server.

As of September 22, 2026, ChromaDB 1.5.9 has published security advisories
for vulnerabilities affecting its server, authentication, authorization,
and remote model configuration paths. No patched stable release is
currently available.

Snowball's current architecture does not expose those ChromaDB server
paths to untrusted clients.

Mitigations:

- ChromaDB must remain embedded and local-only.
- Do not start or expose the ChromaDB HTTP server.
- Do not accept untrusted Chroma model or embedding-function configuration.
- Snowball supplies embeddings itself through Ollama.
- Continue auditing dependencies and upgrade ChromaDB when a patched
  stable release becomes available.

This is a documented residual risk, not a claim that the dependency itself
is vulnerability-free.