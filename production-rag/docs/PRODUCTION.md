# Production gates and runbook

This repository is production-oriented learning code. Passing local tests does not make
it ready for unrestricted public traffic or regulated/private multi-user data.

## Before public deployment

- Put the API behind HTTPS and an authenticated gateway. Use user identities and
  server-enforced document ACLs before introducing multiple users or tenants.
- Never publish a shared service key in GitHub Pages JavaScript. A frontend needs a
  session-authenticated backend/gateway; CORS is not an authorization boundary.
- Keep the model process and its files private. Set secrets through the hosting secret manager and rotate them.
- Enforce body size, connection/time limits, and global/distributed quotas at the gateway.
  Current rate/concurrency counters are per process; use one worker until replaced.
- Parse untrusted documents in isolated, resource-limited workers. Current file/page/text
  limits do not eliminate decompression bombs or pathological PDF parsing.
- Run a representative held-out evaluation, prompt-injection testing, load tests, and
  dependency scans. Validate extraction quality for your actual PDFs.
- Pin dependency hashes, model revisions for a release. The committed
  `requirements.lock` pins the tested base/dev dependency versions but has no hashes.
  Optional embedding and LLM dependencies must be locked separately on your deployment platform.
- Provision disk/RAM/CPU for the chosen model. Establish backup/restore and retention rules.
  Do not store private corpora, SQLite backups, keys, or downloaded models in Git.
- Add request duration histograms, model latency metrics, error alerts, and availability
  checks appropriate to your hosting system. Existing counters reset on restart.

## Health and monitoring

| Signal | Meaning | Suggested initial response |
|---|---|---|
| `/health/live` | HTTP process can respond | Restart/investigate repeated failures |
| `/health/ready` | Authenticated check that the index has documents | Re-ingest or verify volume if 503 |
| `rag doctor` | Checks configuration and model dependencies | Use `--load-model` to explicitly load/download the LLM |
| `/metrics` | Authenticated query, abstention, generation-error counters | Export to monitoring; baseline before alerting |
| HTTP 401 | Missing/incorrect service key | Verify private runtime config |
| HTTP 422 | Invalid request or index/query constraint | Correct input; inspect index configuration |
| HTTP 429 | Single-process quota reached | Wait for the window or adjust measured capacity |
| HTTP 502 | Model unavailable or invalid output | Inspect local memory, model compatibility, and citation behavior |
| HTTP 503 | Busy API or empty index | Retry with backoff; check capacity/ingestion |

Readiness does not execute a model generation; it cannot guarantee an answer will succeed.
Generation runs in the Python process and is serialized per model instance. Its cooperative
time budget does not bound model loading, lock waiting, or a slow decoding step. For hard
deadlines and cancellation, use isolated Python worker processes plus gateway timeouts.
Invalid output is not silently repaired or retried.

## Backup and restore

1. Run `rag backup data/backups/index-001.sqlite3` to use SQLite's online backup API.
2. Store the resulting backup securely outside the host; configure encryption/retention there.
3. To restore, stop the API, choose a new database path, copy the backup there, and point
   `RAG_DB_PATH` to it. Do not overwrite a running SQLite/WAL database.
4. Preserve compatible embedding/chunking settings and model revision.
5. Run `rag list`, a known query, and the regression evaluation before switching traffic.

For an embedding migration, build a separate index, evaluate it, then switch configuration.
Keep the previous index/config for rollback. Never blend vectors from different models.

## Scaling path

Replace full scans with a benchmarked vector and lexical index, add background ingestion
jobs with durable status/retries, enforce ACLs during retrieval, add a model-serving layer,
and move rate limits to a shared store. Add reranking only after measuring its quality/latency
tradeoff. Neither Kubernetes nor an agent framework is required for the first deployment.
