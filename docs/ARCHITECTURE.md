# Architecture and decisions

## Scope

One trusted operator ingests one corpus. One service API key grants access to that whole
corpus. There is no multi-tenant isolation, end-user login, or document-level ACL. Never use
this key model to mix customers' private documents. Ingestion is CLI-only to avoid exposing
arbitrary file reads, URL fetching, or expensive PDF parsing through the public query API.

## Ingestion

1. Validate format and input size, then hash document bytes.
2. Skip unchanged sources based on a SHA-256 fingerprint.
3. Extract page text. PDF page numbers are one-based; TXT/Markdown uses page 1.
4. Chunk with word overlap and, in hybrid mode, the embedding tokenizer's actual token cap.
5. Encode chunks with normalized local embeddings when enabled.
6. Commit document metadata and chunks in one SQLite transaction. Failed parsing/embedding
   or capacity checks leave the old version intact. Directory batches are not one transaction.

Index configuration is persisted to prevent combining incompatible embedding/chunking
settings. This is a configuration signature, not a general database migration system.
The operator should run a single ingestion job at a time; concurrent writers for the same
source follow last-commit-wins semantics.

## Retrieval and generation

The request is authenticated before query execution. A global, process-local sliding
window rate limiter and bounded concurrency protect a single-worker demo. BM25 ranks
lexical matches. Hybrid mode also scans normalized vectors and computes dot products
(cosine similarities), then combines the top 20 of each list with RRF, using k=60.
No cross-encoder reranker is implemented in this version.

The top evidence chunks are capped by a character budget and assigned citation numbers.
A system prompt separates instructions from untrusted evidence, and the evidence is JSON
serialized. This reduces accidental instruction mixing but does not defeat every prompt
injection. No tools, arbitrary code execution, URLs, or write actions are available to the LLM.

No retrieved evidence means abstention without a model call. A nonempty retrieval set is
not proof that a question is answerable. The Python LLM is instructed to abstain when evidence is
insufficient; extractive mode only returns excerpts and cannot make this semantic decision.
The dense threshold is configurable and requires domain calibration. RRF scores are ranks,
not confidence probabilities.

Generated JSON is schema validated. Claimed citation IDs must exist and match inline
bracket references. These checks detect fabricated reference IDs, not whether every factual
claim is supported by its cited passage. Semantic faithfulness is an evaluation/deployment
gate. Model load failure, truncation, malformed JSON, and invalid citations return explicit errors.

## Why these tools

| Choice | Reason | Tradeoff |
|---|---|---|
| FastAPI/Pydantic | Clear API contracts and interactive API docs | Needs a separate server from GitHub Pages |
| SQLite | No external database account; atomic replacement and easy backups | Single-machine store, no distributed vector index |
| Handwritten BM25/RRF | Makes ranking inspectable for learning | English-oriented tokenization; needs multilingual adaptation |
| Sentence Transformers | Local Hugging Face embeddings | Model download and CPU/RAM overhead |
| Transformers + PyTorch | In-process Python inference from Hugging Face weights | Memory, hardware, and model-specific quality |
| Direct Python orchestration | Few moving parts; every pipeline stage is visible | No LangChain/LangGraph features are implied |
| Pytest + synthetic evaluation | Repeatable regression and failure tests | Does not substitute for real-user evaluation |

Dense search is a brute-force Python scan; BM25 is recomputed per query. The 10,000-chunk
cap is a safety bound, not a performance claim. For larger corpora, replace this layer with
PostgreSQL/pgvector, Qdrant, or another benchmarked index, and an incremental lexical index.

## Deliberate exclusions

No conversational memory, streaming, OCR, image/table reasoning, SQL execution,
automatic web crawling, reranking, background ingestion queue, Redis, Kubernetes,
SSO, or high availability is claimed. Add each only when your use case requires it.
The API answers each question independently; follow-up pronouns have no prior context.
