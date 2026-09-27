# Build-and-learn path

The complete source is included. Work through it in this order rather than trying to
learn every deployment concept at once. This sequence is independent of the course chapters.

| Step | Read / run | What to explain afterward |
|---|---|---|
| 1 | README offline quickstart | Why an LLM alone cannot know private documents |
| 2 | `documents.py`; ingest the sample folder | Why chunk size, overlap, and page metadata matter |
| 3 | `store.py`; ingest the same file twice, then edit it | How idempotency and atomic replacement prevent stale duplicates |
| 4 | `retrieval.py`; ask exact-word questions | What BM25 retrieves and where paraphrases fail |
| 5 | Enable hybrid mode; compare paraphrases | How embeddings and keyword search complement each other |
| 6 | `generation.py`; enable Transformers | How evidence grounding differs from model knowledge |
| 7 | Inspect cited passages and try unknown questions | Why valid citations are not proof of factual support |
| 8 | `api.py`; exercise `/docs` | Authentication, validation, limits, and failure behavior |
| 9 | `tests/` and `eval/` | Unit/integration tests versus model-quality evaluation |
| 10 | Python serving and production runbook | What remains before real users and private documents |

## Portfolio evidence to collect

- A screenshot of a cited answer and the source passage.
- A short demo showing ingestion, a correct answer, an unknown question, and a document update.
- An architecture explanation in your own words.
- An evaluation comparison: lexical versus hybrid, same held-out questions.
- Measured latency and memory on your own hardware, with sample size and model versions.
- A short account of a failure you found and how you fixed it.

Do not invent uptime, accuracy, cost savings, customers, or production usage. This is a
personal engineering project until you actually deploy and measure it.
