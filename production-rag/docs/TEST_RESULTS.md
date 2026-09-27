# Verification performed during repository creation

Date: 2026-09-27. Environment: Linux, Python 3.12.14.

| Check | Result |
|---|---|
| Automated tests | 35 passed |
| Ruff lint and formatting | Passed |
| CLI sample ingestion | Three synthetic documents indexed |
| Synthetic regression evaluation | 8 of 8 cases passed in lexical/extractive mode |
| Source recall@4 / MRR@4 | 1.0 / 1.0 on the tiny bundled synthetic set only |
| Real Uvicorn HTTP smoke test | Python init-env, ingestion, serve, cited answers, and abstention passed |
| Base/development environment | Existing isolated environment; Python model extras not installed |

The test suite includes actual PDF extraction, SQLite persistence, replacement rollback,
source deletion, backup restore, query validation, authentication, rate limits, and controlled Python model-adapter tests for context
limits, lazy loading, incomplete output, JSON validation, and citation rejection. One upstream Starlette TestClient deprecation warning
appeared about its httpx transport; it did not fail tests.

## Not validated here

- Real Transformers/PyTorch inference and its answer quality: generator weights were not downloaded here.
- Real Hugging Face embeddings: optional model weights were not downloaded. Fusion behavior
  was tested with controlled vectors, not a semantic relevance benchmark.
- Windows/macOS execution, Python 3.11 execution, and actual GitHub-hosted CI.
- End-user generation quality, semantic relevance, and CPU/GPU performance with real weights.
- Public deployment, concurrent load, security audit, multi-user isolation, and operational SLOs.

Do not describe this as benchmarked production infrastructure. Run the documented model,
deployment, and evaluation steps on your target machine before making performance claims.
