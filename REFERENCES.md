# References and attribution

## User-provided course

- Video: https://www.youtube.com/watch?v=qouhIJpp0Is
- Playlist: https://www.youtube.com/playlist?list=PLTHiB6UhbkpgJhgZ0F4u8Jb6U6dLFXM_f
- User-described learning topic: "How to build production level RAG."

The video page/transcript could not be retrieved during repository creation. The creator,
exact title, chapter sequence, implementation, and licensing have not been verified.
The links are recorded as the user's intended learning reference, not evidence that a
specific function was adapted from the course. The implementation is independently written.
If you later adapt course code, inspect its license and add attribution beside the adapted
function as well as here; a link does not replace license compliance.

## Primary technical references

1. FastAPI documentation: https://fastapi.tiangolo.com/
   — API application, request validation, security dependencies, OpenAPI testing interface.
2. Hugging Face Transformers chat templates:
   https://huggingface.co/docs/transformers/chat_templating
   — Python tokenizer chat formatting and local generation.
   Python model-loading example: https://qwen.readthedocs.io/en/v2.5/inference/chat.html
3. Sentence Transformers API:
   https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
   — model loading and normalized embeddings.
4. Embedding model card:
   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   — model limits, intended use, model-specific license (Apache 2.0 at review).
5. Example generator model card:
   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
   — review license, model behavior, and limitations before use.
6. pypdf extraction:
   https://pypdf.readthedocs.io/en/stable/user/extract-text.html
   — text extraction and PDF limitations.
7. Python SQLite documentation:
   https://docs.python.org/3/library/sqlite3.html
   — transactions, parameter binding, and online backup.
8. Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:
   https://arxiv.org/abs/2005.11401
   — foundational RAG architecture, not a claim that this implementation reproduces the paper.
9. Cormack, Clarke, and Buettcher (2009), Reciprocal Rank Fusion outperforms Condorcet
   and individual Rank Learning Methods: https://doi.org/10.1145/1571941.1572114
   — ranking fusion concept used in `retrieval.py`.

Documentation review date: 2026-09-27. Only the API/model documentation needed for the
implementation was checked live; the course content was unavailable. Model and dependency
licenses must be checked again for the exact versions used in your deployment.
