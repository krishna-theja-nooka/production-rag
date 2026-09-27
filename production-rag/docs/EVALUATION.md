# Evaluation plan

The bundled eight-case synthetic set is deliberately small and deterministic. Run it
with the included sample documents. Each answerable case has expected source IDs and
literal answer terms; the unanswerable case is lexically unrelated.

The runner computes source recall@4 and reciprocal rank@4 for answerable cases, checks
answer terms, and checks expected abstention. A failed case exits with a nonzero status.
Its latency includes both a retrieval probe and an answer call; it is not HTTP p95 latency.
No LLM judge is used and there is no paid evaluation API.

## Expand before claiming answer quality

Create a held-out dataset with at least 50–100 human-reviewed questions from your actual
domain, including synonyms, ambiguous questions, conflicting/obsolete documents, missing
answers, exact identifiers, and malicious instructions embedded in source text. Include
unanswerable questions that share vocabulary with real documents; lexical abstention alone
will not handle these reliably.

Separate development and held-out test sets. Do not tune thresholds on the final test set.
Record corpus version, model revision/digest, prompts, chunk settings, environment, and date.

Review each generated answer for:

1. Correctness against a human reference.
2. Whether each claim is supported by the cited passage (faithfulness).
3. Whether all factual claims have citations (citation coverage).
4. Whether unsupported questions are refused (abstention precision/recall).
5. Whether expected source chunks were retrieved (chunk-level relevance, not only source ID).

Proposed initial release targets, not measured results: source recall@4 >= 0.90,
human-reviewed supported claims >= 0.95, no cross-user disclosure, and a hardware-specific
latency target that you set before measuring. Revisit targets with the intended users.
Re-evaluate whenever documents, embeddings, chunking, prompts, or the generator change.
