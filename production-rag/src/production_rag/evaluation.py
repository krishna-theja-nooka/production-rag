import json
import time
from pathlib import Path
from statistics import mean

from .service import RAGService


def evaluate(service: RAGService, dataset: Path) -> dict:
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    results = []
    for case in cases:
        started = time.perf_counter()
        hits = service.retrieve(case["question"], 4)
        response = service.ask(case["question"], 4)
        expected = set(case.get("expected_sources", []))
        retrieved = [hit["source"] for hit in hits]
        recall = len(expected & set(retrieved)) / len(expected) if expected else None
        reciprocal_rank = (
            next((1 / (idx + 1) for idx, source in enumerate(retrieved) if source in expected), 0)
            if expected
            else None
        )
        terms = case.get("answer_contains", [])
        contains = all(term.lower() in response["answer"].lower() for term in terms)
        abstention_ok = response["abstained"] == case.get("should_abstain", False)
        passed = abstention_ok and contains and (recall is None or recall == 1)
        results.append(
            {
                "id": case["id"],
                "source_recall_at_4": recall,
                "reciprocal_rank": reciprocal_rank,
                "answer_terms_match": contains,
                "abstention_correct": abstention_ok,
                "passed": passed,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    positive = [r for r in results if r["source_recall_at_4"] is not None]
    return {
        "generator": service.settings.generator,
        "retrieval": service.settings.retrieval,
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "source_recall_at_4": mean(r["source_recall_at_4"] for r in positive) if positive else 0,
        "mrr_at_4": mean(r["reciprocal_rank"] for r in positive) if positive else 0,
        "note": "Tiny synthetic regression set; not a production quality benchmark. "
        "Latency includes a separate retrieval probe plus an answer call.",
        "results": results,
    }
