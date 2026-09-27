"""Independent BM25/RRF implementation; see REFERENCES.md for the RRF paper
and Sentence Transformers API. No course source code is copied here.
"""

import json
import math
import re
from collections import Counter
from threading import Lock

from .config import Settings

STOPWORDS = set(
    "a an the is are was were be to of in on for and or how what when where "
    "why do does can i we you it this that with me my please tell about".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class Embedder:
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_revision,
            trust_remote_code=False,
            device="cpu",
        )
        self.tokenizer = self.model.tokenizer
        self.tokenizer.model_max_length = self.model.max_seq_length
        self.lock = Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self.lock:
            return self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()


def search(
    rows: list[dict], question: str, k: int, settings: Settings, embedder: Embedder | None = None
) -> list[dict]:
    query = set(tokens(question))
    if not rows or not query:
        return []
    counters = [Counter(tokens(row["text"])) for row in rows]
    lengths = [sum(c.values()) for c in counters]
    average = max(sum(lengths) / len(rows), 1)
    df = {word: sum(word in c for c in counters) for word in query}
    lexical = []
    for idx, counts in enumerate(counters):
        score = 0.0
        for word in query:
            freq = counts[word]
            if freq:
                idf = math.log(1 + (len(rows) - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * lengths[idx] / average))
        if score > 0:
            lexical.append((idx, score))
    lexical.sort(key=lambda item: (-item[1], rows[item[0]]["id"]))
    ranked_lists = [lexical[:20]]
    dense_scores = {}
    if embedder:
        # Avoid silently truncating a long query differently from the indexed text.
        if len(embedder.tokenizer.encode(question)) > embedder.tokenizer.model_max_length:
            raise ValueError("Question exceeds the embedding model token limit; shorten it")
        vector = embedder.encode([question])[0]
        dense = []
        for idx, row in enumerate(rows):
            stored = json.loads(row["vector"])
            if len(stored) != len(vector):
                raise ValueError("Embedding dimension mismatch; rebuild the index")
            score = sum(a * b for a, b in zip(vector, stored, strict=True))
            dense_scores[idx] = score
            if score >= settings.min_dense_score:
                dense.append((idx, score))
        dense.sort(key=lambda item: (-item[1], rows[item[0]]["id"]))
        ranked_lists.append(dense[:20])
    # Reciprocal-rank fusion: heterogeneous scores are not treated as probabilities.
    fused = Counter()
    for ranking in ranked_lists:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] += 1 / (60 + rank)
    lexical_scores = dict(lexical)
    chosen = sorted(fused, key=lambda idx: (-fused[idx], rows[idx]["id"]))[:k]
    return [
        {
            **{key: value for key, value in rows[idx].items() if key != "vector"},
            "score": fused[idx],
            "lexical_score": lexical_scores.get(idx, 0),
            "dense_score": dense_scores.get(idx),
        }
        for idx in chosen
    ]
