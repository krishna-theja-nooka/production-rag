import hashlib
import json
from pathlib import Path

from .config import Settings
from .documents import chunk_pages, read_pages
from .generation import TransformersGenerator, generate
from .retrieval import Embedder, search
from .store import Store


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator = (
            TransformersGenerator(settings) if settings.generator == "transformers" else None
        )
        self.store = Store(settings.db_path)
        signature = json.dumps(
            {
                "schema": 1,
                "retrieval": settings.retrieval,
                "model": settings.embedding_model,
                "revision": settings.embedding_revision,
                "chunk_words": settings.chunk_words,
                "overlap_words": settings.overlap_words,
            },
            sort_keys=True,
        )
        self.store.ensure_config(signature)
        self.embedder = Embedder(settings) if settings.retrieval == "hybrid" else None

    def ingest(self, path: Path, source: str | None = None):
        source = source or path.name
        if not source.strip() or len(source) > 200:
            raise ValueError("Source must contain 1–200 characters")
        # Check size before reading bytes; parsing is an operator-only operation.
        if path.stat().st_size > self.settings.max_file_bytes:
            raise ValueError("File exceeds the configured size limit")
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.store.fingerprint(source) == fingerprint:
            return {"source": source, "status": "unchanged"}
        pages = read_pages(path, self.settings)
        chunks = chunk_pages(
            source, pages, self.settings, self.embedder.tokenizer if self.embedder else None
        )
        if len(chunks) > self.settings.max_chunks:
            raise ValueError("Document exceeds the index chunk limit")
        vectors = (
            self.embedder.encode([chunk.text for chunk in chunks])
            if self.embedder
            else [None] * len(chunks)
        )
        self.store.replace(source, fingerprint, chunks, vectors, self.settings.max_chunks)
        return {"source": source, "status": "indexed", "chunks": len(chunks)}

    def retrieve(self, question: str, top_k: int = 4):
        return search(self.store.rows(), question, top_k, self.settings, self.embedder)

    def ask(self, question: str, top_k: int = 4):
        hits = self.retrieve(question, top_k)
        remaining = self.settings.context_chars
        evidence = []
        for hit in hits:
            if remaining <= 0:
                break
            text = hit["text"][:remaining]
            remaining -= len(text)
            evidence.append(
                {
                    "citation": len(evidence) + 1,
                    "chunk_id": hit["id"],
                    "source": hit["source"],
                    "page": hit["page"],
                    "text": text,
                }
            )
        answer = generate(question, evidence, self.settings, self.generator)
        return {
            "answer": answer.answer,
            "abstained": answer.abstained,
            "mode": self.settings.generator,
            "retrieval": self.settings.retrieval,
            "citations": [item for item in evidence if item["citation"] in answer.citations],
        }
