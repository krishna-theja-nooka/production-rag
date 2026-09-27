import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .documents import Chunk


class Store:
    """Small-corpus store. Source replacement and its fingerprint commit atomically."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents
                    (source TEXT PRIMARY KEY, fingerprint TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks
                    (id TEXT PRIMARY KEY, source TEXT NOT NULL, page INTEGER NOT NULL,
                     text TEXT NOT NULL, vector TEXT,
                     FOREIGN KEY(source) REFERENCES documents(source) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ensure_config(self, signature: str):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if row and row[0] != signature:
                raise ValueError(
                    "Index configuration changed. Use a new RAG_DB_PATH and re-ingest."
                )
            conn.execute("INSERT OR IGNORE INTO metadata VALUES ('config', ?)", (signature,))

    def fingerprint(self, source: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT fingerprint FROM documents WHERE source=?", (source,)
            ).fetchone()
        return row[0] if row else None

    def replace(
        self,
        source: str,
        fingerprint: str,
        chunks: list[Chunk],
        vectors: list[list[float] | None],
        limit: int,
    ):
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count mismatch")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            remaining = conn.execute(
                "SELECT count(*) FROM chunks WHERE source<>?", (source,)
            ).fetchone()[0]
            if remaining + len(chunks) > limit:
                raise ValueError("Index capacity exceeded; increase limit or use a vector database")
            conn.execute("DELETE FROM documents WHERE source=?", (source,))
            conn.execute("INSERT INTO documents VALUES (?, ?)", (source, fingerprint))
            conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (c.id, c.source, c.page, c.text, json.dumps(v) if v is not None else None)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def rows(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM chunks ORDER BY id")]

    def documents(self):
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute("""
                SELECT d.source, count(c.id) AS chunks FROM documents d
                LEFT JOIN chunks c ON c.source=d.source GROUP BY d.source ORDER BY d.source
            """)
            ]

    def delete(self, source: str) -> bool:
        with self.connect() as conn:
            return conn.execute("DELETE FROM documents WHERE source=?", (source,)).rowcount > 0

    def backup(self, destination: Path):
        if destination.resolve() == self.path.resolve() or destination.exists():
            raise ValueError("Backup destination must be a new file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as src, sqlite3.connect(destination) as dst:
            src.backup(dst)
