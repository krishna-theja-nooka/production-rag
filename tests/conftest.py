import pytest

from production_rag.config import Settings
from production_rag.service import RAGService


@pytest.fixture
def cfg(tmp_path):
    return Settings(
        _env_file=None,
        db_path=tmp_path / "index.sqlite3",
        api_key="test-only-key-with-at-least-24-characters",
    )


@pytest.fixture
def service(cfg, tmp_path):
    rag = RAGService(cfg)
    source = tmp_path / "policy.md"
    source.write_text("Employees receive 20 days of annual leave. Leave requires manager approval.")
    rag.ingest(source)
    return rag
