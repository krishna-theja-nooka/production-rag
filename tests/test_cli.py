import sys
from pathlib import Path

from production_rag.cli import main


def test_init_env_creates_secret_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".env.example").write_text("RAG_API_KEY=\nRAG_GENERATOR=extractive\n")
    monkeypatch.setattr(sys, "argv", ["rag", "init-env"])
    assert main() == 0
    original = Path(".env").read_text()
    assert len(original.splitlines()[0].split("=", 1)[1]) >= 24
    assert main() == 1
    assert Path(".env").read_text() == original
