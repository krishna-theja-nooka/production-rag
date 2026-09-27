import json
from pathlib import Path

import pytest

from production_rag.documents import chunk_pages, read_pages
from production_rag.evaluation import evaluate
from production_rag.retrieval import search
from production_rag.service import RAGService


def test_ingestion_query_persistence_and_abstention(service, cfg):
    answer = service.ask("How many annual leave days?")
    assert "20" in answer["answer"]
    assert answer["citations"][0]["source"] == "policy.md"
    assert not answer["abstained"]
    assert RAGService(cfg).ask("annual leave")["citations"] == answer["citations"]
    assert service.ask("quantum chromodynamics")["abstained"]


def test_idempotency_replacement_and_deletion(service, tmp_path):
    source = tmp_path / "policy.md"
    ids = [row["id"] for row in service.store.rows()]
    assert service.ingest(source)["status"] == "unchanged"
    assert [row["id"] for row in service.store.rows()] == ids
    source.write_text("Employees receive 25 days of annual leave.")
    assert service.ingest(source)["status"] == "indexed"
    assert "25" in service.ask("annual leave")["answer"]
    assert "20" not in service.ask("annual leave")["answer"]
    assert service.store.delete("policy.md")
    assert service.store.rows() == []


def test_failed_replacement_keeps_original(service, cfg, tmp_path):
    cfg.max_chunks = 1
    source = tmp_path / "policy.md"
    original = service.store.fingerprint("policy.md")
    source.write_text("annual leave " * 500)
    with pytest.raises(ValueError):
        service.ingest(source)
    assert service.store.fingerprint("policy.md") == original
    assert "20" in service.ask("annual leave")["answer"]


def test_configuration_change_requires_new_index(service, cfg):
    with pytest.raises(ValueError, match="configuration changed"):
        RAGService(cfg.model_copy(update={"chunk_words": 100}))


def test_page_citations_and_chunk_coverage(cfg):
    cfg.chunk_words = 20
    cfg.overlap_words = 4
    words = [f"word{x}" for x in range(75)]
    chunks = chunk_pages("test.pdf", [(2, " ".join(words)), (3, "last page")], cfg)
    recovered = {w for chunk in chunks for w in chunk.text.split()}
    assert set(words) <= recovered
    assert chunks[0].page == 2 and chunks[-1].page == 3
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks)


def test_tokenizer_aware_chunking(cfg):
    class Tokenizer:
        model_max_length = 10

        def encode(self, text, **kwargs):
            return [0] * (len(text.split()) + 2)

    chunks = chunk_pages("x", [(1, " ".join(f"word{x}" for x in range(40)))], cfg, Tokenizer())
    assert all(len(chunk.text.split()) <= 8 for chunk in chunks)
    assert "word39" in chunks[-1].text


def test_invalid_documents_fail_without_mutation(service, tmp_path, cfg):
    path = tmp_path / "empty.txt"
    path.write_text("  ")
    with pytest.raises(ValueError, match="No extractable text"):
        service.ingest(path)
    cfg.max_file_bytes = 1
    with pytest.raises(ValueError, match="size limit"):
        service.ingest(tmp_path / "policy.md")
    assert len(service.store.documents()) == 1


def test_pdf_text_and_page_numbers(tmp_path, cfg):
    # A minimal valid PDF fixture with an actual text stream, not a mocked parser.
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 250 Td (Annual leave is 20 days.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "policy.pdf"
    with path.open("wb") as out:
        writer.write(out)
    pages = read_pages(path, cfg)
    assert pages[0][0] == 1
    assert "Annual leave is 20 days." in pages[0][1]


def test_hybrid_rrf_with_controlled_vectors(cfg):
    class Tokenizer:
        model_max_length = 100

        def encode(self, text):
            return text.split()

    class FakeEmbedder:
        tokenizer = Tokenizer()

        def encode(self, texts):
            return [[1.0, 0.0] for text in texts]

    rows = [
        {
            "id": "a",
            "source": "a",
            "page": 1,
            "text": "annual holiday",
            "vector": json.dumps([1.0, 0.0]),
        },
        {
            "id": "b",
            "source": "b",
            "page": 1,
            "text": "parking permit",
            "vector": json.dumps([0.0, 1.0]),
        },
    ]
    hits = search(rows, "vacation", 4, cfg, FakeEmbedder())
    assert [hit["id"] for hit in hits] == ["a"]
    assert hits[0]["dense_score"] == 1
    assert hits[0]["lexical_score"] == 0


def test_backup_restores_queryable_index(service, cfg, tmp_path):
    backup = tmp_path / "backup.sqlite3"
    service.store.backup(backup)
    restored = RAGService(cfg.model_copy(update={"db_path": backup}))
    assert "20" in restored.ask("annual leave")["answer"]
    with pytest.raises(ValueError):
        service.store.backup(backup)


def test_sample_regression_dataset(cfg):
    service = RAGService(cfg)
    for path in sorted(Path("data/sample").glob("*.md")):
        service.ingest(path)
    report = evaluate(service, Path("eval/questions.jsonl"))
    assert report["passed"] == report["cases"] == 8


def test_same_filename_distinct_source_ids(cfg, tmp_path):
    service = RAGService(cfg)
    path = tmp_path / "policy.md"
    path.write_text("annual leave policy")
    service.ingest(path, "hr/policy.md")
    service.ingest(path, "legal/policy.md")
    assert len(service.store.documents()) == 2
