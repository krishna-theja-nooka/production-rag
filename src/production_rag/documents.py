import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str


def read_pages(path: Path, settings: Settings) -> list[tuple[int, str]]:
    if path.stat().st_size > settings.max_file_bytes:
        raise ValueError("File exceeds the configured size limit")
    if path.suffix.lower() in {".txt", ".md"}:
        pages = [(1, path.read_text(encoding="utf-8"))]
    elif path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        if len(reader.pages) > 500:
            raise ValueError("PDF exceeds the 500-page limit")
        pages = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            total += len(text)
            if total > settings.max_document_chars:
                raise ValueError("Extracted document exceeds the character limit")
            pages.append((number, text))
    else:
        raise ValueError("Supported formats: .md, .txt, .pdf")
    if sum(len(text) for _, text in pages) > settings.max_document_chars:
        raise ValueError("Document exceeds the character limit")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No extractable text; scanned PDFs require OCR before ingestion")
    return pages


def chunk_pages(
    source: str, pages: list[tuple[int, str]], settings: Settings, tokenizer=None
) -> list[Chunk]:
    chunks = []
    for page, text in pages:
        words = re.findall(r"\S+", text)
        start = 0
        while start < len(words):
            end = min(start + settings.chunk_words, len(words))
            # Respect the actual embedding tokenizer, including special tokens.
            if tokenizer is not None:
                while (
                    end > start + 1
                    and len(tokenizer.encode(" ".join(words[start:end]), add_special_tokens=True))
                    > tokenizer.model_max_length
                ):
                    end -= 1
                if (
                    len(tokenizer.encode(" ".join(words[start:end]), add_special_tokens=True))
                    > tokenizer.model_max_length
                ):
                    raise ValueError("A single token sequence exceeds the embedding context")
            content = " ".join(words[start:end])
            digest = hashlib.sha256(f"{source}\0{page}\0{start}\0{content}".encode()).hexdigest()
            chunks.append(Chunk(digest, source, page, content))
            if end == len(words):
                break
            start = max(start + 1, end - settings.overlap_words)
    return chunks
