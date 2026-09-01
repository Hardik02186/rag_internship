from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import Document


@dataclass
class ChunkConfig:
    chunk_size: int = 600           # target characters per chunk
    chunk_overlap: int = 200         # characters of overlap between chunks
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",                       # paragraph breaks
        "\n",                         # line breaks
        " ",                          # spaces
        "",                           # character level
    ])
    min_chunk_size: int = 100        # discard chunks shorter than this


def _clean(text: str) -> str:
    """Normalise whitespace, remove junk characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)  # strip non-ASCII control chars
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _chunk_text(
    text: str,
    cfg: ChunkConfig,
    source: str = "",
    page_start: int = 0,
) -> list[Document]:
    if not text.strip():
        return []

    # Initialize LangChain text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        length_function=len,
    )

    # Split text into chunks
    split_texts = splitter.split_text(text)

    # Convert to Document objects with metadata
    chunks: list[Document] = []
    for chunk_idx, chunk_text in enumerate(split_texts):
        # Filter out very small chunks
        if len(chunk_text.strip()) >= cfg.min_chunk_size:
            doc = Document(
                id=str(uuid.uuid4()),
                text=chunk_text,
                source=source,
                chunk_index=chunk_idx,
                metadata={
                    "source": source,
                    "chunk_index": chunk_idx,
                    "page_start": page_start,
                    "char_count": len(chunk_text),
                },
            )
            chunks.append(doc)

    return chunks


class PDFLoader:
   

    def __init__(self, cfg: ChunkConfig | None = None) -> None:
        self.cfg = cfg or ChunkConfig()

    def load_file(self, path: str | Path) -> list[Document]:
        """Load a single PDF and return chunks."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(str(path))
        source = path.name
        total_pages = len(reader.pages)
        print(f"  Loading PDF: {source} ({total_pages} pages)")

        # Extract per-page text
        pages_text: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            cleaned = _clean(raw)
            if cleaned:
                pages_text.append((i + 1, cleaned))

        if not pages_text:
            print(f"  ⚠ No extractable text in {source}")
            return []

        full_text = " ".join(text for _, text in pages_text)
        chunks = _chunk_text(full_text, self.cfg, source=source)

        char_count = len(full_text)
        print(f"{source}: extracted {char_count} chars → {len(chunks)} chunks ")
        return chunks
