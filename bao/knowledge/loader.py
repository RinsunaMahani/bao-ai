"""Bao AI - Document & Dataset Loading.

Everything that turns raw bytes (a CSV on disk, an uploaded .txt/.pdf/.csv,
a pasted document) into clean text or structured rows lives here. This
merges three things that were previously scattered: the curated-facts CSV
loader from the old knowledge.py, the chunking logic from
ingestion_service.py, and the PDF/txt extraction that used to live inline
inside nompilo_web.py's Streamlit callback.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

from bao.core.exceptions import RetrievalError
from bao.core.logging import get_logger

logger = get_logger(__name__)


def has_pandas_support() -> bool:
    return _HAS_PANDAS


def has_pdf_support() -> bool:
    return _HAS_PYPDF


@dataclass
class Chunk:
    id: str
    title: str
    content: str


def load_knowledge_csv(path: str):
    """Loads the curated Question/Answer knowledge base CSV.

    Raises RetrievalError (rather than returning None and logging) so the
    caller decides how to degrade — e.g. the orchestrator can fall back to
    Gemini-only mode and tell the user retrieval is unavailable, instead of
    silently returning no facts forever.
    """
    if not _HAS_PANDAS:
        raise RetrievalError("pandas is required to load the knowledge base CSV.")

    import os
    if not os.path.exists(path):
        raise RetrievalError(f"Knowledge base file not found at {path}")

    df = pd.read_csv(path)
    if not {"Question", "Answer"}.issubset(df.columns):
        raise RetrievalError("Dataset missing required 'Question' or 'Answer' columns.")

    # Alternative phrasings are folded into the retrieval corpus but NOT
    # into the displayed question. TF-IDF can only match vocabulary that
    # is present, so a row phrased one way is unreachable by a user who
    # phrases it another way — this is the single cheapest lever on
    # paraphrase recall, and it needs no model change.
    #
    # Do NOT author these against eval/rag_eval.csv. "Other ways someone
    # might ask this" is precisely what the paraphrase eval queries are,
    # so variants written while looking at that file will score against
    # themselves. Validate on eval/rag_holdout.csv instead.
    if "Alt_Questions" not in df.columns:
        df["Alt_Questions"] = ""
    df["Alt_Questions"] = df["Alt_Questions"].fillna("").astype(str)

    df["search_corpus"] = (
        df["Question"].fillna("") + " " + df["Answer"].fillna("")
        + " " + df["Alt_Questions"]
    ).str.strip()

    # Provenance. "verified" means an authority and a URL are attached —
    # not merely that a human typed the row in. A curated row with no
    # source is honest as `curated`; calling it verified is the claim the
    # UI cannot support.
    for column, default in (
        ("Canonical_Id", ""), ("Source", ""), ("Source_URL", ""),
        ("Last_Verified", ""), ("Verified", "curated"),
    ):
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default).astype(str).str.strip()

    # Optional. Marks the language a curated answer is already written in,
    # so an answer that is already in the user's language is served as-is
    # instead of being round-tripped through a translator. Absent in older
    # knowledge bases and in test fixtures, hence the default rather than a
    # required column.
    if "Language" not in df.columns:
        df["Language"] = "English"
    df["Language"] = df["Language"].fillna("English").astype(str).str.strip()
    logger.info(f"Loaded knowledge base with {len(df)} records from {path}.")
    return df


def chunk_text(text: str, source_name: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[Chunk]:
    """Splits text into overlapping word-window chunks for ad-hoc document
    ingestion (uploaded files), as distinct from the curated CSV above.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [Chunk(id=f"{source_name}_chunk_1", title=f"{source_name} (Chunk 1)", content=text.strip())]

    chunks: list[Chunk] = []
    start = 0
    idx = 1
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(Chunk(
            id=f"{source_name}_chunk_{idx}",
            title=f"{source_name} (Chunk {idx})",
            content=" ".join(words[start:end]),
        ))
        idx += 1
        start += (chunk_size - chunk_overlap)
    return chunks


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extracts plain text from an uploaded file's raw bytes, based on
    extension. Supports .txt, .md, .csv, and .pdf (if pypdf is installed).
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext in {"txt", "md"}:
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "csv":
        decoded = file_bytes.decode("utf-8", errors="ignore")
        rows = [", ".join(row) for row in csv.reader(io.StringIO(decoded)) if row]
        return "\n".join(rows)

    if ext == "pdf":
        if not _HAS_PYPDF:
            logger.warning("pypdf not installed; cannot extract text from PDF uploads.")
            return ""
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    logger.warning(f"Unsupported file format for ingestion: .{ext}")
    return ""
