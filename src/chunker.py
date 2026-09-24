"""Reliable word-window chunking that retains source metadata."""
from __future__ import annotations


def chunk_pages(pages: list[dict], chunk_size: int = 180, overlap: int = 35) -> list[dict]:
    if chunk_size < 20:
        raise ValueError("Chunk size must be at least 20 words.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    chunks, sequence = [], 1
    step = chunk_size - overlap
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + chunk_size]).strip()
            if not piece:
                continue
            chunks.append({
                "file_name": page["file_name"], "page_number": page["page_number"],
                "chunk_id": f"C{sequence:04d}", "chunk_text": piece,
            })
            sequence += 1
            if start + chunk_size >= len(words):
                break
    return chunks
