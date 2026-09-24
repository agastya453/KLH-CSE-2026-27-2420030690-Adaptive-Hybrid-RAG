"""Small, dependency-free utilities shared by the application."""
from __future__ import annotations

import re
from typing import Iterable


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that deliberately needs no NLTK data."""
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())


def highlight_terms(text: str, query: str) -> str:
    """Safely highlight query tokens for Streamlit Markdown rendering."""
    terms = sorted(set(tokenize(query)), key=len, reverse=True)
    if not terms:
        return text
    pattern = r"(?i)\\b(" + "|".join(re.escape(term) for term in terms) + r")\\b"
    return re.sub(pattern, r"**\\1**", text)


def unique_sources(results: Iterable[dict]) -> list[str]:
    """Return citation strings in first-seen order."""
    seen, citations = set(), []
    for item in results:
        citation = f"[Source: {item['file_name']}, Page: {item['page_number']}]"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
