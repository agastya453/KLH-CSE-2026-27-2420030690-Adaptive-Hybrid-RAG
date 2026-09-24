"""Evidence-grounded prompting for local LLM answers."""
from __future__ import annotations

from .ollama_client import generate
from .utils import unique_sources


def build_prompt(query: str, evidence: list[dict], query_type: str) -> str:
    context = "\n\n".join(
        f"EVIDENCE {i} | [Source: {item['file_name']}, Page: {item['page_number']}]\n{item['chunk_text']}"
        for i, item in enumerate(evidence, start=1)
    )
    structure = "Use a compact comparison table or structured bullets." if query_type == "Comparison" else "Use clear, concise prose."
    return f"""You are a careful document intelligence assistant. Answer the question ONLY using the evidence below.
Do not add facts that are not supported by the evidence. If the evidence is insufficient, say exactly: The uploaded documents do not provide enough information.
Every factual statement must include a citation formatted [Source: filename, Page: number]. Clearly distinguish information from different documents. {structure}

QUESTION: {query}

EVIDENCE:
{context}
"""


def generate_answer(query: str, evidence: list[dict], query_type: str, model: str) -> dict:
    if not evidence:
        return {"answer": "The uploaded documents do not provide enough information.", "citations": []}
    answer = generate(build_prompt(query, evidence, query_type), model)
    return {"answer": answer, "citations": unique_sources(evidence)}
