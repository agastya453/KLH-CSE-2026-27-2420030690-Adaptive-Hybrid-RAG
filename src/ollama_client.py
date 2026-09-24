"""Minimal resilient client for a locally running Ollama service."""
from __future__ import annotations

import requests

BASE_URL = "http://localhost:11434"


def is_ollama_available() -> bool:
    try:
        return requests.get(f"{BASE_URL}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def generate(prompt: str, model: str, timeout: int = 120) -> str:
    response = requests.post(f"{BASE_URL}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}, timeout=timeout)
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer
