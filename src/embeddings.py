"""Cached CPU embedding-model access."""
from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """Load the compact embedding model once per Streamlit process."""
    return SentenceTransformer(model_name, device="cpu")


def encode_texts(model: SentenceTransformer, texts: list[str]):
    """Create normalized float32 embeddings suitable for IndexFlatIP."""
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
