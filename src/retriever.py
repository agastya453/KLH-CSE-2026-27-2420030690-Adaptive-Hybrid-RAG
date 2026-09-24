"""BM25, FAISS semantic, adaptive routing, and reciprocal-rank fusion."""
from __future__ import annotations

from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .utils import tokenize


class HybridRetriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["chunk_text"]) for c in chunks])
        vectors = np.ascontiguousarray(embeddings.astype("float32"))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [{**self.chunks[int(i)], "bm25_score": float(scores[i]), "rank": rank + 1}
                for rank, i in enumerate(ids)]

    def semantic_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        scores, ids = self.index.search(np.ascontiguousarray(query_embedding.astype("float32")), min(top_k, len(self.chunks)))
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx >= 0:
                results.append({**self.chunks[int(idx)], "semantic_score": float(score), "rank": rank})
        return results

    @staticmethod
    def classify_query(query: str) -> tuple[str, float, float]:
        lowered = query.lower()
        if any(term in lowered for term in ("compare", "difference", "versus", " vs ", "contrast")):
            return "Comparison", 0.50, 0.50
        if any(term in lowered for term in ("what is", "define", "definition", "meaning of", "explain")):
            return "Definition", 0.25, 0.75
        quoted = '"' in query or any(marker in lowered for marker in ("exact", "keyword", "section", "clause", "term "))
        if quoted or any(char.isdigit() for char in query) or "-" in query:
            return "Exact-term", 0.75, 0.25
        return "General", 0.50, 0.50

    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int) -> dict:
        query_type, bm25_weight, semantic_weight = self.classify_query(query)
        candidate_k = min(max(top_k * 4, 8), len(self.chunks))
        bm25_results = self.bm25_search(query, candidate_k)
        semantic_results = self.semantic_search(query_embedding, candidate_k)
        scores: dict[str, float] = defaultdict(float)
        details: dict[str, dict] = {}
        for weight, results, score_key in ((bm25_weight, bm25_results, "bm25_score"), (semantic_weight, semantic_results, "semantic_score")):
            for result in results:
                key = result["chunk_id"]
                scores[key] += weight / (60 + result["rank"])
                details.setdefault(key, {**result, "bm25_score": None, "semantic_score": None})
                details[key][score_key] = result[score_key]
        ranked = []
        for rank, (key, score) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k], start=1):
            item = details[key]
            item["hybrid_score"], item["rank"] = float(score), rank
            ranked.append(item)
        # Comparison questions get at least one result from another available document when possible.
        if query_type == "Comparison" and len({r["file_name"] for r in ranked}) < 2:
            selected_files = {r["file_name"] for r in ranked}
            for candidate in sorted(details.values(), key=lambda r: r.get("semantic_score") or -999, reverse=True):
                if candidate["file_name"] not in selected_files:
                    candidate["rank"] = len(ranked) + 1
                    ranked.append(candidate)
                    break
        return {"query_type": query_type, "bm25_weight": bm25_weight, "semantic_weight": semantic_weight,
                "bm25_results": bm25_results[:top_k], "semantic_results": semantic_results[:top_k], "hybrid_results": ranked[:top_k]}
