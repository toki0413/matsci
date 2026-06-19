"""Semantic + keyword persona matching for query-based auto-routing.

Reuses the same lazy ChromaDB default embedding model used by the RAG vector
store. If the embedding model is not available locally, falls back to a
keyword-overlap scorer so the matcher works offline.
"""

from __future__ import annotations

from typing import Any

from huginn.personas import Persona, PersonaManager
from huginn.rag.vector_store import _embedding_model_cached


def _keyword_score(query: str, persona: Persona) -> float:
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    score = 0.0
    desc = (persona.description or "").lower()
    if desc:
        if desc in query_lower:
            score += 2.0
        if query_lower in desc:
            score += 1.5
        score += 0.5 * len(query_tokens & set(desc.split()))
    for trigger in persona.when_to_use:
        trigger_lower = trigger.lower()
        if trigger_lower in query_lower or query_lower in trigger_lower:
            score += 1.0
        score += 0.3 * len(query_tokens & set(trigger_lower.split()))
    name_lower = persona.name.lower()
    if name_lower in query_lower:
        score += 1.5
    return score


class PersonaMatcher:
    """Match a free-text query to the most suitable persona."""

    def __init__(self, manager: PersonaManager | None = None):
        self.manager = manager
        self._embedding_fn: Any | None = None
        self._embedding_available: bool | None = None

    def _ensure_embedding(self) -> bool:
        if self._embedding_available is not None:
            return self._embedding_available
        if not _embedding_model_cached():
            self._embedding_available = False
            return False
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._embedding_fn = DefaultEmbeddingFunction()
            _ = self._embedding_fn(["test"])
            self._embedding_available = True
        except Exception:
            self._embedding_available = False
        return self._embedding_available

    def _documents(self, persona: Persona) -> list[str]:
        parts = [persona.name, persona.description or ""]
        parts.extend(persona.when_to_use)
        return [p for p in parts if p.strip()]

    def match(
        self,
        query: str,
        top_k: int = 1,
        score_threshold: float = 0.15,
    ) -> list[tuple[Persona, float]]:
        """Return ranked persona matches for ``query``.

        When embeddings are available we score by cosine similarity to the
        concatenated persona description + triggers. Otherwise we fall back to
        keyword overlap. Only results above ``score_threshold`` are returned.
        """
        manager = self.manager or PersonaManager()
        personas = [manager.get(name) for name in manager.list()]
        if not personas:
            return []

        if not self._ensure_embedding():
            scored = [(_keyword_score(query, p), p) for p in personas]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [(p, s) for s, p in scored if s >= score_threshold][:top_k]

        # Embedding path
        query_embedding = self._embedding_fn([query])[0]
        scored: list[tuple[float, Persona]] = []
        for persona in personas:
            docs = self._documents(persona)
            if not docs:
                continue
            embeddings = self._embedding_fn(docs)
            best = max(self._cosine(query_embedding, e) for e in embeddings)
            # Blend with keyword signal to surface exact name/trigger matches.
            keyword = _keyword_score(query, persona)
            blended = best + 0.1 * min(keyword, 3.0)
            if blended >= score_threshold:
                scored.append((blended, persona))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(p, s) for s, p in scored[:top_k]]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(xa * xa for xa in a) ** 0.5
        norm_b = sum(xb * xb for xb in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def match_persona_for_query(
    query: str,
    manager: PersonaManager | None = None,
    score_threshold: float = 0.25,
) -> str | None:
    """Return the best-matching persona name, or None if no match is strong enough."""
    matcher = PersonaMatcher(manager=manager)
    results = matcher.match(query, top_k=1, score_threshold=score_threshold)
    if not results:
        return None
    return results[0][0].name
