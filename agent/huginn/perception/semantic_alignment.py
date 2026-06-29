"""Semantic Alignment Layer — Layer 3 of Multi-Modal Perception.

Maps raw structured data (filesystem events, terminal output, browser DOM,
simulator logs) into a unified semantic embedding space for cross-modal retrieval
and conflict detection.

No external dependencies. Uses numpy for vector math, pure Python for text
embeddings (simplified bag-of-words + TF-IDF style).

Usage:
    from huginn.perception.semantic_alignment import SemanticAligner
    aligner = SemanticAligner()
    
    # Embed different modalities
    vec_file = aligner.embed("File modified: conservation_matrix.py")
    vec_term = aligner.embed("Error: np.clip detected in line 42")
    vec_browser = aligner.embed("Screenshot shows C-S-H crystal structure")
    
    # Cross-modal similarity
    sim = aligner.similarity(vec_file, vec_term)
    
    # Conflict detection
    conflicts = aligner.detect_conflicts([
        ("code", "conservation_matrix uses np.clip"),
        ("doc", "docstring claims NO band-aids"),
    ])
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SemanticConflict:
    """A detected conflict between two modalities."""
    source_a: str  # modality name: code, doc, visual, terminal, browser
    text_a: str
    source_b: str
    text_b: str
    similarity: float  # how close they are semantically
    contradiction_score: float  # how strongly they contradict
    description: str


class SemanticAligner:
    """Unified semantic embedding space for cross-modal alignment.

    Lightweight: uses bag-of-words + TF-IDF with a small domain-specific
    vocabulary (material science terms). No ML model download required.
    """

    # Material science + programming domain vocabulary
    DOMAIN_VOCAB = frozenset({
        # Material science
        "crystal", "structure", "lattice", "atom", "molecule", "bond", "defect",
        "vacancy", "dislocation", "grain", "boundary", "phase", "transition",
        "diffusion", "conductivity", "elasticity", "plasticity", "fracture",
        "stress", "strain", "energy", "enthalpy", "entropy", "free energy",
        "dft", "md", "fem", "vasp", "lammps", "cp2k", "quantum", "classical",
        "simulation", "optimization", "convergence", "scf", "band", "gap",
        "density", "dos", "phonon", "vibration", "spectrum",
        "c-s-h", "cement", "hydration", "silicate", "calcium", "portlandite",
        # Programming / math
        "code", "function", "class", "method", "variable", "constant",
        "loop", "condition", "recursion", "algorithm", "complexity",
        "bug", "error", "exception", "test", "assert", "verify",
        "proof", "theorem", "lemma", "axiom", "invariant", "conservation",
        "matrix", "vector", "tensor", "eigenvalue", "eigenvector",
        "differential", "equation", "integral", "boundary", "initial",
        "numerical", "analytical", "approximation", "convergence", "divergence",
        # Actions
        "create", "modify", "delete", "read", "write", "execute", "run",
        "build", "compile", "install", "deploy", "test", "pass", "fail",
    })

    CONTRADICTION_PATTERNS = [
        # (pattern_a, pattern_b, description)
        (r"\bclip\b|\btruncate\b|\bhard.?cut\b", r"\bno\s+band.aids\b|\bno\s+truncat\b|\bpreserve\b", "Code uses hard truncation but claims no band-aids"),
        (r"\bconverged\b|\bsuccess\b|\bpassed\b", r"\bnot\s+converged\b|\bfailed\b|\berror\b", "Conflicting convergence status"),
        (r"\bfc\.c\b|\bface.centered\b|\bcc\b|\bcubic\b", r"\bbcc\b|\bbody.centered\b|\bhexagonal\b", "Conflicting crystal structure"),
        (r"\bzero\b|\bnone\b|\bempty\b", r"\bnonzero\b|\bsome\b|\bcontains\b", "Conflicting emptiness assertion"),
    ]

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._vocab = sorted(self.DOMAIN_VOCAB)
        self._vocab_index = {w: i for i, w in enumerate(self._vocab)}
        self._idf = np.ones(len(self._vocab))  # Uniform IDF initially

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, alphanumeric only."""
        return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

    def embed(self, text: str) -> np.ndarray:
        """Embed text into semantic vector space."""
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for token, count in counts.items():
            if token in self._vocab_index:
                idx = self._vocab_index[token]
                vec[idx] = count * self._idf[idx]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        # Project to target dim (simple random projection)
        if self.dim != len(self._vocab):
            projection = self._get_projection_matrix(len(self._vocab), self.dim)
            vec = projection @ vec
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        return vec

    def _get_projection_matrix(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Deterministic random projection matrix (cached)."""
        key = (in_dim, out_dim)
        if not hasattr(self, "_proj_cache"):
            self._proj_cache = {}
        if key not in self._proj_cache:
            rng = np.random.RandomState(42)  # Deterministic
            self._proj_cache[key] = rng.randn(out_dim, in_dim).astype(np.float32) / math.sqrt(in_dim)
        return self._proj_cache[key]

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(vec_a, vec_b))

    def detect_conflicts(self, modalities: list[tuple[str, str]]) -> list[SemanticConflict]:
        """Detect semantic conflicts across modalities.

        Args:
            modalities: List of (modality_name, text) pairs.

        Returns:
            List of detected conflicts with contradiction scores.
        """
        conflicts = []
        n = len(modalities)
        for i in range(n):
            for j in range(i + 1, n):
                name_a, text_a = modalities[i]
                name_b, text_b = modalities[j]
                sim = self.similarity(self.embed(text_a), self.embed(text_b))
                # Check contradiction patterns
                for pat_a, pat_b, desc in self.CONTRADICTION_PATTERNS:
                    match_a = bool(re.search(pat_a, text_a, re.IGNORECASE))
                    match_b = bool(re.search(pat_b, text_b, re.IGNORECASE))
                    match_b_rev = bool(re.search(pat_b, text_a, re.IGNORECASE))
                    match_a_rev = bool(re.search(pat_a, text_b, re.IGNORECASE))
                    if (match_a and match_b) or (match_a_rev and match_b_rev):
                        contradiction = sim * 0.5 + 0.5  # Higher similarity = stronger conflict if contradictory
                        conflicts.append(SemanticConflict(
                            source_a=name_a, text_a=text_a,
                            source_b=name_b, text_b=text_b,
                            similarity=sim,
                            contradiction_score=contradiction,
                            description=desc,
                        ))
        return conflicts

    def cross_modal_retrieve(self, query: str, corpus: list[tuple[str, str]], top_k: int = 3) -> list[tuple[str, str, float]]:
        """Retrieve most similar items from corpus across modalities.

        Returns: [(modality_name, text, similarity), ...]
        """
        query_vec = self.embed(query)
        results = []
        for name, text in corpus:
            vec = self.embed(text)
            sim = self.similarity(query_vec, vec)
            results.append((name, text, sim))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
