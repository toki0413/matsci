"""Structure motif mining tool — discrete structural pattern discovery.

Inspired by pharmacophore modeling: abstracts continuous structural features
into discrete graph patterns (coordination polyhedra, bond topology, angular
motifs). Uses subgraph isomorphism for pattern matching.

Math:
  Structure graph G = (V, E) where V=atoms, E=bonds with type labels
  Motif M is a subgraph; find all embeddings φ: M → G
  Coordination polyhedra classification: convex hull of nearest neighbors
    → tetrahedron (CN=4), octahedron (CN=6), trigonal prism (CN=6), etc.
  Polyhedral template matching via vertex-degree sequence + edge-coloring
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class MotifMiningInput(BaseModel):
    action: Literal[
        "coordination_polyhedra",
        "bond_motif_search",
        "ring_analysis",
        "graph_match",
        "frequency_analysis",
        "motif_similarity",
    ] = Field(...)

    # Structure input
    positions: list[list[float]] = Field(
        ..., description="Atomic positions (N×3)"
    )
    species: list[str] = Field(
        ..., description="Element symbols for each atom"
    )
    lattice: list[list[float]] | None = Field(
        default=None, description="3×3 lattice vectors (for periodic structures)"
    )

    # Bonding
    cutoff: float = Field(default=3.0, gt=0, description="Bond cutoff distance (Å)")
    bond_pairs: list[list[str]] | None = Field(
        default=None, description="Allowed bond pairs: [[A, B], ...]. None = all."
    )

    # Coordination
    coordination_type: Literal["tetrahedron", "octahedron", "trigonal_prism",
                                "square_planar", "auto"] = Field(
        default="auto", description="Expected coordination geometry"
    )
    tolerance: float = Field(default=0.3, ge=0, le=1, description="Shape tolerance for polyhedra matching")

    # Ring analysis
    max_ring_size: int = Field(default=8, ge=3, le=20)

    # Graph matching
    query_graph: dict | None = Field(
        default=None, description="Query motif: {nodes: [{id, species, coord}], edges: [{src, dst, type}]}"
    )

    # Frequency analysis
    motif_list: list[dict] | None = Field(
        default=None, description="List of motifs to count: [{name, graph: {...}}]"
    )

    # Motif similarity
    reference_motif: dict | None = Field(default=None)
    candidate_motifs: list[dict] | None = Field(default=None)


class MotifMiningTool(HuginnTool):
    """Discrete structural motif mining via graph algorithms."""

    name = "motif_mining_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("tda_tool", "descriptor_tool"),
    )
    description = (
        "Structure motif mining: coordination polyhedra classification, "
        "bond motif subgraph search, ring analysis, and graph isomorphism "
        "matching. Discrete complement to continuous SOAP/MBTR descriptors."
    )
    input_schema = MotifMiningInput

    async def _execute(self, args: MotifMiningInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "coordination_polyhedra":
                return self._coordination_polyhedra(args)
            if args.action == "bond_motif_search":
                return self._bond_motif_search(args)
            if args.action == "ring_analysis":
                return self._ring_analysis(args)
            if args.action == "graph_match":
                return self._graph_match(args)
            if args.action == "frequency_analysis":
                return self._frequency_analysis(args)
            if args.action == "motif_similarity":
                return self._motif_similarity(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── Coordination polyhedra ──────────────────────────────

    def _coordination_polyhedra(self, args: MotifMiningInput) -> ToolResult:
        """Classify local coordination environment of each atom."""
        pos = np.array(args.positions)
        species = args.species
        n = len(species)

        # Compute neighbor list
        neighbors = self._build_neighbor_list(pos, species, args.cutoff, args.lattice)

        polyhedra = []
        for i in range(n):
            nb = neighbors[i]
            cn = len(nb)
            if cn == 0:
                continue

            nb_pos = pos[nb] - pos[i]
            if args.lattice:
                # Minimum image
                inv_lat = np.linalg.inv(np.array(args.lattice))
                frac = nb_pos @ inv_lat
                frac = frac - np.round(frac)
                nb_pos = frac @ np.array(args.lattice)

            # Classify shape
            shape = self._classify_polyhedron(nb_pos, cn, args.tolerance)

            # Compute OPA (order parameter): measure of shape regularity
            opa = self._compute_opa(nb_pos, shape)

            polyhedra.append({
                "atom_index": i,
                "species": species[i],
                "coordination_number": cn,
                "neighbors": nb,
                "neighbor_species": [species[j] for j in nb],
                "polyhedron_type": shape,
                "order_parameter": round(float(opa), 4),
                "regular": opa > 0.7,
            })

        # Statistics
        shape_counts: dict[str, int] = {}
        for p in polyhedra:
            shape_counts[p["polyhedron_type"]] = shape_counts.get(p["polyhedron_type"], 0) + 1

        return ToolResult(data={
            "action": "coordination_polyhedra",
            "n_atoms": n,
            "n_classified": len(polyhedra),
            "polyhedra": polyhedra,
            "shape_distribution": shape_counts,
            "message": f"Classified {len(polyhedra)} coordination environments. "
                       f"Most common: {max(shape_counts, key=shape_counts.get)}.",
        })

    # ── Bond motif search ───────────────────────────────────

    def _bond_motif_search(self, args: MotifMiningInput) -> ToolResult:
        """Search for specific bond motifs (e.g., A-B-A bridges, linear chains)."""
        pos = np.array(args.positions)
        species = args.species
        neighbors = self._build_neighbor_list(pos, species, args.cutoff, args.lattice)

        motifs = []

        # Find A-B-A bridge motifs (central B bonded to two A)
        for i in range(len(species)):
            nb = neighbors[i]
            for j_idx, j in enumerate(nb):
                for k in nb[j_idx + 1:]:
                    # Bond angle at i: j-i-k
                    v1 = pos[j] - pos[i]
                    v2 = pos[k] - pos[i]
                    if args.lattice:
                        v1 = self._minimum_image(v1, args.lattice)
                        v2 = self._minimum_image(v2, args.lattice)
                    cos_angle = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
                    )
                    angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

                    motif_type = self._classify_angle(angle, species[j], species[i], species[k])
                    if motif_type:
                        motifs.append({
                            "type": motif_type,
                            "atoms": [j, i, k],
                            "species": [species[j], species[i], species[k]],
                            "angle_deg": round(angle, 2),
                        })

        motif_types: dict[str, int] = {}
        for m in motifs:
            motif_types[m["type"]] = motif_types.get(m["type"], 0) + 1

        return ToolResult(data={
            "action": "bond_motif_search",
            "n_motifs": len(motifs),
            "motifs": motifs[:200],  # cap
            "motif_distribution": motif_types,
            "message": f"Found {len(motifs)} bond motifs. Types: {motif_types}.",
        })

    # ── Ring analysis ───────────────────────────────────────

    def _ring_analysis(self, args: MotifMiningInput) -> ToolResult:
        """Find all rings up to max_ring_size using graph traversal."""
        pos = np.array(args.positions)
        species = args.species
        neighbors = self._build_neighbor_list(pos, species, args.cutoff, args.lattice)

        # Build adjacency for DFS-based ring search
        adj = {i: set(neighbors[i]) for i in range(len(species))}
        rings = set()  # store as frozenset of sorted atom indices

        def dfs(start: int, current: int, path: list[int], visited: set):
            if len(path) > args.max_ring_size:
                return
            for nb in adj[current]:
                if nb == start and len(path) >= 3:
                    ring = frozenset(path)
                    rings.add(ring)
                elif nb not in visited and nb > start:  # avoid duplicates
                    dfs(start, nb, path + [nb], visited | {nb})

        for i in range(len(species)):
            dfs(i, i, [i], {i})

        # Classify rings
        ring_data = []
        for ring in rings:
            atoms = sorted(ring)
            ring_size = len(atoms)
            ring_species = [species[a] for a in atoms]
            formula = "-".join(ring_species)
            ring_data.append({
                "atoms": atoms,
                "size": ring_size,
                "species": ring_species,
                "formula": formula,
            })

        size_dist: dict[int, int] = {}
        for r in ring_data:
            size_dist[r["size"]] = size_dist.get(r["size"], 0) + 1

        return ToolResult(data={
            "action": "ring_analysis",
            "n_rings": len(ring_data),
            "rings": ring_data[:200],
            "size_distribution": {str(k): v for k, v in sorted(size_dist.items())},
            "message": f"Found {len(ring_data)} rings. Size distribution: {size_dist}.",
        })

    # ── Graph matching ──────────────────────────────────────

    def _graph_match(self, args: MotifMiningInput) -> ToolResult:
        """Find all embeddings of a query graph in the structure graph."""
        if not args.query_graph:
            return ToolResult(data=None, success=False, error="query_graph required")

        pos = np.array(args.positions)
        species = args.species
        neighbors = self._build_neighbor_list(pos, species, args.cutoff, args.lattice)

        query = args.query_graph
        q_nodes = query.get("nodes", [])
        q_edges = query.get("edges", [])

        # Simple VF2-style subgraph isomorphism
        # ponytail: simplified — matches species + connectivity, not edge types
        matches = []

        q_species = [n.get("species") for n in q_nodes]
        q_adj = {n["id"]: set() for n in q_nodes}
        for e in q_edges:
            q_adj[e["src"]].add(e["dst"])
            q_adj[e["dst"]].add(e["src"])

        q_ids = [n["id"] for n in q_nodes]
        n_query = len(q_ids)

        def try_match(q_idx: int, mapping: dict, used: set):
            if q_idx == n_query:
                matches.append(dict(mapping))
                return
            q_id = q_ids[q_idx]
            q_sp = q_species[q_idx]

            for i in range(len(species)):
                if i in used:
                    continue
                if q_sp and species[i] != q_sp:
                    continue
                # Check connectivity with already-mapped nodes
                ok = True
                for q_other in q_adj[q_id]:
                    if q_other in mapping:
                        if i not in neighbors[mapping[q_other]]:
                            ok = False
                            break
                if ok:
                    mapping[q_id] = i
                    used.add(i)
                    try_match(q_idx + 1, mapping, used)
                    del mapping[q_id]
                    used.remove(i)

        try_match(0, {}, set())

        return ToolResult(data={
            "action": "graph_match",
            "n_matches": len(matches),
            "matches": matches[:100],  # cap
            "query_nodes": n_query,
            "query_edges": len(q_edges),
            "message": f"Found {len(matches)} subgraph embeddings.",
        })

    # ── Frequency analysis ──────────────────────────────────

    def _frequency_analysis(self, args: MotifMiningInput) -> ToolResult:
        """Count occurrences of predefined motifs in the structure."""
        if not args.motif_list:
            return ToolResult(data=None, success=False, error="motif_list required")

        pos = np.array(args.positions)
        species = args.species
        neighbors = self._build_neighbor_list(pos, species, args.cutoff, args.lattice)

        # For each motif, count occurrences via graph matching
        results = []
        for motif in args.motif_list:
            query = motif.get("graph", {})
            q_nodes = query.get("nodes", [])
            q_edges = query.get("edges", [])
            q_species = [n.get("species") for n in q_nodes]
            q_adj = {n["id"]: set() for n in q_nodes}
            for e in q_edges:
                q_adj[e["src"]].add(e["dst"])
                q_adj[e["dst"]].add(e["src"])

            # Count matches (simplified)
            count = 0
            for i in range(len(species)):
                if q_species[0] and species[i] != q_species[0]:
                    continue
                if self._has_motif_at(i, q_species, q_adj, neighbors, species, set(), 0, [i]):
                    count += 1

            results.append({
                "name": motif.get("name", "unnamed"),
                "count": count,
                "frequency": round(count / max(len(species), 1), 4),
            })

        return ToolResult(data={
            "action": "frequency_analysis",
            "motif_counts": results,
            "n_atoms": len(species),
            "message": f"Counted {len(results)} motif types. Total occurrences: {sum(r['count'] for r in results)}.",
        })

    # ── Motif similarity ────────────────────────────────────

    def _motif_similarity(self, args: MotifMiningInput) -> ToolResult:
        """Compare candidate motifs to a reference using graph edit distance
        and topological descriptors."""
        if not args.reference_motif or not args.candidate_motifs:
            return ToolResult(data=None, success=False, error="reference_motif and candidate_motifs required")

        ref = args.reference_motif
        ref_graph = ref.get("graph", {})
        ref_desc = self._graph_descriptors(ref_graph)

        results = []
        for i, cand in enumerate(args.candidate_motifs):
            cand_graph = cand.get("graph", {})
            cand_desc = self._graph_descriptors(cand_graph)

            # Simple descriptor-based similarity
            # Use cosine similarity on degree histogram
            sim = self._cosine_sim(ref_desc["degree_histogram"],
                                    cand_desc["degree_histogram"])

            # Graph edit distance estimate (node count difference + edge difference)
            ged = abs(ref_desc["n_nodes"] - cand_desc["n_nodes"]) + \
                  abs(ref_desc["n_edges"] - cand_desc["n_edges"])

            results.append({
                "index": i,
                "name": cand.get("name", f"motif_{i}"),
                "cosine_similarity": round(float(sim), 4),
                "graph_edit_distance": ged,
                "n_nodes": cand_desc["n_nodes"],
                "n_edges": cand_desc["n_edges"],
                "same_species_set": ref_desc["species_set"] == cand_desc["species_set"],
            })

        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        return ToolResult(data={
            "action": "motif_similarity",
            "reference_name": ref.get("name", "reference"),
            "candidates": results,
            "message": f"Compared {len(results)} candidates to reference. "
                       f"Most similar: {results[0]['name']} (sim={results[0]['cosine_similarity']:.3f}).",
        })

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_neighbor_list(pos, species, cutoff, lattice):
        """Build neighbor list with optional PBC."""
        n = len(species)
        neighbors = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[j] - pos[i]
                if lattice is not None:
                    diff = MotifMiningTool._minimum_image(diff, lattice)
                dist = np.linalg.norm(diff)
                if dist < cutoff and dist > 0.1:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors

    @staticmethod
    def _minimum_image(vec, lattice):
        lat = np.array(lattice)
        inv_lat = np.linalg.inv(lat)
        frac = vec @ inv_lat
        frac = frac - np.round(frac)
        return frac @ lat

    @staticmethod
    def _classify_polyhedron(nb_pos, cn, tol):
        """Classify coordination geometry from neighbor positions."""
        if cn == 0:
            return "isolated"
        if cn == 1:
            return "linear_unit"
        if cn == 2:
            # Check angle
            if len(nb_pos) == 2:
                cos_a = np.dot(nb_pos[0], nb_pos[1]) / (
                    np.linalg.norm(nb_pos[0]) * np.linalg.norm(nb_pos[1]) + 1e-10
                )
                angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
                if abs(angle - 180) < 20:
                    return "linear"
                if abs(angle - 90) < 20:
                    return "bent_90"
                return "bent"
            return "bent"
        if cn == 3:
            return "trigonal_planar"
        if cn == 4:
            # Tetrahedron vs square planar
            opa = MotifMiningTool._tetrahedral_opa(nb_pos)
            if opa > 0.5:
                return "tetrahedron"
            return "square_planar"
        if cn == 5:
            return "trigonal_bipyramidal"
        if cn == 6:
            # Octahedron vs trigonal prism
            opa_oct = MotifMiningTool._octahedral_opa(nb_pos)
            if opa_oct > 0.5:
                return "octahedron"
            return "trigonal_prism"
        if cn == 7:
            return "pentagonal_bipyramidal"
        if cn == 8:
            return "cubic"  # or square antiprism
        if cn == 12:
            return "icosahedral"
        return f"cn_{cn}"

    @staticmethod
    def _tetrahedral_opa(pos):
        """Tetrahedral order parameter: average of |cos(θ_ij - 109.47°)|."""
        n = len(pos)
        if n < 4:
            return 0.0
        target = -1.0 / 3.0  # cos(109.47°)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos_ij = np.dot(pos[i], pos[j]) / (
                    np.linalg.norm(pos[i]) * np.linalg.norm(pos[j]) + 1e-10
                )
                total += (cos_ij - target) ** 2
                count += 1
        return 1.0 - math.sqrt(total / count) if count > 0 else 0.0

    @staticmethod
    def _octahedral_opa(pos):
        """Octahedral order parameter: check for 3 pairs of opposite vectors."""
        if len(pos) < 6:
            return 0.0
        # For octahedron, opposite atoms have dot product ~ -|r|²
        n = len(pos)
        matched = 0
        used = set()
        for i in range(n):
            if i in used:
                continue
            best_j = -1
            best_cos = 2.0  # start high, look for most negative
            for j in range(n):
                if j in used or j == i:
                    continue
                cos_ij = np.dot(pos[i], pos[j]) / (
                    np.linalg.norm(pos[i]) * np.linalg.norm(pos[j]) + 1e-10
                )
                if cos_ij < best_cos:
                    best_cos = cos_ij
                    best_j = j
            if best_j >= 0 and best_cos < -0.5:
                matched += 1
                used.add(i)
                used.add(best_j
                        )
        return matched / 3.0  # 3 pairs for octahedron

    @staticmethod
    def _compute_opa(pos, shape):
        if shape == "tetrahedron":
            return MotifMiningTool._tetrahedral_opa(pos)
        if shape == "octahedron":
            return MotifMiningTool._octahedral_opa(pos)
        return 0.5

    @staticmethod
    def _classify_angle(angle, s1, s2, s3):
        if abs(angle - 180) < 15:
            return f"linear_{s1}-{s2}-{s3}"
        if abs(angle - 109.47) < 15:
            return f"tetrahedral_{s1}-{s2}-{s3}"
        if abs(angle - 90) < 15:
            return f"orthogonal_{s1}-{s2}-{s3}"
        if abs(angle - 120) < 15:
            return f"trigonal_{s1}-{s2}-{s3}"
        return None

    @staticmethod
    def _has_motif_at(start, q_species, q_adj, neighbors, species, visited, depth, path):
        if depth == len(q_species):
            return True
        current = path[-1]
        for nb in neighbors[current]:
            if nb in visited:
                continue
            if q_species[depth] and species[nb] != q_species[depth]:
                continue
            if MotifMiningTool._has_motif_at(
                start, q_species, q_adj, neighbors, species,
                visited | {nb}, depth + 1, path + [nb]
            ):
                return True
        return False

    @staticmethod
    def _graph_descriptors(graph):
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)

        adj = {n["id"]: 0 for n in nodes}
        for e in edges:
            adj[e["src"]] = adj.get(e["src"], 0) + 1
            adj[e["dst"]] = adj.get(e["dst"], 0) + 1

        degrees = list(adj.values())
        max_deg = max(degrees) if degrees else 0
        hist = [0] * (max_deg + 1)
        for d in degrees:
            hist[d] += 1

        species_set = set(n.get("species", "") for n in nodes)

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "degree_histogram": hist,
            "species_set": species_set,
        }

    @staticmethod
    def _cosine_sim(a, b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = a + [0] * (max_len - len(a))
            b = b + [0] * (max_len - len(b))
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-10)
