"""Topological data analysis tool — persistent homology for materials science.

Computes persistence diagrams, persistence images, landscapes, bottleneck
distances, and topological descriptors of energy landscapes and crystal
structures. Tries ripser first, then gudhi, then falls back to a scipy-based
implementation so the tool degrades gracefully when neither is installed.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult


class TDAToolInput(BaseModel):
    action: Literal[
        "persistence_diagram",
        "persistence_image",
        "bottleneck_distance",
        "landscape",
        "energy_landscape_topology",
        "structure_topology",
    ] = Field(..., description="TDA action to perform.")
    point_cloud: list[list[float]] | None = Field(
        default=None,
        description="N points x D dimensions. With distance_matrix=True this "
        "is a precomputed N x N distance matrix.",
    )
    distance_matrix: bool = Field(
        default=False,
        description="Treat point_cloud as a precomputed square distance matrix.",
    )
    max_dim: int = Field(default=2, description="Maximum homology dimension.")
    diagram: list[dict] | None = Field(
        default=None,
        description="Precomputed persistence diagram (list of "
        "{dim, birth, death, persistence}) for image / landscape / bottleneck.",
    )
    diagram2: list[dict] | None = Field(
        default=None,
        description="Second diagram, used only by bottleneck_distance.",
    )
    resolution: int = Field(
        default=20, description="Grid resolution for persistence_image / landscape."
    )
    sigma: float = Field(
        default=0.1, description="Gaussian smoothing width for persistence_image."
    )
    k: int = Field(default=3, description="Number of landscape functions.")
    energies: list[float] | None = Field(
        default=None,
        description="Per-structure energies for energy_landscape_topology.",
    )
    structures: list[list[float]] | None = Field(
        default=None,
        description="Per-structure coordinates for energy_landscape_topology.",
    )
    threshold: float | None = Field(
        default=None,
        description="Energy threshold for energy_landscape_topology edges. "
        "Auto-derived from the data when None.",
    )
    structure: dict | None = Field(
        default=None,
        description="Structure dict ({lattice, sites}) for structure_topology.",
    )
    radii: list[float] | None = Field(
        default=None,
        description="Neighbour cutoff radii for structure_topology. "
        "Auto-derived from the lattice when None.",
    )


class TDAToolOutput(BaseModel):
    """Loose output envelope — the real payload lives in ToolResult.data."""

    action: str
    data: dict[str, Any] | None = None


class TDATool(HuginnTool[TDAToolInput, TDAToolOutput]):
    """Persistent homology and topological descriptors for materials science."""

    name = "tda"
    category = "sci"
    description = (
        "Topological data analysis: persistent homology, persistence diagrams, "
        "Betti numbers, and energy landscape topology for materials science."
    )
    input_schema = TDAToolInput
    output_schema = TDAToolOutput

    read_only = True
    destructive = False

    def is_read_only(self, args: TDAToolInput) -> bool:
        return True

    async def validate_input(
        self, args: TDAToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "persistence_diagram" and not args.point_cloud:
            return ValidationResult(
                result=False, message="persistence_diagram requires point_cloud."
            )
        if args.action == "persistence_image" and not args.diagram:
            return ValidationResult(
                result=False, message="persistence_image requires diagram."
            )
        if args.action == "landscape" and not args.diagram:
            return ValidationResult(
                result=False, message="landscape requires diagram."
            )
        if args.action == "bottleneck_distance" and (
            not args.diagram or not args.diagram2
        ):
            return ValidationResult(
                result=False,
                message="bottleneck_distance requires diagram and diagram2.",
            )
        if args.action == "energy_landscape_topology" and (
            not args.energies or not args.structures
        ):
            return ValidationResult(
                result=False,
                message="energy_landscape_topology requires energies and structures.",
            )
        if args.action == "structure_topology" and not args.structure:
            return ValidationResult(
                result=False, message="structure_topology requires structure."
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = TDAToolInput(**args)

        try:
            if input_data.action == "persistence_diagram":
                return self._persistence_diagram(input_data)
            if input_data.action == "persistence_image":
                return self._persistence_image(input_data)
            if input_data.action == "bottleneck_distance":
                return self._bottleneck_distance(input_data)
            if input_data.action == "landscape":
                return self._landscape(input_data)
            if input_data.action == "energy_landscape_topology":
                return self._energy_landscape_topology(input_data)
            if input_data.action == "structure_topology":
                return self._structure_topology(input_data)
            return ToolResult(
                data=None, success=False, error=f"Unknown action: {input_data.action}"
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"TDA action '{input_data.action}' failed: {e}",
            )

    # ── persistence diagram ───────────────────────────────────────────

    def _persistence_diagram(self, args: TDAToolInput) -> ToolResult:
        if not args.point_cloud:
            return ToolResult(
                data=None, success=False, error="point_cloud is required."
            )
        points = np.asarray(args.point_cloud, dtype=float)

        if args.distance_matrix:
            if points.ndim != 2 or points.shape[0] != points.shape[1]:
                return ToolResult(
                    data=None,
                    success=False,
                    error="distance_matrix=True requires a square point_cloud.",
                )
            dist = points
            n = int(points.shape[0])
        else:
            if points.ndim != 2:
                return ToolResult(
                    data=None, success=False, error="point_cloud must be N x D."
                )
            dist = None
            n = int(points.shape[0])

        pairs: list[dict] | None = None
        backend = "fallback"

        # 1) ripser — fastest, dedicated VR engine
        if pairs is None:
            try:
                from ripser import ripser

                result = ripser(
                    points,
                    maxdim=args.max_dim,
                    distance_matrix=args.distance_matrix,
                )
                tmp: list[dict] = []
                for dim, dgm in enumerate(result["dgms"]):
                    if dim > args.max_dim:
                        break
                    for row in dgm:
                        tmp.append(self._pair(dim, row[0], row[1]))
                pairs = tmp
                backend = "ripser"
            except ImportError:
                pass
            except Exception:
                # Numerical edge cases (empty input, degenerate distances) —
                # let the next backend have a go.
                pass

        # 2) gudhi — full-featured, slower but reliable
        if pairs is None:
            try:
                import gudhi

                if args.distance_matrix:
                    max_edge = float(dist.max()) if dist.size else 0.0
                    rips = gudhi.RipsComplex(
                        distance_matrix=dist.tolist(), max_edge_length=max_edge
                    )
                else:
                    from scipy.spatial.distance import pdist

                    max_edge = float(pdist(points).max()) if n > 1 else 0.0
                    rips = gudhi.RipsComplex(
                        points=points.tolist(), max_edge_length=max_edge
                    )
                st = rips.create_simplex_tree(max_dimension=args.max_dim + 1)
                persistence = st.persistence()
                pairs = [
                    self._pair(dim, b, d)
                    for dim, (b, d) in persistence
                    if dim <= args.max_dim
                ]
                backend = "gudhi"
            except ImportError:
                pass
            except Exception:
                pass

        # 3) scipy fallback — exact H0, approximate higher dimensions
        if pairs is None:
            pairs = self._persistence_fallback(
                points if not args.distance_matrix else dist,
                args.distance_matrix,
                args.max_dim,
            )
            backend = "fallback"

        betti_numbers = self._betti_from_pairs(pairs, args.max_dim)

        return ToolResult(
            data={
                "action": "persistence_diagram",
                "backend": backend,
                "diagram": pairs,
                "n_points": n,
                "max_dim": args.max_dim,
                "betti_numbers": betti_numbers,
            },
            success=True,
        )

    def _persistence_fallback(
        self, points: np.ndarray, distance_matrix: bool, max_dim: int
    ) -> list[dict]:
        """Scipy-only persistence: exact H0 via union-find, higher dims via
        Betti-number tracking across the edge filtration.

        Only invoked when neither ripser nor gudhi is available. Higher-dim
        bars are derived from how Betti numbers evolve across scales, which is
        an approximation of the standard reduction algorithm but good enough
        for a degraded mode.
        """
        from scipy.spatial.distance import pdist, squareform

        n = int(points.shape[0])
        if distance_matrix:
            D = np.asarray(points, dtype=float)
        else:
            D = squareform(pdist(points)) if n > 1 else np.zeros((1, 1))

        pairs = self._h0_persistence(D, n)

        if max_dim >= 1 and 2 <= n <= 40:
            radii = sorted(
                {float(D[i, j]) for i in range(n) for j in range(i + 1, n)}
            )
            # Subsample the filtration so the flag complex is rebuilt at most
            # ~30 times — keeps the O(n^3) clique enumeration in check.
            if len(radii) > 30:
                idx = np.linspace(0, len(radii) - 1, 30).astype(int)
                radii = [radii[i] for i in idx]

            b1_list: list[int] = []
            b2_list: list[int] = []
            for r in radii:
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if D[i, j] <= r:
                            adj[i].add(j)
                            adj[j].add(i)
                _, b1, b2 = self._flag_complex_betti(adj, max_dim)
                b1_list.append(b1 if b1 is not None else 0)
                b2_list.append(b2 if b2 is not None else 0)

            pairs.extend(self._betti_to_bars(radii, b1_list, 1))
            if max_dim >= 2:
                pairs.extend(self._betti_to_bars(radii, b2_list, 2))

        return pairs

    @staticmethod
    def _h0_persistence(D: np.ndarray, n: int) -> list[dict]:
        """Exact H0 persistence of a Vietoris-Rips filtration via union-find.

        Vertices are born at time 0; each edge that merges two components kills
        the younger one. Exactly one component survives to infinity.
        """
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(D[i, j]), i, j))
        edges.sort()

        parent = list(range(n))
        # Birth time of each component (rooted at the oldest surviving vertex).
        comp_birth = [0.0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs: list[dict] = []
        for d, i, j in edges:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # Merge the younger component into the older one.
            if comp_birth[ri] <= comp_birth[rj]:
                older, younger = ri, rj
            else:
                older, younger = rj, ri
            parent[younger] = older
            pairs.append(
                TDATool._pair(0, comp_birth[younger], d)
            )
            # The merged component keeps the earlier birth time.
            comp_birth[older] = min(comp_birth[older], comp_birth[younger])

        # The single surviving component is the infinite bar.
        if n > 0:
            root = find(0)
            pairs.append(TDATool._pair(0, comp_birth[root], None))
        return pairs

    # ── persistence image ─────────────────────────────────────────────

    def _persistence_image(self, args: TDAToolInput) -> ToolResult:
        if not args.diagram:
            return ToolResult(
                data=None, success=False, error="diagram is required."
            )
        diagram = args.diagram
        res = max(2, args.resolution)
        sigma = args.sigma if args.sigma > 0 else 1.0

        image_vectors: dict[int, list[float]] = {}
        shapes: dict[int, list[int]] = {}
        dims = sorted({p["dim"] for p in diagram})

        for dim in dims:
            pts = [
                (float(p["birth"]), float(p["death"]) - float(p["birth"]))
                for p in diagram
                if p["dim"] == dim and p["death"] is not None
            ]
            if not pts:
                image_vectors[dim] = [0.0] * (res * res)
                shapes[dim] = [res, res]
                continue

            births = np.array([b for b, _ in pts])
            perts = np.array([p for _, p in pts])
            b_min, b_max = float(births.min()), float(births.max())
            p_min, p_max = float(perts.min()), float(perts.max())
            if b_max == b_min:
                b_max = b_min + 1.0
            if p_max == p_min:
                p_max = p_min + 1.0

            xs = np.linspace(b_min, b_max, res)
            ys = np.linspace(p_min, p_max, res)
            XX, YY = np.meshgrid(xs, ys)
            img = np.zeros((res, res))
            for b, p in pts:
                img += np.exp(-((XX - b) ** 2 + (YY - p) ** 2) / (2.0 * sigma * sigma))

            image_vectors[dim] = img.flatten().tolist()
            shapes[dim] = [res, res]

        return ToolResult(
            data={
                "action": "persistence_image",
                "image_vectors": image_vectors,
                "shapes": shapes,
                "resolution": res,
                "sigma": sigma,
            },
            success=True,
        )

    # ── bottleneck distance ───────────────────────────────────────────

    def _bottleneck_distance(self, args: TDAToolInput) -> ToolResult:
        if not args.diagram or not args.diagram2:
            return ToolResult(
                data=None,
                success=False,
                error="diagram and diagram2 are required.",
            )
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="scipy is required for bottleneck_distance.",
            )

        dims = sorted(
            {p["dim"] for p in args.diagram} | {p["dim"] for p in args.diagram2}
        )
        max_dist = 0.0
        total_matched = 0
        per_dim: dict[int, float] = {}

        for dim in dims:
            d1 = [
                (float(p["birth"]), float(p["death"]))
                for p in args.diagram
                if p["dim"] == dim and p["death"] is not None
            ]
            d2 = [
                (float(p["birth"]), float(p["death"]))
                for p in args.diagram2
                if p["dim"] == dim and p["death"] is not None
            ]
            dist, matched = self._bottleneck_dim(d1, d2, linear_sum_assignment)
            per_dim[dim] = dist
            max_dist = max(max_dist, dist)
            total_matched += matched

        return ToolResult(
            data={
                "action": "bottleneck_distance",
                "distance": float(max_dist),
                "matched_pairs": int(total_matched),
                "per_dimension": per_dim,
            },
            success=True,
        )

    @staticmethod
    def _bottleneck_dim(
        d1: list[tuple[float, float]],
        d2: list[tuple[float, float]],
        lsa: Any,
    ) -> tuple[float, int]:
        """Bottleneck distance for a single dimension.

        Builds an (m+n)x(m+n) cost matrix where off-diagonal points can match
        each other (L-infinity cost) or match the diagonal (cost = half their
        persistence). The Hungarian algorithm gives an assignment; the
        bottleneck is the largest finite cost used.
        """
        m, n = len(d1), len(d2)
        if m == 0 and n == 0:
            return 0.0, 0

        size = m + n
        BIG = 1e18
        C = np.full((size, size), BIG)

        for i in range(m):
            b1, dd1 = d1[i]
            for j in range(n):
                b2, dd2 = d2[j]
                C[i, j] = max(abs(b1 - b2), abs(dd1 - dd2))

        # Match d1 points to the diagonal.
        for i in range(m):
            b1, dd1 = d1[i]
            C[i, n + i] = (dd1 - b1) / 2.0
        # Match d2 points to the diagonal.
        for j in range(n):
            b2, dd2 = d2[j]
            C[m + j, j] = (dd2 - b2) / 2.0
        # Diagonal-to-diagonal pairings are free.
        for j in range(n):
            for i in range(m):
                C[m + j, n + i] = 0.0

        row_ind, col_ind = lsa(C)
        dist = 0.0
        matched = 0
        for r, c in zip(row_ind, col_ind):
            cost = C[r, c]
            if cost >= BIG:
                continue
            if cost > dist:
                dist = float(cost)
            if r < m and c < n:
                matched += 1
        return dist, matched

    # ── persistence landscape ─────────────────────────────────────────

    def _landscape(self, args: TDAToolInput) -> ToolResult:
        if not args.diagram:
            return ToolResult(data=None, success=False, error="diagram is required.")
        diagram = args.diagram
        res = max(2, args.resolution)
        k = max(1, args.k)

        finite = [
            (float(p["birth"]), float(p["death"]))
            for p in diagram
            if p["death"] is not None
        ]
        if not finite:
            return ToolResult(
                data={
                    "action": "landscape",
                    "landscapes": [[0.0] * res for _ in range(k)],
                    "x_values": [0.0] * res,
                    "k": k,
                },
                success=True,
            )

        births = np.array([b for b, _ in finite])
        deaths = np.array([d for _, d in finite])
        x_min = float(births.min())
        x_max = float(deaths.max())
        if x_max == x_min:
            x_max = x_min + 1.0
        xs = np.linspace(x_min, x_max, res)

        # Each (b, d) pair is a tent: 0 at b, peak (d-b)/2 at (b+d)/2, 0 at d.
        tents = []
        for b, d in finite:
            tent = np.minimum(xs - b, d - xs)
            tents.append(np.maximum(tent, 0.0))
        tent_matrix = np.array(tents)  # (n_tents, res)

        # Sort each column descending so row i holds the i-th largest value.
        tent_matrix = -np.sort(-tent_matrix, axis=0)
        landscapes: list[list[float]] = []
        for i in range(k):
            if i < tent_matrix.shape[0]:
                landscapes.append(tent_matrix[i].tolist())
            else:
                landscapes.append([0.0] * res)

        return ToolResult(
            data={
                "action": "landscape",
                "landscapes": landscapes,
                "x_values": xs.tolist(),
                "k": k,
            },
            success=True,
        )

    # ── energy landscape topology ─────────────────────────────────────

    def _energy_landscape_topology(self, args: TDAToolInput) -> ToolResult:
        if not args.energies or not args.structures:
            return ToolResult(
                data=None,
                success=False,
                error="energies and structures are required.",
            )
        energies = np.asarray(args.energies, dtype=float)
        structs = np.asarray(args.structures, dtype=float)
        n = int(len(energies))
        if structs.shape[0] != n:
            return ToolResult(
                data=None,
                success=False,
                error="energies and structures must have the same length.",
            )
        if n == 0:
            return ToolResult(
                data=None, success=False, error="No structures provided."
            )

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial.distance import pdist, squareform

        e_diff = squareform(pdist(energies.reshape(-1, 1))) if n > 1 else np.zeros((1, 1))
        d_mat = squareform(pdist(structs)) if n > 1 else np.zeros((1, 1))

        # Energy cutoff: user-supplied, otherwise the median pairwise gap so
        # roughly half the structure pairs are within reach.
        if args.threshold is None:
            off = e_diff[np.triu_indices(n, 1)]
            e_thr = float(np.percentile(off, 50)) if off.size else 0.0
        else:
            e_thr = float(args.threshold)
        # Spatial cutoff is always auto — median pairwise distance.
        off_d = d_mat[np.triu_indices(n, 1)]
        d_thr = float(np.percentile(off_d, 50)) if off_d.size else 0.0

        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                if e_diff[i, j] <= e_thr and d_mat[i, j] <= d_thr:
                    w = 1.0 / (1.0 + e_diff[i, j])
                    rows.extend([i, j])
                    cols.extend([j, i])
                    weights.extend([w, w])

        adj = csr_matrix(
            (weights, (rows, cols)), shape=(n, n)
        ) if rows else csr_matrix((n, n))
        n_comp, labels = connected_components(csgraph=adj, directed=False)

        sizes = [int(np.sum(labels == c)) for c in range(n_comp)]
        sizes.sort(reverse=True)

        n_edges = len(rows) // 2
        # Betti-1 = E - V + C (independent cycles via spanning-tree edge count).
        betti1 = n_edges - n + n_comp
        density = (2.0 * n_edges) / (n * (n - 1)) if n > 1 else 0.0

        # Dense transition matrix only for modest graphs; otherwise ship the
        # edge list so we don't balloon the payload.
        trans = adj.toarray().tolist() if n <= 80 else None
        edges = [
            [rows[2 * t], cols[2 * t], weights[2 * t]] for t in range(n_edges)
        ]

        return ToolResult(
            data={
                "action": "energy_landscape_topology",
                "n_structures": n,
                "energy_threshold": e_thr,
                "spatial_threshold": d_thr,
                "n_basins": int(n_comp),
                "n_pathways": int(betti1),
                "n_edges": int(n_edges),
                "connectivity": float(density),
                "basin_sizes": sizes,
                "transition_matrix": trans,
                "edges": edges,
            },
            success=True,
        )

    # ── structure topology ────────────────────────────────────────────

    def _structure_topology(self, args: TDAToolInput) -> ToolResult:
        if not args.structure:
            return ToolResult(
                data=None, success=False, error="structure is required."
            )
        try:
            positions, lattice = self._parse_structure(args.structure)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to parse structure: {e}"
            )

        n = int(len(positions))
        if n == 0:
            return ToolResult(
                data=None, success=False, error="Structure has no sites."
            )

        D = self._pbc_distance_matrix(positions, lattice)
        min_lat = float(np.min(np.linalg.norm(lattice, axis=1)))

        if args.radii:
            radii = sorted(float(r) for r in args.radii)
        else:
            # Sweep from short bonds out to roughly half the shortest cell
            # vector — beyond that the minimum-image convention double counts.
            radii = [float(r) for r in np.linspace(0.3, 0.5, 8) * min_lat]

        betti0_list: list[int] = []
        betti1_list: list[int] = []
        betti2_list: list[int] = []
        for r in radii:
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i, j] <= r:
                        adj[i].add(j)
                        adj[j].add(i)
            b0, b1, b2 = self._flag_complex_betti(adj, max_dim=3)
            betti0_list.append(b0)
            betti1_list.append(b1 if b1 is not None else 0)
            betti2_list.append(b2 if b2 is not None else 0)

        bars0 = self._betti_to_bars(radii, betti0_list, 0)
        bars1 = self._betti_to_bars(radii, betti1_list, 1)
        bars2 = self._betti_to_bars(radii, betti2_list, 2)

        return ToolResult(
            data={
                "action": "structure_topology",
                "n_atoms": n,
                "radii": radii,
                "betti_0": betti0_list,
                "betti_1": betti1_list,
                "betti_2": betti2_list,
                "persistence_bars": bars0 + bars1 + bars2,
            },
            success=True,
        )

    # ── shared helpers ────────────────────────────────────────────────

    @staticmethod
    def _pair(dim: int, birth: Any, death: Any) -> dict:
        """Normalise a persistence pair to a JSON-safe dict.

        Infinite deaths (ripser/gudhi return +inf) become None so the result
        survives serialisation downstream.
        """
        dim = int(dim)
        birth = float(birth)
        if death is None:
            return {"dim": dim, "birth": birth, "death": None, "persistence": None}
        d = float(death)
        if not np.isfinite(d):
            return {"dim": dim, "birth": birth, "death": None, "persistence": None}
        return {
            "dim": dim,
            "birth": birth,
            "death": d,
            "persistence": d - birth,
        }

    @staticmethod
    def _betti_from_pairs(pairs: list[dict], max_dim: int) -> dict[int, int]:
        """Betti numbers = count of infinite bars per dimension."""
        betti: dict[int, int] = {d: 0 for d in range(max_dim + 1)}
        for p in pairs:
            if p["death"] is None:
                betti[p["dim"]] = betti.get(p["dim"], 0) + 1
        return betti

    @staticmethod
    def _betti_to_bars(
        radii: list[float], betti: list[int], dim: int
    ) -> list[dict]:
        """Turn a Betti-number sequence across a filtration into bars.

        A rise in Betti means new features are born; a fall means features die.
        Deaths are matched to the most recent births (FIFO), which is an
        approximation of the standard persistence reduction but cheap and
        stable enough for descriptor extraction.
        """
        bars: list[dict] = []
        active: list[float] = []
        prev = 0
        for r, b in zip(radii, betti):
            delta = b - prev
            if delta > 0:
                active.extend([r] * delta)
            elif delta < 0:
                for _ in range(-delta):
                    if active:
                        br = active.pop(0)
                        bars.append(TDATool._pair(dim, br, r))
            prev = b
        for br in active:
            bars.append(TDATool._pair(dim, br, None))
        return bars

    @staticmethod
    def _gf2_rank(columns: list[int]) -> int:
        """Rank of a binary matrix (columns as int bitmasks over rows).

        Standard Gaussian elimination over GF(2): pivot on the highest set
        bit of each column and XOR away. Cheap for the modest simplex counts
        we deal with here.
        """
        pivots: dict[int, int] = {}
        for col in columns:
            v = col
            while v:
                hb = v.bit_length() - 1
                if hb in pivots:
                    v ^= pivots[hb]
                else:
                    pivots[hb] = v
                    break
        return len(pivots)

    @staticmethod
    def _flag_complex_betti(
        adj: list[set[int]], max_dim: int = 3
    ) -> tuple[int, int | None, int | None]:
        """Betti numbers of the flag (clique) complex of a neighbour graph.

        Builds simplices up to dimension 3 and computes ranks of the GF(2)
        boundary matrices. Returns (b0, b1, b2); b1/b2 are None when the
        clique count blew past the safety budget and the boundary rank could
        not be determined reliably.
        """
        n = len(adj)
        if n == 0:
            return 0, 0, 0

        # C1: edges
        edges: list[tuple[int, int]] = []
        edge_index: dict[tuple[int, int], int] = {}
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edge_index[(i, j)] = len(edges)
                    edges.append((i, j))

        d1_cols = [(1 << i) | (1 << j) for (i, j) in edges]
        rank_d1 = TDATool._gf2_rank(d1_cols)
        betti0 = n - rank_d1
        if max_dim < 1:
            return betti0, 0, 0

        # C2: triangles (3-cliques)
        tri_budget = 50000
        triangles: list[tuple[int, int, int]] = []
        tri_index: dict[tuple[int, int, int], int] = {}
        tri_complete = True
        for (i, j) in edges:
            common = adj[i] & adj[j]
            for k in common:
                if k > j:
                    tri_index[(i, j, k)] = len(triangles)
                    triangles.append((i, j, k))
                    if len(triangles) >= tri_budget:
                        tri_complete = False
                        break
            if not tri_complete:
                break

        d2_cols = [
            (1 << edge_index[(i, j)])
            | (1 << edge_index[(i, k)])
            | (1 << edge_index[(j, k)])
            for (i, j, k) in triangles
        ]
        rank_d2 = TDATool._gf2_rank(d2_cols)
        betti1 = len(edges) - rank_d1 - rank_d2
        if max_dim < 2 or not tri_complete:
            return betti0, betti1, None

        # C3: tetrahedra (4-cliques) — only when triangles are complete and
        # the system is small enough that 4-cliques stay tractable.
        tetra_budget = 5000
        tetra: list[tuple[int, int, int, int]] = []
        tetra_complete = True
        if n <= 50:
            for (i, j, k) in triangles:
                common = adj[i] & adj[j] & adj[k]
                for l in common:
                    if l > k:
                        tetra.append((i, j, k, l))
                        if len(tetra) >= tetra_budget:
                            tetra_complete = False
                            break
                if not tetra_complete:
                    break

        d3_cols: list[int] = []
        for (i, j, k, l) in tetra:
            f1 = tri_index.get((i, j, k))
            f2 = tri_index.get((i, j, l))
            f3 = tri_index.get((i, k, l))
            f4 = tri_index.get((j, k, l))
            if f1 is None or f2 is None or f3 is None or f4 is None:
                continue
            d3_cols.append((1 << f1) | (1 << f2) | (1 << f3) | (1 << f4))
        rank_d3 = TDATool._gf2_rank(d3_cols)
        betti2 = len(triangles) - rank_d2 - rank_d3
        if not tetra_complete:
            return betti0, betti1, None
        return betti0, betti1, betti2

    @staticmethod
    def _parse_structure(structure: dict) -> tuple[np.ndarray, np.ndarray]:
        """Pull Cartesian positions and a 3x3 lattice matrix out of a
        {lattice, sites} dict.

        Mirrors the format used by the other structure tools: lattice may be
        a dict of cell parameters (a, b, c, alpha, beta, gamma) or a 3x3 row
        matrix; sites carry either cartesian 'xyz' or fractional 'abc'.
        """
        lattice = structure.get("lattice")
        sites = structure.get("sites", [])
        if not sites:
            raise ValueError("Structure dict contains no sites")

        positions: list[list[float]] = []
        use_fractional = False
        for site in sites:
            if "xyz" in site:
                positions.append(list(site["xyz"]))
            elif "abc" in site:
                positions.append(list(site["abc"]))
                use_fractional = True
            else:
                positions.append([0.0, 0.0, 0.0])
        pos = np.asarray(positions, dtype=float)

        if isinstance(lattice, dict):
            lat = TDATool._cellpar_to_matrix(
                lattice.get("a", 1.0),
                lattice.get("b", 1.0),
                lattice.get("c", 1.0),
                lattice.get("alpha", 90.0),
                lattice.get("beta", 90.0),
                lattice.get("gamma", 90.0),
            )
        elif isinstance(lattice, list):
            lat = np.asarray(lattice, dtype=float)
            if lat.shape != (3, 3):
                raise ValueError("Lattice matrix must be 3x3")
        else:
            raise ValueError("Invalid lattice format in structure dict")

        if use_fractional:
            pos = pos @ lat
        return pos, lat

    @staticmethod
    def _cellpar_to_matrix(
        a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> np.ndarray:
        """Build a 3x3 lattice matrix (row vectors) from cell parameters.

        Angles come in degrees, as in crystallography. The convention matches
        `positions @ lattice` for fractional-to-Cartesian conversion.
        """
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)
        cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
        sin_g = np.sin(gamma_r)

        ax, ay, az = float(a), 0.0, 0.0
        bx, by, bz = float(b) * cos_g, float(b) * sin_g, 0.0
        cx = float(c) * cos_b
        cy = float(c) * (cos_a - cos_b * cos_g) / sin_g if sin_g > 0 else 0.0
        cz_sq = float(c) ** 2 - cx * cx - cy * cy
        cz = np.sqrt(cz_sq) if cz_sq > 0 else 0.0
        return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])

    @staticmethod
    def _pbc_distance_matrix(positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """Pairwise distances under periodic boundary conditions.

        Uses the minimum-image convention in fractional space. Falls back to
        plain Euclidean distances if the lattice is singular (e.g. a molecule
        with a dummy unit cell).
        """
        n = len(positions)
        try:
            inv_lat = np.linalg.inv(lattice)
            frac = positions @ inv_lat
            D = np.zeros((n, n))
            for i in range(n):
                diff = frac - frac[i]
                diff -= np.round(diff)
                cart = diff @ lattice
                D[i] = np.sqrt(np.sum(cart * cart, axis=1))
            return D
        except np.linalg.LinAlgError:
            from scipy.spatial.distance import pdist, squareform

            return squareform(pdist(positions)) if n > 1 else np.zeros((1, 1))
