"""High-throughput workflow orchestration for parameter sweeps and screening.

Generates parameter combinations from a design space (grid, random, or Latin
hypercube), submits jobs, tracks status, and aggregates results. Designed
for computational materials science workflows like convergence testing,
property screening, and structure-property mapping.

Usage:
    from huginn.workflows.high_throughput import ParameterSweep, GridSpace

    sweep = ParameterSweep(
        name="encut_convergence",
        tool_name="vasp_tool",
        parameter_space=GridSpace({
            "encut": [300, 400, 500, 600],
            "kpoints": ["2 2 2", "4 4 4", "6 6 6"],
        }),
        base_input={"structure": "POSCAR", "mode": "scf"},
    )
    jobs = sweep.generate_jobs()
    # Submit and track...
    results = sweep.aggregate_results(job_results)
"""

from __future__ import annotations

import itertools
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Parameter spaces ──────────────────────────────────────────────────

class ParameterSpace(ABC):
    """Abstract base for parameter space designs."""

    @abstractmethod
    def sample(self) -> list[dict[str, Any]]:
        """Return a list of parameter combinations."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Total number of parameter combinations."""
        ...


class GridSpace(ParameterSpace):
    """Full factorial grid over discrete parameter values."""

    def __init__(self, params: dict[str, list[Any]]) -> None:
        self.params = params

    def sample(self) -> list[dict[str, Any]]:
        keys = list(self.params.keys())
        values = list(self.params.values())
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    @property
    def size(self) -> int:
        n = 1
        for v in self.params.values():
            n *= len(v)
        return n


class RandomSpace(ParameterSpace):
    """Random sampling from uniform distributions over parameter ranges."""

    def __init__(
        self,
        params: dict[str, tuple[float, float]],
        n_samples: int = 20,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.n_samples = n_samples
        self._rng = random.Random(seed)

    def sample(self) -> list[dict[str, Any]]:
        results = []
        for _ in range(self.n_samples):
            combo = {}
            for key, (lo, hi) in self.params.items():
                combo[key] = self._rng.uniform(lo, hi)
            results.append(combo)
        return results

    @property
    def size(self) -> int:
        return self.n_samples


class LatinHypercubeSpace(ParameterSpace):
    """Latin Hypercube Sampling for efficient space-filling designs."""

    def __init__(
        self,
        params: dict[str, tuple[float, float]],
        n_samples: int = 20,
        seed: int | None = None,
    ) -> None:
        self.params = params
        self.n_samples = n_samples
        self._rng = random.Random(seed)

    def sample(self) -> list[dict[str, Any]]:
        n = self.n_samples
        dim = len(self.params)
        keys = list(self.params.keys())

        # Generate LHS: divide each dimension into n equal bins,
        # shuffle, and pick one point per bin.
        result = []
        bins = list(range(n))
        permuted_bins = []
        for _ in range(dim):
            perm = list(bins)
            self._rng.shuffle(perm)
            permuted_bins.append(perm)

        for i in range(n):
            combo = {}
            for j, key in enumerate(keys):
                lo, hi = self.params[key]
                bin_idx = permuted_bins[j][i]
                # Random point within the bin
                frac = (bin_idx + self._rng.random()) / n
                combo[key] = lo + frac * (hi - lo)
            result.append(combo)
        return result

    @property
    def size(self) -> int:
        return self.n_samples


# ── Job and sweep definitions ─────────────────────────────────────────

@dataclass
class SweepJob:
    """A single job in a parameter sweep."""

    job_id: str
    tool_name: str
    parameters: dict[str, Any]
    base_input: dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def full_input(self) -> dict[str, Any]:
        """Merge base_input with sweep parameters."""
        return {**self.base_input, **self.parameters}

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class ParameterSweep:
    """A high-throughput parameter sweep over a tool's input space.

    Generates jobs from a parameter space, tracks their status, and
    provides aggregation utilities for analyzing results.
    """

    name: str
    tool_name: str
    parameter_space: ParameterSpace
    base_input: dict[str, Any] = field(default_factory=dict)
    max_parallel: int = 4
    early_termination: str | None = None  # safe_eval expression for early stop

    _jobs: list[SweepJob] = field(default_factory=list, repr=False)

    def generate_jobs(self) -> list[SweepJob]:
        """Generate SweepJob objects for all parameter combinations."""
        combos = self.parameter_space.sample()
        self._jobs = [
            SweepJob(
                job_id=f"{self.name}_{i:04d}",
                tool_name=self.tool_name,
                parameters=combo,
                base_input=self.base_input,
            )
            for i, combo in enumerate(combos)
        ]
        return self._jobs

    @property
    def jobs(self) -> list[SweepJob]:
        return self._jobs

    @property
    def pending_jobs(self) -> list[SweepJob]:
        return [j for j in self._jobs if j.status == "pending"]

    @property
    def completed_jobs(self) -> list[SweepJob]:
        return [j for j in self._jobs if j.status == "completed"]

    @property
    def failed_jobs(self) -> list[SweepJob]:
        return [j for j in self._jobs if j.status == "failed"]

    def update_job(self, job_id: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Update the status of a job."""
        for job in self._jobs:
            if job.job_id == job_id:
                job.status = status
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error
                break

    def check_early_termination(self) -> bool:
        """Check if the early termination condition is met."""
        if self.early_termination is None:
            return False
        try:
            from huginn.security import safe_eval

            # Build context from completed jobs
            context = {
                "n_completed": len(self.completed_jobs),
                "n_total": len(self._jobs),
                "results": [j.result for j in self.completed_jobs if j.result],
            }
            return bool(safe_eval(self.early_termination, context))
        except Exception:
            return False

    def aggregate_results(self, job_results: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """Aggregate results from completed jobs.

        Args:
            job_results: Optional dict mapping job_id → result. If None,
                uses the results stored on each SweepJob.
        """
        results = []
        params = []

        for job in self._jobs:
            if job.status != "completed":
                continue
            r = job_results.get(job.job_id, job.result) if job_results else job.result
            if r is None:
                continue
            results.append(r)
            params.append(job.parameters)

        if not results:
            return {"n_completed": 0, "results": [], "parameters": []}

        # Extract common numeric keys for summary statistics
        all_keys = set()
        for r in results:
            if isinstance(r, dict):
                all_keys.update(k for k, v in r.items() if isinstance(v, (int, float)))

        summary: dict[str, dict[str, float]] = {}
        for key in all_keys:
            values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
            if values:
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "n": len(values),
                }

        # Find best result per metric (assumes higher is better unless prefixed with _min_)
        best: dict[str, dict[str, Any]] = {}
        for key in all_keys:
            values_with_params = [
                (r[key], p) for r, p in zip(results, params)
                if isinstance(r.get(key), (int, float))
            ]
            if values_with_params:
                best_val, best_params = max(values_with_params, key=lambda x: x[0])
                best[key] = {"value": best_val, "parameters": best_params}

        return {
            "n_completed": len(results),
            "n_failed": len(self.failed_jobs),
            "n_total": len(self._jobs),
            "summary_stats": summary,
            "best_results": best,
            "all_results": results,
            "all_parameters": params,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tool_name": self.tool_name,
            "max_parallel": self.max_parallel,
            "jobs": [j.to_dict() for j in self._jobs],
            "n_total": len(self._jobs),
            "n_completed": len(self.completed_jobs),
            "n_failed": len(self.failed_jobs),
            "n_pending": len(self.pending_jobs),
        }
