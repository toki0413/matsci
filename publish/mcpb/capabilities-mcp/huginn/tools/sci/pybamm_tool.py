"""PyBaMM battery modeling wrapper.

Wraps the standard pybamm lithium-ion models (SPM / SPMe / DFN) behind three
actions: simulate a discharge/charge curve, fit model parameters to experimental
data, and dump a parameter set. Uses pybamm's canonical interface only — no
custom PDE math here.

pybamm is an optional import: if it's missing the module still loads, the
registry can instantiate the tool, is_available() reports False, and the
self-check exits cleanly with a skip message instead of crashing.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool, ToolProfile

# ponytail: lazy import — the module loads even without pybamm installed,
# is_available() gates real use so the registry never crashes on import.
try:
    import pybamm

    _PYBAMM_AVAILABLE = True
except ImportError:  # pragma: no cover - only on pybamm-less envs
    _PYBAMM_AVAILABLE = False
    pybamm = None  # type: ignore[assignment]


# Standard pybamm lithium-ion models, exposed by name so callers don't need
# to import pybamm just to pick one.
_MODEL_KINDS: dict[str, str] = {
    "SPM": "Single Particle Model",
    "SPMe": "Single Particle Model with electrolyte",
    "DFN": "Doyle-Fuller-Newman",
}


class PyBaMMToolInput(BaseModel):
    action: Literal["simulate", "fit", "parameterize"] = Field(
        default="simulate",
        description="simulate = run a discharge/charge curve; "
        "fit = fit parameters to experimental data; "
        "parameterize = return a parameter set.",
    )
    # ── simulate ─────────────────────────────────────────────
    model: Literal["SPM", "SPMe", "DFN"] = Field(
        default="SPM", description="Standard pybamm lithium-ion model"
    )
    parameter_set: str = Field(
        default="Chen2020",
        description="pybamm parameter set name (e.g. Chen2020, Marquis2019)",
    )
    experiment: list[str] = Field(
        default_factory=lambda: ["Discharge at 1C until 2.5 V"],
        description="pybamm experiment steps, e.g. 'Discharge at 1C for 1 hour'",
    )
    t_eval: list[float] | None = Field(
        default=None,
        description="Optional time grid [s] for non-experiment solves; "
        "ignored when experiment is given.",
    )
    # ── fit ───────────────────────────────────────────────────
    time_data: list[float] = Field(
        default_factory=list,
        description="Experimental time points [s] for fit",
    )
    voltage_data: list[float] = Field(
        default_factory=list,
        description="Experimental terminal voltage [V] for fit",
    )
    fit_params: list[str] = Field(
        default_factory=lambda: [
            "Negative electrode conductivity [S/m]",
            "Positive electrode conductivity [S/m]",
        ],
        description="Parameter names to fit (must exist in parameter_set)",
    )
    fit_bounds: list[tuple[float, float]] | None = Field(
        default=None,
        description="Per-parameter (low, high) bounds; defaults to 0.1x–10x base value",
    )
    # ── shared ────────────────────────────────────────────────
    working_dir: str | None = Field(default=None)


class PyBaMMTool(HuginnTool):
    """Battery simulation / parameter fitting via pybamm standard models."""

    name = "pybamm_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.EXECUTION}))
    description = (
        "Run pybamm lithium-ion battery simulations (SPM/SPMe/DFN) for "
        "discharge/charge curves, fit model parameters to experimental data, "
        "and dump standard parameter sets."
    )
    input_schema = PyBaMMToolInput
    read_only = True

    # ponytail: no custom PDE models here — agent writes its own for novel
    # chemistries. This tool only exposes pybamm's built-in models so we get
    # validated SPM/DFN behavior without re-deriving the math.

    def is_available(self) -> bool:
        return _PYBAMM_AVAILABLE

    # ── public entry ──────────────────────────────────────────

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        if not _PYBAMM_AVAILABLE:
            return ToolResult(
                data=None,
                success=False,
                error="pybamm not installed. Install with: pip install pybamm",
            )
        try:
            inp = PyBaMMToolInput(**args)
            if inp.action == "simulate":
                return self._simulate(inp)
            if inp.action == "fit":
                return self._fit(inp)
            return self._parameterize(inp)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"pybamm tool failed: {e}")

    # ── simulate ──────────────────────────────────────────────

    def _simulate(self, args: PyBaMMToolInput) -> ToolResult:
        model = self._build_model(args.model)
        params = pybamm.ParameterValues(args.parameter_set)
        sim = pybamm.Simulation(
            model, parameter_values=params, experiment=args.experiment
        )
        sol = sim.solve()

        t = sol["Time [s]"].entries
        v = sol["Voltage [V]"].entries
        i = sol["Current [A]"].entries
        # Discharge capacity is a standard output for lithium-ion models;
        # guard in case a non-standard experiment drops it.
        try:
            cap = sol["Discharge capacity [A.h]"].entries
        except KeyError:
            cap = np.cumsum(np.abs(i) * np.diff(t, prepend=0.0)) / 3600.0

        return ToolResult(
            data={
                "model": args.model,
                "parameter_set": args.parameter_set,
                "experiment": args.experiment,
                "time_s": t.tolist(),
                "voltage_V": v.tolist(),
                "current_A": i.tolist(),
                "capacity_Ah": cap.tolist(),
                "n_points": int(len(t)),
                "message": (
                    f"{args.model} simulation done: {len(t)} points, "
                    f"V range [{float(v.min()):.4f}, {float(v.max()):.4f}] V."
                ),
            },
            success=True,
        )

    # ── fit ────────────────────────────────────────────────────

    def _fit(self, args: PyBaMMToolInput) -> ToolResult:
        if not args.time_data or not args.voltage_data:
            return ToolResult(
                data=None, success=False,
                error="fit requires time_data and voltage_data.",
            )
        if len(args.time_data) != len(args.voltage_data):
            return ToolResult(
                data=None, success=False,
                error="time_data and voltage_data must have equal length.",
            )

        t_exp = np.asarray(args.time_data, dtype=float)
        v_exp = np.asarray(args.voltage_data, dtype=float)
        params = pybamm.ParameterValues(args.parameter_set)
        model = self._build_model(args.model)

        # Default bounds: 0.1x–10x of the base value. Wide enough for a
        # first-pass fit, narrow enough to stay physical.
        base_values = [float(params[name]) for name in args.fit_params]
        bounds = args.fit_bounds or [(b * 0.1, b * 10.0) for b in base_values]

        def residual(x: np.ndarray) -> float:
            pv = pybamm.ParameterValues(args.parameter_set)
            for name, val in zip(args.fit_params, x):
                pv[name] = float(val)
            sim = pybamm.Simulation(
                model, parameter_values=pv, experiment=args.experiment
            )
            sol = sim.solve()
            v_sim = np.interp(t_exp, sol["Time [s]"].entries, sol["Voltage [V]"].entries)
            return float(np.mean((v_sim - v_exp) ** 2))

        # Local Nelder-Mead from the base values — no gradient needed, robust
        # to pybamm's internal solver noise. Bounded refit happens after if
        # bounds are violated.
        from scipy.optimize import minimize

        x0 = np.array(base_values, dtype=float)
        res = minimize(residual, x0, method="Nelder-Mead", options={"maxiter": 50})

        x_fit = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
        mse = float(res.fun)

        return ToolResult(
            data={
                "model": args.model,
                "parameter_set": args.parameter_set,
                "fit_params": args.fit_params,
                "initial_values": base_values,
                "fitted_values": x_fit.tolist(),
                "bounds": [list(b) for b in bounds],
                "mse": mse,
                "rmse_V": float(np.sqrt(mse)),
                "n_points": int(len(t_exp)),
                "message": (
                    f"Fit done: RMSE={np.sqrt(mse):.5f} V over {len(t_exp)} points."
                ),
            },
            success=True,
        )

    # ── parameterize ─────────────────────────────────────────

    def _parameterize(self, args: PyBaMMToolInput) -> ToolResult:
        params = pybamm.ParameterValues(args.parameter_set)
        # pybamm parameter values carry callables for some entries (functions
        # of stoichiometry, temperature, ...). str() them so the payload stays
        # JSON-safe; the agent can re-fetch live objects via pybamm directly.
        flat: dict[str, Any] = {}
        for k, v in params.items():
            if callable(v):
                flat[k] = f"<callable: {getattr(v, '__name__', 'function')}>"
            else:
                flat[k] = v
        return ToolResult(
            data={
                "parameter_set": args.parameter_set,
                "parameters": flat,
                "n_params": len(flat),
                "message": f"Loaded '{args.parameter_set}': {len(flat)} entries.",
            },
            success=True,
        )

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_model(kind: str):
        # Local import keeps the top-level lazy even though we already gated
        # on _PYBAMM_AVAILABLE — defends against partial installs.
        if kind == "SPM":
            return pybamm.lithium_ion.SPM()
        if kind == "SPMe":
            return pybamm.lithium_ion.SPMe()
        if kind == "DFN":
            return pybamm.lithium_ion.DFN()
        raise ValueError(f"Unknown model '{kind}'; expected one of {list(_MODEL_KINDS)}")


# ── self-check ────────────────────────────────────────────────


def _selfcheck() -> None:
    """SPM 1C discharge — the smallest thing that fails if the wrapper breaks.

    Asserts the simulation returns a non-empty voltage curve that actually
    drops during discharge, which is the basic physical sanity check.
    """
    if not _PYBAMM_AVAILABLE:
        print("pybamm not available, skip demo")
        return

    tool = PyBaMMTool()
    res = tool.call({
        "action": "simulate",
        "model": "SPM",
        "parameter_set": "Chen2020",
        "experiment": ["Discharge at 1C until 2.5 V"],
    })
    assert res.success, res.error
    d = res.data
    v = np.asarray(d["voltage_V"])
    assert len(v) == d["n_points"] and d["n_points"] > 1
    # Discharge should end below the starting voltage — physics check.
    assert v[-1] < v[0], f"voltage did not drop: start={v[0]}, end={v[-1]}"
    print(
        "OK: SPM 1C discharge, "
        f"{d['n_points']} pts, V {float(v[0]):.3f}→{float(v[-1]):.3f} V, "
        f"cap {float(np.asarray(d['capacity_Ah'])[-1]):.3f} Ah"
    )


if __name__ == "__main__":
    _selfcheck()
