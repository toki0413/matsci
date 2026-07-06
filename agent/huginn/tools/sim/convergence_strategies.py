"""Progressive SCF convergence strategies for DFT calculations.

When SCF fails to converge, try strategies in order of increasing cost:
1. Reduce mixing alpha (0.7 -> 0.4 -> 0.2)
2. Change ALGO (Fast -> Normal -> All)
3. Increase NELM (60 -> 100 -> 200)
4. Add AMIX/BMIX parameters
5. Switch to ALGO=Exact (last resort)

Works with both VASP (ALGO, NELM, AMIX, BMIX, IALGO) and QE
(mixing_beta, mixing_type, conv_thr, diago_thr_init).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConvergenceStrategy:
    """A single SCF convergence fix attempt.

    Fields:
        name: short identifier (used to track which strategies were tried)
        param_changes: code-specific parameter overrides.
            VASP keys are uppercase (ALGO, NELM, AMIX, BMIX, IALGO),
            QE keys are lowercase (mixing_beta, mixing_type, conv_thr, ...).
            apply_strategy() only writes the keys that match the detected code.
        description: human-readable summary of what this strategy does
        cost_level: 1 (cheapest) to 5 (most expensive)
    """

    name: str
    param_changes: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    cost_level: int = 1


# Ordered cheapest -> most expensive. get_next_strategy walks this list
# and returns the first one not yet attempted.
STRATEGY_CHAIN: list[ConvergenceStrategy] = [
    ConvergenceStrategy(
        name="reduce_mixing_alpha",
        param_changes={
            # VASP: AMIX is the charge-density mixing amplitude
            "AMIX": 0.4,
            "BMIX": 0.0001,
            # QE: mixing_beta is the same concept
            "mixing_beta": 0.4,
        },
        description="Reduce SCF mixing amplitude to 0.4 for more conservative updates",
        cost_level=1,
    ),
    ConvergenceStrategy(
        name="change_algo",
        param_changes={
            # VASP: Normal uses conjugate-gradient, more robust than Fast
            "ALGO": "Normal",
            "IALGO": 38,
            # QE: plain mixing is the simplest/most stable
            "mixing_type": "plain",
        },
        description="Switch to a more stable SCF algorithm (Normal / plain mixing)",
        cost_level=2,
    ),
    ConvergenceStrategy(
        name="increase_nelm",
        param_changes={
            # VASP
            "NELM": 100,
            # QE: electron_maxstep is the equivalent
            "electron_maxstep": 100,
        },
        description="Increase max electronic SCF steps to 100",
        cost_level=3,
    ),
    ConvergenceStrategy(
        name="tune_amix_bmix",
        param_changes={
            # VASP: aggressive mixing parameter tuning
            "AMIX": 0.2,
            "BMIX": 0.0001,
            "AMIX_MAG": 0.4,
            "BMIX_MAG": 0.0001,
            # QE: lower mixing_beta further + tighten diag threshold
            "mixing_beta": 0.2,
            "diago_thr_init": 1e-6,
        },
        description="Aggressive mixing tuning: reduce AMIX to 0.2 and tighten diagonalization",
        cost_level=4,
    ),
    ConvergenceStrategy(
        name="switch_to_exact",
        param_changes={
            # VASP: exact diagonalization is the fallback of last resort
            "ALGO": "Exact",
            "IALGO": 48,
            "NELM": 200,
            "AMIX": 0.1,
            "BMIX": 0.0001,
            # QE: everything conservative
            "mixing_beta": 0.1,
            "mixing_type": "plain",
            "conv_thr": 1e-8,
            "electron_maxstep": 200,
        },
        description="Last resort: exact diagonalization with very conservative mixing",
        cost_level=5,
    ),
]


# VASP INCAR tags are uppercase; QE control keywords are lowercase.
# We use this to detect which code a params dict belongs to so
# apply_strategy only writes the relevant keys.
_VASP_HINTS = frozenset({"ALGO", "NELM", "AMIX", "BMIX", "IALGO", "ENCUT", "ISMEAR", "EDIFF"})
_QE_HINTS = frozenset({"mixing_beta", "mixing_type", "conv_thr", "diago_thr_init", "ecutwfc", "electron_maxstep"})


def _detect_code(params: dict[str, Any]) -> str:
    """Figure out if *params* is a VASP or QE input set."""
    for k in params:
        if k in _VASP_HINTS:
            return "vasp"
        if k in _QE_HINTS:
            return "qe"
    # default to vasp — most common in this codebase
    return "vasp"


def get_next_strategy(
    current_params: dict[str, Any],
    attempted_strategies: list[str | ConvergenceStrategy],
) -> ConvergenceStrategy | None:
    """Return the cheapest strategy that hasn't been attempted yet.

    Args:
        current_params: the active INCAR / QE input dict (used only for code
            detection — the strategy chain is the same regardless of code).
        attempted_strategies: names (or the objects themselves) that were
            already tried.

    Returns:
        The next ConvergenceStrategy, or None if all have been exhausted.
    """
    # ponytail: current_params is accepted for API symmetry with apply_strategy
    # and future per-code strategy filtering. Right now the chain is universal.
    _ = current_params

    attempted_names: set[str] = set()
    for s in attempted_strategies:
        if isinstance(s, ConvergenceStrategy):
            attempted_names.add(s.name)
        else:
            attempted_names.add(str(s))

    for strategy in STRATEGY_CHAIN:
        if strategy.name not in attempted_names:
            return strategy
    return None


def apply_strategy(
    params: dict[str, Any], strategy: ConvergenceStrategy
) -> dict[str, Any]:
    """Merge *strategy.param_changes* into *params*, writing only the keys
    that match the detected DFT code (VASP uppercase / QE lowercase).

    The dict is modified in-place and also returned for convenience.
    """
    code = _detect_code(params)
    for key, val in strategy.param_changes.items():
        if code == "vasp" and key.isupper():
            params[key] = val
        elif code == "qe" and not key.isupper():
            params[key] = val
    return params
