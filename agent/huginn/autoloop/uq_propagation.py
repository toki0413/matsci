"""Uncertainty propagation along the tool execution chain.

When tool A produces output with uncertainty sigma_A, and tool B takes A's
output as input, B's output uncertainty should account for sigma_A.

Methods:
1. Linear propagation: sigma_B = sqrt((dB/dA * sigma_A)^2 + sigma_B_intrinsic^2)
2. Monte Carlo: sample A's output from its distribution, run B on each sample,
   measure output spread
3. Simple bound: sigma_B = |dB/dA| * sigma_A + sigma_B_intrinsic (worst case)

Only the linear and Monte Carlo paths are wired up here — the worst-case bound
is a one-liner callers can reach via ``linear_propagate`` with intrinsic sigma
treated additively if they really want it. Kept stdlib-only (math / random /
statistics) so the autoloop doesn't pay a numpy import on every iteration.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = [
    "UQState",
    "UQChain",
    "ChainTracker",
    "linear_propagate",
    "monte_carlo_propagate",
]


@dataclass
class UQState:
    """A single value tagged with its uncertainty and where it came from.

    sigma == 0 means "treat as exact". Negative sigma is nonsense and gets
    clamped to 0 on construction so downstream sqrt/normal calls stay sane.
    """

    value: float
    sigma: float = 0.0
    source: str = ""
    method: str = "exact"

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            self.sigma = 0.0


def _resolve_sensitivity(sensitivity: float | Callable[[float], float],
                         value: float) -> float:
    """sensitivity may be a scalar derivative or a callable dB/dA(x).

    Callables are evaluated at the input value so callers can pass a real
    local sensitivity function instead of pre-computing the slope.
    """
    if callable(sensitivity):
        return float(sensitivity(value))
    return float(sensitivity)


def linear_propagate(
    input_uq: UQState,
    sensitivity_fn: float | Callable[[float], float],
    output_value: float,
    intrinsic_sigma: float = 0.0,
) -> UQState:
    """First-order (GUM) uncertainty propagation.

    sigma_out = sqrt((dB/dA * sigma_in)^2 + intrinsic_sigma^2)

    ``sensitivity_fn`` is the local derivative dB/dA — either a number or a
    callable evaluated at ``input_uq.value``. ``intrinsic_sigma`` is tool B's
    own noise floor, independent of A's uncertainty.
    """
    s = _resolve_sensitivity(sensitivity_fn, input_uq.value)
    propagated = (s * input_uq.sigma) ** 2
    intrinsic = float(intrinsic_sigma) ** 2
    sigma_out = math.sqrt(propagated + intrinsic)
    return UQState(
        value=float(output_value),
        sigma=sigma_out,
        source=input_uq.source,
        method="linear",
    )


def monte_carlo_propagate(
    input_uq: UQState,
    fn: Callable[[float], float],
    n_samples: int = 100,
) -> UQState:
    """Propagate uncertainty by sampling.

    Draws ``n_samples`` from Normal(value, sigma), pushes each through ``fn``,
    and reports the mean of the outputs as the value and their (population)
    standard deviation as sigma. With sigma_in == 0 every sample is identical,
    so sigma_out collapses to 0 — that's the constant-function case.

    Uses a private Random instance so it never disturbs the global RNG state.
    """
    n = max(1, int(n_samples))
    rng = random.Random()
    mu = float(input_uq.value)
    sd = float(input_uq.sigma)

    if sd == 0.0:
        # degenerate distribution — skip the sampling loop, result is exact
        out = float(fn(mu))
        return UQState(value=out, sigma=0.0, source=input_uq.source,
                      method="monte_carlo")

    outputs = [float(fn(rng.gauss(mu, sd))) for _ in range(n)]
    value_out = statistics.fmean(outputs)
    sigma_out = statistics.pstdev(outputs)
    return UQState(
        value=value_out,
        sigma=sigma_out,
        source=input_uq.source,
        method="monte_carlo",
    )


class UQChain:
    """Ordered list of UQStates produced as one tool feeds the next.

    ``propagate`` is the convenience entry point: give it the upstream UQState,
    the sensitivity of the current tool, and the tool's nominal output, and it
    appends a new linearly-propagated UQState. The chain's cumulative uncertainty
    is just the last state's sigma — each link already folded in everything
    upstream, so there's nothing extra to combine.
    """

    def __init__(self) -> None:
        self.states: list[UQState] = []

    def propagate(
        self,
        input_uq: UQState,
        sensitivity: float | Callable[[float], float],
        output_value: float,
        intrinsic_sigma: float = 0.0,
    ) -> UQState:
        out = linear_propagate(input_uq, sensitivity, output_value, intrinsic_sigma)
        self.states.append(out)
        return out

    def add(self, state: UQState) -> None:
        """Push a pre-built state (e.g. a Monte Carlo result) onto the chain."""
        self.states.append(state)

    @property
    def cumulative_sigma(self) -> float:
        """Uncertainty at the end of the chain. 0 for an empty chain."""
        return self.states[-1].sigma if self.states else 0.0

    def summary(self) -> dict:
        return {
            "n_steps": len(self.states),
            "states": [
                {"value": s.value, "sigma": s.sigma, "source": s.source,
                 "method": s.method}
                for s in self.states
            ],
            "cumulative_sigma": self.cumulative_sigma,
        }


class ChainTracker:
    """Accumulates (tool_name, input_uq, output_uq) across a run.

    The autoloop calls ``add`` after each tool fires. ``summary`` rebuilds the
    chain of output states so the caller can see how uncertainty grew (or
    didn't) from the first tool to the last.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, UQState, UQState]] = []

    def add(self, tool_name: str, input_uq: UQState, output_uq: UQState) -> None:
        self._entries.append((tool_name, input_uq, output_uq))

    @property
    def steps(self) -> Sequence[tuple[str, UQState, UQState]]:
        return self._entries

    def summary(self) -> UQChain:
        chain = UQChain()
        for tool_name, _inp, out in self._entries:
            chain.add(UQState(
                value=out.value,
                sigma=out.sigma,
                source=tool_name,
                method=out.method,
            ))
        return chain
