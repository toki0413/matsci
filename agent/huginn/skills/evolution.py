"""Bayesian skill evolution layer — learn tool config beliefs from past trajectories.

Reads trajectory JSON files saved by telemetry, extracts per-tool parameter
outcomes, and maintains Bayesian beliefs (Beta distribution) about which
parameter combinations work. The beliefs are injected as context into future
agent runs so the agent starts with accumulated experience instead of zero.

Trajectory format (from huginn.telemetry.save_trajectory):
    {
        "tool_calls": [
            {"tool": "vasp_tool", "args": {"action": "relax", "encut": 520}, "success": true},
            ...
        ]
    }

Bayesian model: Beta(α, β) prior with α=β=1 (uniform). After observing
s successes and f failures: α=1+s, β=1+f. Posterior mean = α/(α+β).
This is the standard Bernoulli-Beta conjugate update.

Neuroscience-inspired extensions (ANCCR + "Dopamine takes a hit"):
  - Time-weighted updates: learning rate proportional to inter-observation
    interval (ANCCR's IRI finding). Longer gaps → stronger belief updates.
  - Multi-dimensional feedback: tracks avg duration and info gain alongside
    success/failure, reflecting dopamine's role beyond reward signaling.
  - Forgetting factor: decay_all() pulls beliefs toward prior to prevent
    path lock-in (ANCCR's addiction insight — 100% backward associations
    are hard to break without active forgetting).
  - UCB exploration bonus in recommend_params() balances exploitation vs
    exploration of under-sampled parameter values.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Parameters worth tracking per tool call
_TRACKED_PARAMS = frozenset(
    {"action", "encut", "ediff", "kpoints", "functional", "basis_set",
     "method", "timestep", "temperature", "prec", "xc", "ecutwfc",
     "max_scf", "mixing_beta", "pseudo_potential"}
)

# Minimum observations before a belief shows up in context (avoids noise)
_MIN_SAMPLES = 2

# ANCCR: below this interval (seconds), observations are "rapid succession"
# and get standard weight=1.0. Above it, weight increases with gap length.
_IRI_BASELINE = 10.0
_IRI_SCALE = 300.0  # ~5 min — weight reaches ~1.3x at this gap
_IRI_CAP = 2.0      # max additional weight (so weight ∈ [1.0, 3.0])


@dataclass
class ToolBelief:
    """Beta(α, β) belief about one (tool, param, value) combination."""

    tool_name: str
    param_key: str
    param_value: str
    # raw integer counts — kept for backward compat and display
    successes: int = 0
    failures: int = 0
    last_updated: float = 0.0
    # ANCCR: time-weighted learning — inter-observation interval modulates update strength
    weighted_success: float = 0.0
    weighted_failure: float = 0.0
    last_obs_time: float = 0.0
    # multi-dimensional signal — dopamine encodes more than just reward
    avg_duration: float = 0.0    # EMA of tool call duration (seconds)
    avg_info_gain: float = 0.0  # EMA of output info content (key count)
    n_obs: int = 0               # total observations (for EMA weighting)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        """Time-weighted Beta posterior mean — expected success probability."""
        ws = self.weighted_success
        wf = self.weighted_failure
        return (1 + ws) / (2 + ws + wf)

    @property
    def confidence(self) -> float:
        """How much we trust this belief. Saturates at 10 samples."""
        return min(1.0, self.total / 10.0)

    def update(
        self,
        success: bool,
        *,
        timestamp: float | None = None,
        duration: float = 0.0,
        info_gain: float = 0.0,
    ) -> None:
        """Update belief with a single observation.

        Weight scales with inter-observation interval (ANCCR):
        rapid successive calls get weight=1.0, long gaps get up to 3.0x.
        """
        now = timestamp if timestamp is not None else time.time()

        # raw counts always +1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # ANCCR: longer gap since last obs → stronger update
        if self.last_obs_time > 0:
            dt = now - self.last_obs_time
            weight = 1.0 + min(
                _IRI_CAP,
                0.3 * max(0.0, (dt - _IRI_BASELINE) / _IRI_SCALE),
            )
        else:
            weight = 1.0

        if success:
            self.weighted_success += weight
        else:
            self.weighted_failure += weight

        # EMA for process metrics
        self.n_obs += 1
        a = 2.0 / (self.n_obs + 1)
        self.avg_duration = self.avg_duration * (1 - a) + duration * a
        self.avg_info_gain = self.avg_info_gain * (1 - a) + info_gain * a

        self.last_obs_time = now
        self.last_updated = now

    def decay(self, gamma: float = 0.99) -> None:
        """Forgetting factor — pull weighted belief toward prior (0.5).

        Prevents path lock-in: without this, a belief with 100 successes
        and 0 failures becomes nearly impossible to change even if the
        environment has shifted. Mirrors ANCCR's insight that addiction
        stems from 100% backward associations that resist extinction.
        """
        self.weighted_success *= gamma
        self.weighted_failure *= gamma

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["posterior_mean"] = self.posterior_mean
        d["confidence"] = self.confidence
        return d


class SkillEvolutionLayer:
    """Accumulates tool-parameter beliefs from trajectory history.

    Singleton — one set of beliefs per process. Beliefs are persisted to
    a JSON file so they survive restarts.

    Usage:
        layer = SkillEvolutionLayer.shared()
        layer.update_from_directory(workspace / ".huginn" / "trajectories")
        ctx = layer.get_skill_context()  # inject into agent prompt
    """

    _instance: SkillEvolutionLayer | None = None

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._beliefs: dict[tuple[str, str, str], ToolBelief] = {}
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        if self._persist_path is not None:
            self._load()

    @classmethod
    def shared(cls) -> SkillEvolutionLayer:
        if cls._instance is None:
            cache_dir = os.environ.get("HUGINN_CACHE_DIR", "")
            if cache_dir:
                path = Path(cache_dir) / "skill_beliefs.json"
            else:
                path = Path.home() / ".huginn" / "skill_beliefs.json"
            cls._instance = cls(persist_path=path)
        return cls._instance

    # ── Update ───────────────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        *,
        duration: float = 0.0,
        info_gain: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        """Record a single tool call outcome, updating all matching beliefs.

        duration: wall-clock seconds the tool call took.
        info_gain: rough measure of output information content (key count).
        timestamp: explicit time for the observation (defaults to now). Mainly
            useful for trajectory replay where the original time is known.
        """
        for key in _TRACKED_PARAMS:
            val = args.get(key)
            if val is None:
                continue
            val_str = str(val)
            bkey = (tool_name, key, val_str)
            belief = self._beliefs.get(bkey)
            if belief is None:
                belief = ToolBelief(tool_name, key, val_str)
                self._beliefs[bkey] = belief
            belief.update(
                success,
                timestamp=timestamp,
                duration=duration,
                info_gain=info_gain,
            )

    def update_from_trajectory(self, traj_path: str | Path) -> int:
        """Read one trajectory JSON, update beliefs. Returns tool call count."""
        from huginn.telemetry import load_trajectory

        data = load_trajectory(traj_path)
        count = 0
        for tc in data.get("tool_calls", []):
            tool = tc.get("tool", "")
            if not tool:
                continue
            args = tc.get("args")
            if not isinstance(args, dict):
                args = {}
            success = tc.get("success", True)
            duration = tc.get("duration", 0.0)
            info_gain = tc.get("info_gain", 0.0)
            ts = tc.get("timestamp")
            self.record_tool_call(
                tool, args, success,
                duration=duration,
                info_gain=info_gain,
                timestamp=ts,
            )
            count += 1
        return count

    def update_from_directory(self, dir_path: str | Path) -> int:
        """Scan all trajectory JSON files in a directory. Returns total tool calls."""
        total = 0
        p = Path(dir_path)
        if not p.is_dir():
            return 0
        for f in sorted(p.glob("*.json")):
            try:
                total += self.update_from_trajectory(f)
            except Exception:
                logger.debug("skip trajectory %s", f, exc_info=True)
        if total > 0:
            self._save()
            logger.info("SkillEvolutionLayer: learned from %d tool calls", total)
        return total

    def decay_all(self, gamma: float = 0.95) -> None:
        """Apply forgetting factor to all beliefs — pull toward prior.

        Call periodically (e.g. once per session) to prevent any single
        parameter value from becoming permanently dominant. Without this,
        early successes can lock the agent into a suboptimal path forever.
        """
        for b in self._beliefs.values():
            b.decay(gamma)
        if self._beliefs:
            self._save()

    # ── Query ─────────────────────────────────────────────────────

    def get_belief(
        self, tool_name: str, param_key: str, param_value: str
    ) -> ToolBelief | None:
        return self._beliefs.get((tool_name, param_key, param_value))

    def get_tool_beliefs(self, tool_name: str) -> list[ToolBelief]:
        return [b for k, b in self._beliefs.items() if k[0] == tool_name]

    def recommend_params(
        self, tool_name: str, param_key: str, *, exploration: float = 0.3
    ) -> list[tuple[str, float, int]]:
        """Ranked recommendations with UCB exploration bonus.

        Returns (value, posterior_mean, sample_count) tuples.
        Sorting uses UCB score = posterior_mean + C*sqrt(ln(N)/n_i),
        so under-explored values get a small ranking boost.
        """
        matching = [
            b for b in self._beliefs.values()
            if b.tool_name == tool_name
            and b.param_key == param_key
            and b.total > 0
        ]
        if not matching:
            return []
        n_total = sum(b.total for b in matching)
        scored = []
        for b in matching:
            bonus = exploration * math.sqrt(math.log(n_total + 1) / (b.total + 1))
            ucb = b.posterior_mean + bonus
            scored.append((b.param_value, b.posterior_mean, b.total, ucb))
        scored.sort(key=lambda x: -x[3])  # sort by UCB
        return [(v, p, n) for v, p, n, _ in scored]

    def get_skill_context(self, tool_name: str | None = None) -> str:
        """Generate context string for injection into agent prompts.

        Only includes beliefs with enough samples to be meaningful.
        Sorted by sample count (most evidence first).
        """
        beliefs = list(self._beliefs.values())
        if tool_name is not None:
            beliefs = [b for b in beliefs if b.tool_name == tool_name]
        beliefs = [b for b in beliefs if b.total >= _MIN_SAMPLES]
        if not beliefs:
            return ""
        beliefs.sort(key=lambda b: -b.total)
        lines = ["### Skill Evolution (learned from past trajectories):"]
        for b in beliefs[:15]:
            extra = ""
            if b.avg_duration > 0:
                extra = f" [{b.avg_duration:.1f}s, {b.avg_info_gain:.0f}k]"
            lines.append(
                f"  {b.tool_name}.{b.param_key}={b.param_value}: "
                f"{b.successes}/{b.total} success "
                f"(P={b.posterior_mean:.0%}, conf={b.confidence:.0%}){extra}"
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "total_beliefs": len(self._beliefs),
            "tools": sorted({k[0] for k in self._beliefs}),
            "params_tracked": sorted(_TRACKED_PARAMS),
            "top_beliefs": [
                b.to_dict()
                for b in sorted(self._beliefs.values(), key=lambda x: -x.total)[:5]
            ],
        }

    # ── Persistence ───────────────────────────────────────────────

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "2.0",
                "saved_at": time.time(),
                "beliefs": [b.to_dict() for b in self._beliefs.values()],
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("belief persist failed", exc_info=True)

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for b_dict in data.get("beliefs", []):
                key = (b_dict["tool_name"], b_dict["param_key"], b_dict["param_value"])
                s = b_dict.get("successes", 0)
                f = b_dict.get("failures", 0)
                self._beliefs[key] = ToolBelief(
                    tool_name=b_dict["tool_name"],
                    param_key=b_dict["param_key"],
                    param_value=b_dict["param_value"],
                    successes=s,
                    failures=f,
                    last_updated=b_dict.get("last_updated", 0.0),
                    # v2 fields — old v1 files fall back to raw counts
                    weighted_success=b_dict.get("weighted_success", float(s)),
                    weighted_failure=b_dict.get("weighted_failure", float(f)),
                    last_obs_time=b_dict.get("last_obs_time", 0.0),
                    avg_duration=b_dict.get("avg_duration", 0.0),
                    avg_info_gain=b_dict.get("avg_info_gain", 0.0),
                    n_obs=b_dict.get("n_obs", 0),
                )
            if self._beliefs:
                logger.info(
                    "SkillEvolutionLayer: loaded %d beliefs from %s",
                    len(self._beliefs), self._persist_path,
                )
        except Exception:
            logger.debug("belief load failed", exc_info=True)

    def clear(self) -> None:
        self._beliefs.clear()
