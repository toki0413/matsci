"""H2: Workflow Evolutionary Search — bandit + archive + novelty.

对同一 objective 跑多个 workflow variant, Thompson sampling 选, Pareto 归档.
跟 H1 平行, 独立 store 不走 EvolutionEngine.

数学:
- WorkflowBelief: Beta(α, β), key = (objective_hash, variant_id). 复用
  ToolBelief (skills/evolution.py:62) 的 ANCCR 时间加权, key schema 不同.
- select_variant: Thompson sampling via random.betavariate(1+ws, 1+wf).
- VariantArchive: ADAS 式归档, top-K Pareto (fitness = r_phys, efficiency, novelty).
- compute_novelty: 参数级 Jaccard diff (P6: TF-IDF 对 encut 520 vs 540 不敏感).

toggle: cfg.feature_flags.harness_workflow_evolution (默认 off).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IRI_CAP = 2.0
_IRI_BASELINE = 10.0
_IRI_SCALE = 300.0
_PARETO_K = 5


def _harness_enabled(key: str, default: bool = False) -> bool:
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get(key, default))
    except Exception:
        return default


def _objective_hash(objective: str) -> str:
    return hashlib.md5(objective.encode("utf-8")).hexdigest()[:12]


@dataclass
class WorkflowBelief:
    """一个 (objective_hash, variant_id) 的 Beta 信念.

    复用 ToolBelief 的 ANCCR 数学, key schema 不同 — variant 级不是
    (tool, param, value) 级. fitness 多目标供 Pareto 比较.
    """
    variant_id: str
    objective_hash: str
    successes: int = 0
    failures: int = 0
    weighted_success: float = 0.0
    weighted_failure: float = 0.0
    last_obs_time: float = 0.0
    last_updated: float = 0.0
    last_r_phys: float = 0.0
    last_efficiency: float = 0.0
    last_novelty: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        ws = self.weighted_success
        wf = self.weighted_failure
        return (1 + ws) / (2 + ws + wf)

    def update(
        self,
        success: bool,
        *,
        r_phys: float = 0.0,
        efficiency: float = 0.0,
        novelty: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        now = timestamp if timestamp is not None else time.time()
        if success:
            self.successes += 1
        else:
            self.failures += 1
        if self.last_obs_time > 0:
            dt = now - self.last_obs_time
            weight = 1.0 + min(
                _IRI_CAP, 0.3 * max(0.0, (dt - _IRI_BASELINE) / _IRI_SCALE)
            )
        else:
            weight = 1.0
        if success:
            self.weighted_success += weight
        else:
            self.weighted_failure += weight
        self.last_r_phys = float(r_phys)
        self.last_efficiency = float(efficiency)
        self.last_novelty = float(novelty)
        self.last_obs_time = now
        self.last_updated = now

    def decay(self, gamma: float = 0.99) -> None:
        self.weighted_success *= gamma
        self.weighted_failure *= gamma

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["posterior_mean"] = self.posterior_mean
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowBelief":
        return cls(
            variant_id=d["variant_id"],
            objective_hash=d["objective_hash"],
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            weighted_success=d.get("weighted_success", 0.0),
            weighted_failure=d.get("weighted_failure", 0.0),
            last_obs_time=d.get("last_obs_time", 0.0),
            last_updated=d.get("last_updated", 0.0),
            last_r_phys=d.get("last_r_phys", 0.0),
            last_efficiency=d.get("last_efficiency", 0.0),
            last_novelty=d.get("last_novelty", 0.0),
            created_at=d.get("created_at", time.time()),
        )


class WorkflowBandit:
    """Thompson sampling bandit. 冷启动随机选, 不断 _MIN_SAMPLES 门槛."""
    _instance: "WorkflowBandit | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("HUGINN_CACHE_DIR", Path.home() / ".huginn")
        )
        self._store_dir = cache_dir / "workflow_beliefs"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("workflow_beliefs dir create failed", exc_info=True)
        self._beliefs: dict[tuple[str, str], WorkflowBelief] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> "WorkflowBandit":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_all(self) -> None:
        try:
            for f in self._store_dir.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    b = WorkflowBelief.from_dict(d)
                    self._beliefs[(b.objective_hash, b.variant_id)] = b
                except Exception:
                    logger.debug("belief load fail: %s", f, exc_info=True)
        except Exception:
            logger.debug("belief dir scan fail", exc_info=True)

    def _save(self, b: WorkflowBelief) -> None:
        try:
            f = self._store_dir / f"{b.objective_hash}_{b.variant_id}.json"
            f.write_text(
                json.dumps(b.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("belief save fail: %s", b.variant_id, exc_info=True)

    def record_variant_outcome(
        self,
        variant_id: str,
        objective_hash: str,
        success: bool,
        *,
        r_phys: float = 0.0,
        efficiency: float = 0.0,
        novelty: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        key = (objective_hash, variant_id)
        with self._lock:
            b = self._beliefs.get(key)
            if b is None:
                b = WorkflowBelief(
                    variant_id=variant_id, objective_hash=objective_hash
                )
                self._beliefs[key] = b
            b.update(
                success,
                r_phys=r_phys,
                efficiency=efficiency,
                novelty=novelty,
                timestamp=timestamp,
            )
        self._save(b)

    def select_variant(
        self,
        candidates: list[str],
        objective_hash: str,
        exploration: float = 0.3,
    ) -> str | None:
        """Thompson sampling: 每个候选采 Beta(1+ws, 1+wf), 取 max."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        with self._lock:
            samples: list[tuple[str, float]] = []
            for vid in candidates:
                b = self._beliefs.get((objective_hash, vid))
                if b is None or b.total == 0:
                    samples.append((vid, random.random()))
                else:
                    alpha = 1 + b.weighted_success
                    beta = 1 + b.weighted_failure
                    try:
                        s = random.betavariate(alpha, beta)
                    except Exception:
                        s = b.posterior_mean
                    samples.append((vid, s))
        return max(samples, key=lambda x: x[1])[0]

    def get_belief(
        self, variant_id: str, objective_hash: str
    ) -> WorkflowBelief | None:
        with self._lock:
            return self._beliefs.get((objective_hash, variant_id))

    def list_beliefs(
        self, objective_hash: str | None = None
    ) -> list[WorkflowBelief]:
        with self._lock:
            bs = list(self._beliefs.values())
        if objective_hash:
            bs = [b for b in bs if b.objective_hash == objective_hash]
        bs.sort(key=lambda b: b.posterior_mean, reverse=True)
        return bs


class VariantArchive:
    """ADAS 式归档: 存 variant script + fitness, 维护 Pareto 前沿.

    存 .huginn/workflow_archive/<objective_hash>.json
    fitness = [r_phys, efficiency, novelty] (越大越好)
    """
    _instance: "VariantArchive | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("HUGINN_CACHE_DIR", Path.home() / ".huginn")
        )
        self._store_dir = cache_dir / "workflow_archive"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("workflow_archive dir create failed", exc_info=True)

    @classmethod
    def get_instance(cls) -> "VariantArchive":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _archive_path(self, objective_hash: str) -> Path:
        return self._store_dir / f"{objective_hash}.json"

    def _load(self, objective_hash: str) -> dict[str, Any]:
        f = self._archive_path(objective_hash)
        if not f.exists():
            return {"variants": []}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("archive load fail: %s", f, exc_info=True)
            return {"variants": []}

    def _save(self, objective_hash: str, data: dict[str, Any]) -> None:
        f = self._archive_path(objective_hash)
        try:
            f.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("archive save fail: %s", f, exc_info=True)

    def add_variant(
        self,
        objective_hash: str,
        objective: str,
        variant_id: str,
        script_dict: dict[str, Any],
        fitness: list[float],
        alpha: int = 1,
        beta: int = 1,
    ) -> None:
        with self._lock:
            data = self._load(objective_hash)
            data.setdefault("objective", objective)
            variants = data.setdefault("variants", [])
            variants = [v for v in variants if v.get("variant_id") != variant_id]
            variants.append({
                "variant_id": variant_id,
                "script_dict": script_dict,
                "fitness": list(fitness),
                "alpha": alpha,
                "beta": beta,
                "created_at": time.time(),
            })
            variants = self._pareto_front(variants, k=_PARETO_K)
            data["variants"] = variants
            self._save(objective_hash, data)

    def _pareto_front(
        self, variants: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        if len(variants) <= k:
            return variants
        non_dominated: list[dict[str, Any]] = []
        for v in variants:
            fv = v.get("fitness", [0, 0, 0])
            dominated = False
            for u in variants:
                if u is v:
                    continue
                fu = u.get("fitness", [0, 0, 0])
                if all(x >= y for x, y in zip(fu, fv)) and any(
                    x > y for x, y in zip(fu, fv)
                ):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(v)
        if len(non_dominated) > k:
            non_dominated.sort(
                key=lambda v: v.get("fitness", [0])[0], reverse=True
            )
            non_dominated = non_dominated[:k]
        return non_dominated if non_dominated else variants[:k]

    def list_variants(self, objective_hash: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load(objective_hash)
            return list(data.get("variants", []))

    def get_variant(
        self, objective_hash: str, variant_id: str
    ) -> dict[str, Any] | None:
        for v in self.list_variants(objective_hash):
            if v.get("variant_id") == variant_id:
                return v
        return None


def _flatten_script_args(script_dict: dict[str, Any]) -> set[tuple[str, str, str]]:
    """把 script subtasks.args 拍平成 (tool, param, value) 集合. 参数级 diff 用."""
    flat: set[tuple[str, str, str]] = set()
    for st in script_dict.get("subtasks", []):
        if not isinstance(st, dict):
            continue
        tool = str(st.get("tool", st.get("tool_name", "")))
        args = st.get("args", {})
        if not isinstance(args, dict):
            continue
        for k, v in args.items():
            flat.add((tool, str(k), str(v)))
    return flat


def compute_novelty(
    new_script: dict[str, Any],
    archive_variants: list[dict[str, Any]],
) -> float:
    """参数级 Jaccard novelty: 1 - max(overlap) over archive.

    ponytail: 不用 _compute_semantic_overlap (context_builder.py:72),
    BoW TF-IDF cosine 对 encut 520 vs 540 ≈1.0 不敏感.
    """
    new_flat = _flatten_script_args(new_script)
    if not new_flat:
        return 0.0
    if not archive_variants:
        return 1.0
    max_overlap = 0.0
    for v in archive_variants:
        old_flat = _flatten_script_args(v.get("script_dict", {}))
        if not old_flat:
            continue
        inter = len(new_flat & old_flat)
        union = len(new_flat | old_flat)
        if union > 0:
            j = inter / union
            if j > max_overlap:
                max_overlap = j
    return 1.0 - max_overlap


def _selfcheck() -> None:
    """H2 selfcheck: bandit + archive + novelty."""
    import shutil
    import tempfile

    import huginn.autoloop.bandit as bd

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    bd.WorkflowBandit._instance = None
    bd.VariantArchive._instance = None

    bandit = bd.WorkflowBandit.get_instance()
    candidates = ["v1", "v2", "v3"]
    chosen = bandit.select_variant(candidates, "objhash_test")
    assert chosen in candidates, f"cold start failed: {chosen}"
    print(f"1. cold start select_variant OK (chose {chosen})")

    bandit.record_variant_outcome("v1", "objhash_test", success=True, r_phys=0.8)
    bandit.record_variant_outcome("v1", "objhash_test", success=True, r_phys=0.9)
    bandit.record_variant_outcome("v2", "objhash_test", success=False, r_phys=0.1)
    bandit.record_variant_outcome("v3", "objhash_test", success=False, r_phys=0.2)
    b1 = bandit.get_belief("v1", "objhash_test")
    b2 = bandit.get_belief("v2", "objhash_test")
    assert b1 and b2 and b1.posterior_mean > b2.posterior_mean
    print(f"2. record + posterior: v1={b1.posterior_mean:.2f} > v2={b2.posterior_mean:.2f} OK")

    counts = {"v1": 0, "v2": 0, "v3": 0}
    for _ in range(1000):
        c = bandit.select_variant(candidates, "objhash_test")
        counts[c] += 1
    assert counts["v1"] > counts["v2"] and counts["v1"] > counts["v3"]
    print(f"3. Thompson 1000x: {counts} OK")

    archive = bd.VariantArchive.get_instance()
    sa = {"subtasks": [{"tool": "vasp", "args": {"encut": 520, "kpoints": "2 2 2"}}]}
    sb = {"subtasks": [{"tool": "vasp", "args": {"encut": 540, "kpoints": "3 3 3"}}]}
    sc = {"subtasks": [{"tool": "vasp", "args": {"encut": 520, "kpoints": "2 2 2"}}]}
    archive.add_variant("objhash_test", "test", "va", sa, [0.8, 0.9, 0.7])
    archive.add_variant("objhash_test", "test", "vb", sb, [0.7, 0.8, 1.0])
    archive.add_variant("objhash_test", "test", "vc", sc, [0.8, 0.9, 0.7])
    vs = archive.list_variants("objhash_test")
    assert len(vs) >= 2, f"archive should have >=2: {len(vs)}"
    print(f"4. VariantArchive add+list OK ({len(vs)} variants)")

    n_same = bd.compute_novelty(sc, vs)
    assert n_same == 0.0, f"identical novelty should be 0: {n_same}"
    n_new = bd.compute_novelty(
        {"subtasks": [{"tool": "vasp", "args": {"encut": 600, "kpoints": "4 4 4"}}]},
        vs,
    )
    assert n_new > 0.5, f"novel should be high: {n_new}"
    print(f"5. compute_novelty: same={n_same:.2f}, new={n_new:.2f} OK")

    bd.WorkflowBandit._instance = None
    b2r = bd.WorkflowBandit.get_instance()
    b1r = b2r.get_belief("v1", "objhash_test")
    assert b1r and b1r.successes == 2, f"persisted wrong: {b1r}"
    print(f"6. persistence reload OK (v1 successes={b1r.successes})")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nH2 bandit selfcheck OK (6/6)")


if __name__ == "__main__":
    _selfcheck()
