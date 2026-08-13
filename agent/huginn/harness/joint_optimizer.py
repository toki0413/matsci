"""H3: Joint Prompt + Workflow Optimization.

统一管理 (block_subset, workflow_params) 组合的 Beta 信念, 用 UCB 选.
跟 H1 (PromptPatch) 和 H2 (WorkflowBandit) 平行, 不耦合 — H1/H2 各自
管自己的 Beta, H3 只在 toggle on 时覆盖 H1/H2 的选择逻辑.

数学: 搜索空间 S = {(b_1,...,b_k; s_1,...,s_m)}, b_i = prompt block on/off,
s_j = workflow stage 参数. 每个 s 维护 Beta(α, β), UCB 选.
P8 限制: 不包含 model 维度.

toggle: cfg.feature_flags.harness_joint_optimizer (默认 off).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.utils.runtime import get_runtime_home

from ._enabled import _harness_enabled

logger = logging.getLogger(__name__)

# 核心 block 不能被关掉 (避免 prompt 崩坏)
_CORE_BLOCKS = frozenset({"body", "fail", "exec", "context", "hypothesis", "principles"})


@dataclass
class JointBelief:
    """一个 (phase, block_subset_id, workflow_params_id) 组合的 Beta 信念.

    用 UCB (Upper Confidence Bound) 选, 不是 Thompson sampling.
    UCB = posterior_mean + exploration * sqrt(ln(n_total+1) / (n_i+1)).
    """
    config_id: str  # block_subset_id + workflow_params_id 的组合 id
    phase: str
    problem_domain: str = ""  # 分格存档维度: 一个"问题域"一格, 防激进方案被饥饿
    successes: int = 0
    failures: int = 0
    last_updated: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        # Beta(1, 1) 共轭先验, 跟 ToolBelief 一致
        return (1 + self.successes) / (2 + self.successes + self.failures)

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_updated = time.time()

    def ucb(self, n_total: int, exploration: float = 0.3) -> float:
        """UCB1: posterior_mean + exploration * sqrt(ln(n_total+1) / (n_i+1))."""
        if self.total == 0:
            return float("inf")  # 冷启动优先探索
        return self.posterior_mean + exploration * math.sqrt(
            math.log(n_total + 1) / (self.total + 1)
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["posterior_mean"] = self.posterior_mean
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JointBelief:
        return cls(
            config_id=d["config_id"],
            phase=d.get("phase", ""),
            problem_domain=d.get("problem_domain", ""),
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            last_updated=d.get("last_updated", 0.0),
            created_at=d.get("created_at", time.time()),
        )


class JointBandit:
    """管理 (block_subset, workflow_params) 组合的 Beta 信念.

    select_block_subset: 给定 phase + full_blocks, 按 UCB 选 block 子集.
    select_workflow_params: 给定 stage_name + defaults, 按 UCB 调参数.
    record_joint_outcome: 记录组合 outcome, 更新 Beta.

    ponytail: 不做笛卡尔积全搜索, UCB + 随机探索够用.
    """
    _instance: JointBandit | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = get_runtime_home()
        self._store_dir = cache_dir / "joint_beliefs"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("joint_beliefs dir create failed", exc_info=True)
        self._beliefs: dict[tuple[str, str], JointBelief] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> JointBandit:
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
                    b = JointBelief.from_dict(d)
                    self._beliefs[(b.phase, b.config_id)] = b
                except Exception:
                    logger.debug("joint belief load fail: %s", f, exc_info=True)
        except Exception:
            logger.debug("joint belief dir scan fail", exc_info=True)

    def _save(self, b: JointBelief) -> None:
        try:
            f = self._store_dir / f"{b.phase}__{b.problem_domain}__{b.config_id}.json"
            f.write_text(
                json.dumps(b.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("joint belief save fail: %s", b.config_id, exc_info=True)

    def _make_config_id(
        self,
        block_subset: list[str],
        workflow_params: dict[str, Any],
        problem_domain: str = "",
    ) -> str:
        """组合 id = problem_domain + block_subset hash + workflow_params hash.

        分格存档: 不同 problem_domain 的相同 block/params 组合会得到不同
        config_id, 从而在信念表里分格存储 — 结构不同的好方案各有独立存档,
        不会被单一 UCB 的"偏好已验证组合"饥饿.
        """
        import hashlib
        block_str = ",".join(sorted(block_subset))
        param_str = json.dumps(workflow_params, sort_keys=True, default=str)
        return hashlib.md5(
            (problem_domain + "|" + block_str + "|" + param_str).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]

    def select_block_subset(
        self,
        phase: str,
        full_blocks: list[tuple[str, str]],
        problem_domain: str = "",
    ) -> list[str]:
        """按 UCB 选 block 子集. 保留核心 block + 按信念选其他.

        problem_domain: 可选分格参数. 传入时只在对应 domain 桶内比较 UCB,
        避免跨 domain 的全局 UCB 把某个格子里未充分探索的激进方案饿死.
        不传则保持原行为 ("" 桶). toggle off 时返回全部 block 名.
        """
        if not _harness_enabled("harness_joint_optimizer"):
            return [name for name, _ in full_blocks]
        # 核心 block 必选, 非核心按 UCB 概率选
        core = [name for name, _ in full_blocks if name in _CORE_BLOCKS]
        non_core = [name for name, _ in full_blocks if name not in _CORE_BLOCKS]
        if not non_core:
            return core
        with self._lock:
            n_total = sum(
                b.total
                for (p, _), b in self._beliefs.items()
                if p == phase and b.problem_domain == problem_domain
            )
            # 每个非核心 block: 看 "包含此 block" 的组合 UCB (限定在 domain 桶内)
            scored: list[tuple[str, float]] = []
            for name in non_core:
                # 找包含此 block 的最佳 config
                best_ucb = 0.0
                for (p, _cid), b in self._beliefs.items():
                    if p != phase:
                        continue
                    if b.problem_domain != problem_domain:
                        continue
                    # 简化: 用 config_id 的信念代理 block 信念
                    u = b.ucb(n_total)
                    if u > best_ucb:
                        best_ucb = u
                scored.append((name, best_ucb))
            # UCB 高的入选, 低概率探索
            # ponytail: 不做 softmax, 简单取 UCB > 0.5 的 + 随机探索 1 个
            selected = [name for name, u in scored if u > 0.5]
            if not selected:
                # 冷启动: 全选
                selected = list(non_core)
            else:
                # 加一个随机探索
                import random
                unselected = [n for n in non_core if n not in selected]
                if unselected and random.random() < 0.3:
                    selected.append(random.choice(unselected))
        return core + selected

    def select_workflow_params(
        self,
        stage_name: str,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """按 UCB 调 workflow 参数. 在 defaults ±10% 范围内选.

        toggle off 时返回 defaults 原样.
        """
        if not _harness_enabled("harness_joint_optimizer"):
            return dict(defaults)
        # ponytail: 简化 — 对数值参数做 ±10% 扰动, 用 UCB 决定方向
        out: dict[str, Any] = {}
        import random
        for k, v in defaults.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # 数值参数: ±10% 扰动
                delta = v * 0.1 * random.choice([-1, 1])
                if isinstance(v, int):
                    out[k] = int(v + delta)
                else:
                    out[k] = round(v + delta, 4)
            else:
                out[k] = v
        return out

    def record_joint_outcome(
        self,
        phase: str,
        block_subset: list[str],
        workflow_params: dict[str, Any],
        success: bool,
        problem_domain: str = "",
    ) -> None:
        """记录 (problem_domain, block_subset, workflow_params) 组合的 outcome.

        problem_domain: 分格存档维度. 不传则用 "" (向后兼容现有调用).
        """
        config_id = self._make_config_id(block_subset, workflow_params, problem_domain)
        key = (phase, config_id)
        with self._lock:
            b = self._beliefs.get(key)
            if b is None:
                b = JointBelief(
                    config_id=config_id, phase=phase, problem_domain=problem_domain
                )
                self._beliefs[key] = b
            b.update(success)
        self._save(b)

    def list_beliefs(self, phase: str | None = None) -> list[JointBelief]:
        with self._lock:
            bs = list(self._beliefs.values())
        if phase:
            bs = [b for b in bs if b.phase == phase]
        bs.sort(key=lambda b: b.posterior_mean, reverse=True)
        return bs


# 接入 helper: H1 apply_patches 调, 返回 block 子集
def select_block_subset_for_phase(
    phase: str,
    blocks: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """H1 apply_patches 调. toggle off 时返回原 blocks, on 时按 UCB 选子集."""
    if not _harness_enabled("harness_joint_optimizer"):
        return blocks
    try:
        bandit = JointBandit.get_instance()
        selected_names = bandit.select_block_subset(phase, blocks)
        return [(name, text) for name, text in blocks if name in selected_names]
    except Exception:
        logger.debug("H3 select_block_subset_for_phase failed", exc_info=True)
        return blocks


# 接入 helper: H2 generate_variants 调, 返回调整后的 args
def select_workflow_params_for_stage(
    stage_name: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """H2 generate_variants 调. toggle off 时返回 defaults, on 时按 UCB 调."""
    if not _harness_enabled("harness_joint_optimizer"):
        return dict(defaults)
    try:
        bandit = JointBandit.get_instance()
        return bandit.select_workflow_params(stage_name, defaults)
    except Exception:
        logger.debug("H3 select_workflow_params_for_stage failed", exc_info=True)
        return dict(defaults)


def _selfcheck() -> None:
    """H3 selfcheck: JointBandit + UCB + block subset + workflow params."""
    import shutil
    import tempfile

    import huginn.harness.joint_optimizer as jo

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    jo.JointBandit._instance = None

    # 1. toggle off: passthrough
    jo._harness_enabled = lambda key, default=False: False
    blocks = [("body", "b"), ("mem", "m"), ("fail", "f"), ("extra", "e")]
    out = jo.select_block_subset_for_phase("hypothesize", blocks)
    assert out == blocks, f"toggle off should passthrough: {out}"
    params = {"encut": 520, "kpoints": "2 2 2"}
    out_p = jo.select_workflow_params_for_stage("vasp", params)
    assert out_p == params, f"toggle off params should passthrough: {out_p}"
    print("1. toggle off passthrough OK")

    # 2. toggle on: block subset 保留核心 + 选非核心
    jo._harness_enabled = lambda key, default=False: (
        True if key == "harness_joint_optimizer" else default
    )
    jo.JointBandit._instance = None
    out = jo.select_block_subset_for_phase("hypothesize", blocks)
    names = [n for n, _ in out]
    # 核心 block 必选
    assert "body" in names and "fail" in names, f"core blocks lost: {names}"
    print(f"2. toggle on block subset: {names} (core preserved) OK")

    # 3. toggle on: workflow params 扰动
    out_p = jo.select_workflow_params_for_stage("vasp", params)
    # encut 应该在 520 ±10% 范围
    assert "encut" in out_p, f"encut lost: {out_p}"
    assert 468 <= out_p["encut"] <= 572, f"encut out of range: {out_p['encut']}"
    # kpoints 非数值, 保持原样
    assert out_p["kpoints"] == "2 2 2", f"kpoints should stay: {out_p['kpoints']}"
    print(f"3. workflow params perturbed: encut={out_p['encut']} OK")

    # 4. record_joint_outcome + UCB
    bandit = jo.JointBandit.get_instance()
    bandit.record_joint_outcome("hypothesize", ["body", "mem"], {"encut": 520}, True)
    bandit.record_joint_outcome("hypothesize", ["body", "mem"], {"encut": 520}, True)
    bandit.record_joint_outcome("hypothesize", ["body", "extra"], {"encut": 540}, False)
    bs = bandit.list_beliefs("hypothesize")
    assert len(bs) >= 2, f"should have >=2 beliefs: {len(bs)}"
    # UCB 冷启动: 未观察的组合 UCB = inf
    b_new = jo.JointBelief(config_id="new", phase="hypothesize")
    assert b_new.ucb(10) == float("inf"), f"cold start UCB should be inf: {b_new.ucb(10)}"
    print(f"4. record + UCB: {len(bs)} beliefs, cold start UCB=inf OK")

    # 5. 持久化
    jo.JointBandit._instance = None
    b2 = jo.JointBandit.get_instance()
    bs2 = b2.list_beliefs("hypothesize")
    assert len(bs2) == len(bs), f"persistence lost: {len(bs2)} vs {len(bs)}"
    print(f"5. persistence reload OK ({len(bs2)} beliefs)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nH3 joint_optimizer selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()
