"""认知热机 — v7 G59.

用户思想: 想象力作为主动生产熵、放大差异、解构既有秩序的认知热机.
用湍流理论统一 belief_entropy (被动测熵) + _should_imaginate (主动产熵) + darwin_ratchet (有序提取).

核心方程 (类比 Navier-Stokes):
    ∂ψ/∂t = -∇V(ψ) + ν∇²ψ + f(ψ)
    -∇V: paradigm 收敛势能; ν: cognitive viscosity; f: surprise 驱动

认知 Reynolds 数:
    Re_cog = U · L / ν
    U: 思维速度 (idea generation rate); L: 概念特征尺度; ν: paradigm 约束强度
    Re_cog > Re_crit → 层流失稳, 想象力启动 (转捩)

认知 Carnot 效率:
    η_cog = 1 - T_cold / T_hot
    T_hot: idea 池熵 (belief_entropy 代理); T_cold: paradigm 秩序 (supported_ratio 代理)
    η 接近 0: 想象力产熵但提取不出有序 (空想), 或 paradigm 太紧没产熵 (保守)
    η 接近 1: paradigm 秩序度趋零 (混沌), 想象力失控

能量级联 (Kolmogorov):
    层流 (Re 小): E(k) 集中低 k, 单一 paradigm 主导
    湍流 (Re 大): E(k) ∝ k^(-5/3), 跨尺度都有结构
    涡旋拉伸 ↔ 概念拉伸: idea 被拉伸成多个变体

ponytail: 不做 Navier-Stokes 数值积分 (太重), 只算 0 维代理量.
升级路径: 加 E(k) 谱估计 (RAG recall 距离分布) + 间歇性 kurtosis (idea 产量高阶矩).

认知科学锚点 (arXiv:2510.11503, Collins et al. 2026):
    "People use fast, flat goal-directed simulation to reason about novel problems"
    人类面对全新问题的默认推理模式四特征 — 与本热机的四个量结构同构:
        fast         ↔ U  (少量采样, k≈5-20 次 self-play)
        flat         ↔ ν  (depth-limited 单步 lookahead, paradigm 压扁搜索)
        goal-directed↔ -∇V (抽象目标启发式 = 微观 paradigm 收敛势能)
        probabilistic↔ T_hot (softmax temperature 控制探索/利用)
    Intuitive Gamer (flat+fast) = 低 Re_cog 层流 (default 推理模式)
    Expert Gamer (deep+slow)    = 高 Re_cog 湍流 (should_imaginate 触发后)
    Re_crit 对应 "novice → expert" 相变点.
    k 次 self-play ≈ k 次 Carnot 循环 (每次产功+产熵).
    "Whether a task is worth thinking about at all" = should_imaginate 的 prior.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CognitiveHeatEngine",
    "get_heat_engine",
]


@dataclass
class CognitiveHeatEngine:
    """认知热机状态 — 0 维代理量.

    所有字段都是 [0, 1] 或 [0, ∞) 的标量, 不做向量场积分.
    状态在 engine 关键节点更新: belief_entropy 测后更新 T_hot;
    darwin_ratchet 算后更新 T_cold; hypothesize 后更新 U; _should_imaginate 读 Re_cog.
    """

    # ── 热力学量 ───────────────────────────────────────────
    # T_hot: idea 池熵, belief_entropy 代理. 范围 [0, 1].
    # 高 = 信念系统不确定 (压缩丢信息 / assumption 矛盾), 想象力的热源.
    T_hot: float = 0.0

    # T_cold: paradigm 秩序度, darwin supported_ratio 代理. 范围 [0, 1].
    # 高 = 假设网络证据扎实, validation 提取有序能力强, 想象力的冷源.
    T_cold: float = 0.0

    # ── 流体力学量 ─────────────────────────────────────────
    # U: 思维速度, idea generation rate. 范围 [0, ∞).
    # 用 hypothesis_graph 节点数 / 时间窗口代理 (单位: ideas/轮).
    U: float = 0.0

    # L: 概念特征尺度, idea 的抽象层级. 范围 [0, ∞).
    # 用 stable_principles 数量 + 1 代理 (paradigm 越深 L 越大).
    L: float = 1.0

    # nu: cognitive viscosity, paradigm 约束强度. 范围 (0, ∞).
    # 用 system_prompt 长度 / 1000 代理 (prompt 越长约束越强).
    # ponytail: 粗代理, 升级路径是 LLM 评估 paradigm 严格度.
    nu: float = 1.0

    # ── 派生量 ─────────────────────────────────────────────
    # Re_cog: 认知 Reynolds 数 = U·L/ν. > Re_crit → 想象力启动.
    Re_cog: float = 0.0
    # 转捩阈值. ponytail: 硬编码 2.0, 升级路径是从数据学出来.
    # 锚点: arXiv:2510.11503 的 novice→expert 相变在 k≈5-20 次模拟附近,
    # 对应 U/L 比值约 2-5 (flat 模式), 取下界作保守阈值.
    Re_crit: float = 2.0

    # eta_cog: 认知 Carnot 效率 = 1 - T_cold/T_hot. 范围 (-∞, 1].
    # T_hot=0 时 eta 未定义, 记 0. 超过 0.9 触发 warning (想象力失控).
    eta_cog: float = 0.0

    # ── 状态追踪 ───────────────────────────────────────────
    # 上次 imagination 触发时间 (轮数), 算间歇性用
    last_imaginate_round: int = -1

    # idea 产量历史 (最近 N 轮的 idea 数), 算间歇性 kurtosis 用
    idea_history: list[int] = field(default_factory=list)

    # 累计做功 (belief space 体积变化), Carnot 循环的 W
    cumulative_work: float = 0.0

    # 累计产熵 (T_hot 变化量), Carnot 循环的 Q_hot
    cumulative_entropy_produced: float = 0.0

    # ponytail: RLock (reentrant) — health_check 持锁调 intermittency_kurtosis,
    # 同线程再入不会死锁. ceiling: 拆 lock-free 读副本, 但 0 维代理量不需要.
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def update_T_hot(self, belief_entropy: float) -> None:
        """belief_entropy 测后调. T_hot = belief_entropy."""
        with self._lock:
            old = self.T_hot
            self.T_hot = max(0.0, min(1.0, belief_entropy))
            self.cumulative_entropy_produced += max(0.0, self.T_hot - old)
            self._recompute_eta()

    def update_T_cold(self, supported_ratio: float, darwin_score: float) -> None:
        """darwin_ratchet 算后调. T_cold = supported_ratio (paradigm 秩序代理)."""
        with self._lock:
            self.T_cold = max(0.0, min(1.0, supported_ratio))
            self._recompute_eta()
            # ponytail: darwin_score (0-10) 也含信息, 但跟 supported_ratio 重叠, 不单独记

    def update_kinematics(
        self,
        idea_count: int,
        stable_principles_count: int,
        system_prompt_len: int,
    ) -> None:
        """hypothesize 后调. 更新 U / L / ν, 重算 Re_cog.

        params:
            idea_count: hypothesis_graph 当前节点数
            stable_principles_count: stable_principles 数量 + 1
            system_prompt_len: system_prompt 字符数
        """
        with self._lock:
            # U: idea generation rate, 用 idea_count / 时间窗口代理
            # ponytail: 没真做时间窗口, 用 idea_count 本身做 0 阶代理
            self.U = float(max(idea_count, 0))
            # L: 概念特征尺度, paradigm 越深 L 越大
            self.L = float(max(stable_principles_count, 1))
            # ν: cognitive viscosity, prompt 越长约束越强
            # ponytail: /1000 粗估, 升级路径是 LLM 评估 paradigm 严格度
            self.nu = max(0.1, float(system_prompt_len) / 1000.0)
            self._recompute_Re()
            # 记 idea 历史, 算间歇性
            self.idea_history.append(idea_count)
            if len(self.idea_history) > 20:
                self.idea_history.pop(0)

    def should_imaginate(self, current_round: int) -> bool:
        """转捩判据: Re_cog > Re_crit 时启动想象力.

        替代 engine._should_imaginate 的纯 surprise threshold.
        同时保留 surprise 触发 (向后兼容): T_hot > 0.7 也启动.

        双模式架构 (arXiv:2510.11503 启示):
            低 Re_cog (层流): Intuitive Gamer 式 flat+fast 推理, 走 default path
            高 Re_cog (湍流): Carnot 循环 (forget→generate→critique→ratchet)
            should_imaginate = "task worth thinking about" 的计算对应物
        """
        with self._lock:
            re_triggered = self.Re_cog > self.Re_crit
            entropy_triggered = self.T_hot > 0.7
            if re_triggered or entropy_triggered:
                self.last_imaginate_round = current_round
                return True
            return False

    def record_work(self, belief_space_delta: float) -> None:
        """forget_then_generate 前后调, 记 belief space 体积变化 (做功)."""
        with self._lock:
            self.cumulative_work += float(belief_space_delta)

    def intermittency_kurtosis(self) -> float:
        """idea 产量的高阶矩 (kurtosis), 检测想象力是否间歇性爆发.

        正态分布 kurtosis=3, 间歇性 >3 (尖峰厚尾), 匀速 <3.
        返回 excess kurtosis = kurtosis - 3. 正数 = 间歇性, 负数 = 匀速.

        ponytail: 样本少时不准, 至少 8 个点才算. 升级: 滑动窗口 + bootstrap.
        """
        with self._lock:
            data = list(self.idea_history)
        if len(data) < 8:
            return 0.0
        n = len(data)
        mean = sum(data) / n
        if mean == 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in data) / n
        if var == 0:
            return 0.0
        m4 = sum((x - mean) ** 4 for x in data) / n
        kurt = m4 / (var * var)
        return kurt - 3.0  # excess kurtosis

    def health_check(self) -> dict[str, Any]:
        """健康状态报告. 给前端 / 日志用.

        返回:
            eta_cog: Carnot 效率
            Re_cog: Reynolds 数
            intermittency: 间歇性 excess kurtosis
            status: "healthy" / "stagnant" / "chaotic" / "conservative"
            warnings: list[str]
        """
        with self._lock:
            eta = self.eta_cog
            re = self.Re_cog
            kur = self.intermittency_kurtosis()
            warnings: list[str] = []
            status = "healthy"

            # η 接近 1 + T_hot 高 = 想象力失控 (混沌)
            if eta > 0.9 and self.T_hot > 0.7:
                status = "chaotic"
                warnings.append(
                    f"η_cog={eta:.2f} 接近 1 + T_hot={self.T_hot:.2f} 高: "
                    "想象力产熵但 paradigm 秩序度低, 可能空想"
                )
            # η 接近 0 + Re_cog 低 = paradigm 太紧 (保守)
            elif eta < 0.1 and re < self.Re_crit:
                status = "conservative"
                warnings.append(
                    f"η_cog={eta:.2f} 低 + Re_cog={re:.2f} 低: "
                    "paradigm 约束太紧, 想象力被压制"
                )
            # η 负 = T_cold > T_hot, validation 比 hypothesis 还有序 (异常)
            elif eta < 0:
                status = "stagnant"
                warnings.append(
                    f"η_cog={eta:.2f} 负: T_cold={self.T_cold:.2f} > T_hot={self.T_hot:.2f}, "
                    "validation 提取的有序超过 hypothesis 产熵, 可能停滞"
                )

            # 间歇性爆发检测
            if kur > 3.0:
                warnings.append(
                    f"intermittency kurtosis={kur:.2f} > 3: 想象力间歇性爆发 (burst 模式)"
                )

            return {
                "eta_cog": round(eta, 3),
                "Re_cog": round(re, 3),
                "Re_crit": self.Re_crit,
                "T_hot": round(self.T_hot, 3),
                "T_cold": round(self.T_cold, 3),
                "U": round(self.U, 3),
                "L": round(self.L, 3),
                "nu": round(self.nu, 3),
                "intermittency_kurtosis": round(kur, 3),
                "cumulative_work": round(self.cumulative_work, 3),
                "cumulative_entropy_produced": round(self.cumulative_entropy_produced, 3),
                "status": status,
                "warnings": warnings,
            }

    def reset(self) -> None:
        """新会话时调, 清状态."""
        with self._lock:
            self.T_hot = 0.0
            self.T_cold = 0.0
            self.U = 0.0
            self.L = 1.0
            self.nu = 1.0
            self.Re_cog = 0.0
            self.eta_cog = 0.0
            self.last_imaginate_round = -1
            self.idea_history.clear()
            self.cumulative_work = 0.0
            self.cumulative_entropy_produced = 0.0

    def _recompute_Re(self) -> None:
        """Re_cog = U · L / ν."""
        if self.nu > 0:
            self.Re_cog = self.U * self.L / self.nu
        else:
            self.Re_cog = float("inf")

    def _recompute_eta(self) -> None:
        """η_cog = 1 - T_cold/T_hot. T_hot=0 时记 0 (未定义)."""
        if self.T_hot > 1e-6:
            self.eta_cog = 1.0 - self.T_cold / self.T_hot
        else:
            self.eta_cog = 0.0


# ── 单例 ──────────────────────────────────────────────────────

_singleton: CognitiveHeatEngine | None = None
_singleton_lock = threading.Lock()


def get_heat_engine() -> CognitiveHeatEngine:
    """模块级单例. 一个 agent 进程一个热机."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CognitiveHeatEngine()
    return _singleton


# ── self-check ────────────────────────────────────────────────

def _self_check() -> None:
    """ponytail: 非平凡逻辑留一个 runnable check."""
    eng = CognitiveHeatEngine()

    # 1. Carnot 效率边界
    eng.update_T_hot(0.8)
    eng.update_T_cold(0.2, 5.0)
    assert 0.74 < eng.eta_cog < 0.76, f"η_cog 应该 0.75, 实际 {eng.eta_cog}"

    # 2. Reynolds 数计算
    eng.update_kinematics(idea_count=10, stable_principles_count=3, system_prompt_len=2000)
    # U=10, L=3, ν=2.0 → Re=15
    assert 14.9 < eng.Re_cog < 15.1, f"Re_cog 应该 15, 实际 {eng.Re_cog}"

    # 3. should_imaginate: Re > Re_crit 触发
    assert eng.should_imaginate(1), "Re=15 > 2 应该触发想象力"

    # 4. should_imaginate: Re 低但 T_hot 高也触发
    eng2 = CognitiveHeatEngine()
    eng2.update_T_hot(0.8)
    eng2.update_kinematics(0, 1, 5000)  # Re = 0*1/5 = 0
    assert eng2.should_imaginate(1), "T_hot=0.8 应该触发想象力"

    # 5. 健康状态: chaotic
    eng3 = CognitiveHeatEngine()
    eng3.update_T_hot(0.9)
    eng3.update_T_cold(0.05, 0.5)
    h = eng3.health_check()
    assert h["status"] == "chaotic", f"应该 chaotic, 实际 {h['status']}"

    # 6. 健康状态: conservative
    eng4 = CognitiveHeatEngine()
    eng4.update_T_hot(0.2)
    eng4.update_T_cold(0.15, 1.5)
    eng4.update_kinematics(1, 1, 5000)  # Re = 1/5 = 0.2 < 2
    h4 = eng4.health_check()
    assert h4["status"] == "conservative", f"应该 conservative, 实际 {h4['status']}"

    # 7. 间歇性 kurtosis (匀速 → ~0)
    eng5 = CognitiveHeatEngine()
    eng5.idea_history = [5] * 10
    assert abs(eng5.intermittency_kurtosis()) < 0.1, "匀速 kurtosis 应该 0"

    # 8. 间歇性 kurtosis (爆发 → 正)
    eng6 = CognitiveHeatEngine()
    eng6.idea_history = [0, 0, 0, 0, 20, 0, 0, 0, 0, 20]
    assert eng6.intermittency_kurtosis() > 1.0, "爆发 kurtosis 应该 >1"

    print("CognitiveHeatEngine self-check: 8/8 PASSED")


if __name__ == "__main__":
    _self_check()
