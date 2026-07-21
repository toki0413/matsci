"""材料科学领域失败模式清单.

给对抗 agent (RedTeamReviewer / PhysicsAuditor) 用的显式 checklist.
不是泛泛"做验证", 而是按清单逐项检查具体的失败模式.

设计原则:
- 每条 mode 有 match_keywords (廉价启发式) + description (给 LLM 增强用)
- severity: block (硬阻断) | warn (先警告, 用户可 force proceed) | info (记录)
- 与 physical_precheck.py 的 warn-first 设计对齐: first-principles-violation 是 warn

扩充这个文件 = 扩充元认知层的核心资产.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Severity = Literal["block", "warn", "info"]
Category = Literal[
    "data", "physics", "ml", "transfer", "interpretation", "methodology"
]


@dataclass
class FailureMode:
    """一条领域失败模式."""

    id: str
    category: Category
    description: str
    severity: Severity
    # 廉价启发式: evidence 文本里出现这些关键词就 flag, 让 LLM 做深入判断
    match_keywords: list[str] = field(default_factory=list)
    # 该模式适用的方法族 (None = 所有族). 用于特定族的特例检查
    applies_to_families: list[str] | None = None
    mitigation: str = ""

    def matches(self, evidence: dict[str, Any]) -> bool:
        """廉价关键词匹配, 命中任一关键词即 flag.

        真正的判定逻辑在 red_team / LLM 增强里, 这里只做粗筛.
        """
        if not self.match_keywords:
            return False
        # 把 evidence 拍平成文本, 容忍 dict / str / list 混合
        blob = _flatten_to_text(evidence).lower()
        return any(kw.lower() in blob for kw in self.match_keywords)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "match_keywords": list(self.match_keywords),
            "applies_to_families": list(self.applies_to_families)
            if self.applies_to_families
            else None,
            "mitigation": self.mitigation,
        }


def _flatten_to_text(obj: Any) -> str:
    """递归把 dict/list/any 拍平成空格分隔文本, 给关键词匹配用."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten_to_text(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten_to_text(v) for v in obj)
    return str(obj)


# ── v0 清单 ──────────────────────────────────────────────────────
# 初始版本, 应随使用扩充. 每条都是材料科学里高频出现的具体失败模式,
# 不是泛泛的"数据质量"或"模型准确度".


_DEFAULT_MODES: list[FailureMode] = [
    FailureMode(
        id="data-leakage-structural",
        category="data",
        description="训练集和测试集存在结构相似性泄漏 (同构型/同空间群/同原型)",
        severity="block",
        match_keywords=["split", "train", "test", "cif", "structure"],
        mitigation="按构型原型或空间群做 scaffold-aware split, 报告泄漏率",
    ),
    FailureMode(
        id="data-leakage-temporal",
        category="data",
        description="时间序列数据用未来信息预测过去 (MD 轨迹反向 / 时间窗重叠)",
        severity="block",
        match_keywords=["trajectory", "time", "window", "frame"],
        mitigation="用 rolling-window 或 walk-forward split, 严禁 overlap",
    ),
    FailureMode(
        id="symmetry-unconstrained",
        category="physics",
        description="空间群等价原子被当成不同样本, 破坏对称性约束",
        severity="block",
        match_keywords=["symmetry", "space group", "wyckoff", "equivalent"],
        mitigation="用 pymatgen SymmetrizedStructure 或 spglib 做对称化预处理",
    ),
    FailureMode(
        id="unit-inconsistency",
        category="physics",
        description="单位混乱 (eV vs kJ/mol, GPa vs MPa, K vs eV/kT)",
        severity="block",
        match_keywords=["ev", "kj", "gpa", "mpa", "kelvin", "kbt", "unit"],
        mitigation="用 ase.units 或 pint 强制量纲跟踪, 关键计算前后做单位审计",
    ),
    FailureMode(
        id="extrapolation-as-interp",
        category="ml",
        description="外推伪装成内插 (训练分布外的体系报了高置信度)",
        severity="warn",
        match_keywords=["out-of-distribution", "ood", "extrapolat", "uncertainty"],
        mitigation="报告 GP 不确定性或 ensemble 方差, OOD 体系强制降置信度",
    ),
    FailureMode(
        id="benchmark-selection-bias",
        category="ml",
        description="只在容易的体系上报指标, 隐藏难体系的退化",
        severity="warn",
        match_keywords=["benchmark", "sota", "accuracy", "r2", "mae"],
        mitigation="按体系类别分组报指标, 含 OOD 和难体系子集",
    ),
    FailureMode(
        id="pseudo-correlation-as-cause",
        category="interpretation",
        description="伪相关伪装成因果 (结构与性质恰好同源于合成条件)",
        severity="block",
        match_keywords=["correlat", "cause", "mapping", "descriptor"],
        mitigation="做反事实测试或干预分析, 报告因果方向证据",
    ),
    FailureMode(
        id="first-principles-violation",
        category="methodology",
        description="ML 组件不符合第一性原理 (纯黑盒拟合, 无物理约束)",
        severity="warn",  # warn 不 block, 对齐用户"先警告再 force proceed"偏好
        match_keywords=["blackbox", "black-box", "fit", "regression"],
        mitigation="改用物理约束 ML 或符号回归, 或显式标注为经验模型",
    ),
    FailureMode(
        id="pcr-nondecomposable",
        category="physics",
        description="PCR (相场/晶体塑性) 不可分解性未处理, 多物理场耦合丢项",
        severity="block",
        match_keywords=["phase field", "crystal plasticity", "coupling", "coupled"],
        mitigation="显式列出耦合项, 验证能量/动量守恒, 不可分解项用算子分裂",
    ),
    FailureMode(
        id="convergence-masked",
        category="physics",
        description="收敛性可疑 (k-spacing 过大 / encut 不足 / SCF 未收敛) 但结果报为可靠",
        severity="block",
        match_keywords=["convergence", "encut", "k-spacing", "kpoint", "scf"],
        mitigation="报 convergence test 结果, 关键量做 encut/k-sp 收敛扫描",
    ),
]


_OBS_LOG_PATH = Path(".huginn/failure_mode_obs.jsonl")


class FailureModeRegistry:
    """失败模式注册表, 支持按 category / family 过滤."""

    def __init__(self, modes: list[FailureMode] | None = None) -> None:
        self._modes: dict[str, FailureMode] = {
            m.id: m for m in (modes or _DEFAULT_MODES)
        }
        # ponytail: observed_counts 无 LRU 上限, 长期跑会膨胀;
        # 升级路径是加窗口 LRU + 按 timestamp 老化
        self.observed_counts: dict[str, int] = {}
        self._load_observed_counts()

    def _load_observed_counts(self) -> None:
        """启动时从 jsonl 恢复计数, 文件缺失或损坏就跳过."""
        path = _OBS_LOG_PATH
        if not path.is_file():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    mid = rec.get("mode_id")
                    if mid:
                        self.observed_counts[mid] = (
                            self.observed_counts.get(mid, 0) + 1
                        )
        except (OSError, json.JSONDecodeError):
            # 损坏的日志不阻断, 后续 append 会继续写
            pass

    def record_observation(
        self, mode_id: str, tool_name: str, evidence: str
    ) -> None:
        """记一次失败观察: 计数+1, 追加落盘, 路由信号给 SignalHub.

        任何一步失败都不抛 — 调用方 (tool base) 依赖这个不阻断特性.
        """
        self.observed_counts[mode_id] = (
            self.observed_counts.get(mode_id, 0) + 1
        )
        rec = {
            "timestamp": time.time(),
            "mode_id": mode_id,
            "tool_name": tool_name,
            "evidence": evidence,
        }
        try:
            path = _OBS_LOG_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            # 落盘失败不抹掉内存计数, 下次进程起来会丢这条历史
            pass
        try:
            from huginn.metacog.signal_hub import SignalHub

            # H1: 用 emit (enqueue) 替代 route (丢弃). 之前 route 返回值被丢,
            # 信号永远不进 CSM. 现在 enqueue, reflection drain 拉.
            SignalHub.shared().emit(
                "skill_failure",
                {"mode_id": mode_id, "tool_name": tool_name},
            )
        except Exception:
            # SignalHub / TransitionSignal 链路出问题不阻断
            pass

    def all(self) -> list[FailureMode]:
        return list(self._modes.values())

    def by_id(self, mode_id: str) -> FailureMode | None:
        return self._modes.get(mode_id)

    def for_family(self, family_id: str | None) -> list[FailureMode]:
        """返回适用某方法族的失败模式.

        family_id=None 时返回 applies_to_families 为 None 的通用模式.
        """
        result = []
        for m in self._modes.values():
            if m.applies_to_families is None:
                result.append(m)
            elif family_id and family_id in m.applies_to_families:
                result.append(m)
        return result

    def scan(self, evidence: dict[str, Any], family_id: str | None = None) -> list[FailureMode]:
        """扫描 evidence, 返回命中的失败模式 (按 severity 排序, block 优先)."""
        candidates = self.for_family(family_id)
        hit = [m for m in candidates if m.matches(evidence)]
        # block > warn > info
        order = {"block": 0, "warn": 1, "info": 2}
        hit.sort(key=lambda m: order.get(m.severity, 3))
        return hit

    def add(self, mode: FailureMode) -> None:
        self._modes[mode.id] = mode


# 默认单例, red_team / equivalence_auditor 直接 import 用
DEFAULT_REGISTRY = FailureModeRegistry()


# ── 自检 ─────────────────────────────────────────────────────────
# ponytail: 非平凡逻辑留一个 runnable check. 这里验证 scan 能正确命中
# 关键词, 且 for_family 对特定族的过滤生效.

def _selfcheck() -> None:
    reg = FailureModeRegistry()

    # 1. 通用模式应在任意 family 下命中 (applies_to_families=None)
    all_modes = reg.for_family(None)
    ids = {m.id for m in all_modes}
    assert "data-leakage-structural" in ids

    # 2. scan 命中关键词
    evidence = {"description": "we trained on the full CIF structure set"}
    hit = reg.scan(evidence)
    hit_ids = {m.id for m in hit}
    assert "data-leakage-structural" in hit_ids, f"应命中 structural leakage, got {hit_ids}"

    # 3. block 优先于 warn 排序
    mixed_evidence = {
        "description": "blackbox fit on CIF structures, extrapolated to OOD",
    }
    mixed_hit = reg.scan(mixed_evidence)
    severities = [m.severity for m in mixed_hit]
    if "block" in severities and "warn" in severities:
        assert severities.index("block") < severities.index("warn"), "block 应排在 warn 前"

    # 4. record_observation 计数 + 持久化 (跑完清掉日志, 不污染仓库)
    log_path = _OBS_LOG_PATH
    if log_path.is_file():
        log_path.unlink()
    try:
        r = FailureModeRegistry()
        r.record_observation("selfcheck_mode", "selfcheck_tool", "boom")
        assert r.observed_counts.get("selfcheck_mode") == 1, r.observed_counts
        # 重新构造应从 jsonl 恢复计数
        r2 = FailureModeRegistry()
        assert r2.observed_counts.get("selfcheck_mode") == 1, r2.observed_counts
    finally:
        if log_path.is_file():
            log_path.unlink()

    print("failure_modes selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
