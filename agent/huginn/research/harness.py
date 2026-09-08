"""Self-Harness 报告 — 对齐 Qoder/Better Harness 五维 + 组织层任务实录(Self-Harness Report, M1).

阐明红线(Better Harness 同源): 「配置存在 ≠ 能力可用」——本模块不复制 Better Harness
实现, 只采纳它的方法论: 用一条**证据绑定**的五维评分, 把我们散装的机械门禁 (planner /
security / claim_grounding / structural_gate / C-Space / 能力自省) 收敛成能
「横跨项目/会话看」的治理面.

evidence 三态 (取自 spec §2):
  - Observed  : 该检查在本任务实录里真实执行并留下了结果.
  - Unobserved: 机制存在(已接线), 但本任务没走到.
  - Missing   : 无对应机制或无从推断 → 诚实标缺失并计入缺口.

透明计分 (spec §3): passed=1 / unobserved=0.5 / missing=0 (observed 但 failed=0).
保留各检查项明细, 供 reading 用「这条实录到底触发过哪些门」.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


# ── Task Episode 统一身份 (spec §4) ─────────────────────────────────────────
def make_task_episode_id(goal: str, salt: str = "") -> str:
    """为同一 intent 生成稳定、可跨 Agent/项目/机器聚合的 episode id.

    设计: goal 的规范化 slug + 一个「非突变」时间桶种子. 同一任务在同一天内的
    多次 run 都落到同一个 episode id, 从而把「同一需求的实录」聚合到治理账本的一行.
    传入 salt(如 agent/machine 名) 可把 id 限定到单实例维度.

    用 day 级时间桶而非精确时戳: 保证聚合稳定性; 想要随时间演化, 把时间桶改成
    周/次版本即可(spec deferred).
    """
    slug = re.sub(r"[^0-9a-z_]+", "_", (goal or "task").lower()).strip("_") or "task"
    bucket = time.strftime("%Y%m%d", time.localtime())  # day 级桶 → 同任务当日稳定
    seed = f"{slug}::{bucket}::{salt}".encode("utf-8")
    return f"ep-{hashlib.sha256(seed).hexdigest()[:20]}"


# ── 检查项 / 维度结果 ────────────────────────────────────────────────────────
# evidence 三态: observed / unobserved / missing
EVIDENCE_OBSERVED = "observed"
EVIDENCE_UNOBSERVED = "unobserved"
EVIDENCE_MISSING = "missing"

# 分值: passed=1 / unobserved=0.5 / missing=0 / observed-but-failed=0
_SCORE = {
    "passed": 1.0,
    "unobserved": 0.5,
    "missing": 0.0,
    "failed": 0.0,
}


@dataclass
class CheckResult:
    """一条检查项的实录结果(可被 reading 逐条还原)."""

    dimension: str
    name: str
    evidence: str            # observed / unobserved / missing
    outcome: str             # passed / failed / unobserved / missing
    detail: str = ""
    ref: str = ""            # 指向形如 "out.plan_summary" 的证据来源, 便于溯源

    def score(self) -> float:
        return _SCORE[self.outcome]


#── P2 聚合头投影表(模块级): 六维 ← consolidated.head_details 的映射 ──────
# 缺头 → 生成 unobserved 检查(诚实标"该源未注册/未触发", 不伪造).
_P2_DIM_HEADS: dict[str, list[str]] = {
    "task_understanding": ["audit.plan_revision"],
    "controlled_execution": ["controlled.supervision"],
    "change_validation": ["gate.claim_grounding", "gate.structural"],
    "reliable_delivery": ["reliable.evidence_cache", "gate.workspace"],
    "learning_capture": ["learning.self_audit", "meta.overbuild_guard"],
    "safety_authority": ["wm.actually_used", "audit.score_usage",
                         "team.diversity", "governance.external_verify"],
}
_P2_DIM_LABEL = {
    "task_understanding": "需求拆解/plan 修订(聚合投影)",
    "controlled_execution": "受控执行(聚合投影)",
    "change_validation": "变更验证(声明/结构闸门聚合)",
    "reliable_delivery": "可靠交付(证据缓存/工作区聚合)",
    "learning_capture": "自省与经验沉淀(聚合投影)",
    "safety_authority": "权威审计(世界模型真用/得分≠使用聚合)",
}


@dataclass
class DimensionScore:
    name: str
    score: float
    evidence: str            # 该维聚合后的证据态: observed 若任一子项 observed
    checks: list[CheckResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "evidence": self.evidence,
            "checks": [c.__dict__ for c in self.checks],
        }


@dataclass
class HarnessReport:
    """一条任务实录的五维治理报告 (spec §3/§4).

    构造后调用 :meth:`assess` 从 ``out``(ResearchOutcome) 提炼五维评分; 输出
    :meth:`to_json()` 带 task_episode 身份, 可直接作为组织层治理账本的一行.
    红线: 打分「跟证据走」 — 例: 没跑 planner 就说 Task Understanding=unobserved,
    而不是因为代码里有 planning.py 就给满分.
    """

    goal: str
    task_episode: str = ""
    agent: str = ""
    machine: str = ""
    dimensions: list[DimensionScore] = field(default_factory=list)
    overall: float = 0.0

    def assess(self, out: Any, *, agent: str = "", machine: str = "") -> "HarnessReport":
        """从 ResearchOutcome 提炼五维评分. 只复用 out 已记录的 gate 结果, 不重复计算.

        spec §1 映射表:
          1 Task Understanding   <- out.plan_summary      (planner)
          2 Controlled Execution <- out.supervision_log + plan budget (HITL/权限)
          3 Change Validation    <- out.verdict/structural_gate/report_source
          4 Reliable Delivery    <- out.cache/workspace_verified (reconcile/对账/回滚)
          5 Learning Capture     <- 能力自省提案 + 审计产出 (structural_gate 等)
          6 Safety/Authority     <- 谁有权叫停/是否真 D (外部安全否决器 + 世界模型真用?)
             (对齐 Physical AI 六格判据: "谁有下一步决定权" + 误区二"训练用过 future ≠
              部署时在规划"。权威审计诚实标真——law_model 只存在于代码里但未进决策路径,
              就是 Unobserved, 不给它"世界模型规划"的虚名。)
        """
        self.agent = agent
        self.machine = machine
        self.task_episode = self.task_episode or make_task_episode_id(self.goal, salt=f"{agent}:{machine}")

        # P2 双轨: 有收敛聚合头(管道产物) → 从 out.consolidated.head_details 投影六维
        # (单一入口, 不再 ad hoc 扫 15 个 out.* 字段); 无 consolidated(直构 out /
        # 旧数据) → 退回既有的逐字段扫描, 保证单元测试与旧调用语义不变.
        cons = getattr(out, "consolidated", None)
        if cons and cons.get("head_details"):
            checks = self._project_consolidated(cons)
        else:
            checks = [
                *self._task_understanding(out),
                *self._controlled_execution(out),
                *self._change_validation(out),
                *self._reliable_delivery(out),
                *self._learning_capture(out),
                *self._safety_authority(out),
            ]

        by_dim: dict[str, list[CheckResult]] = {}
        for c in checks:
            by_dim.setdefault(c.dimension, []).append(c)

        self.dimensions = []
        for dim in ("task_understanding", "controlled_execution",
                    "change_validation", "reliable_delivery", "learning_capture",
                    "safety_authority"):
            items = by_dim.get(dim, [])
            if not items:
                continue
            scores = [c.score() for c in items]
            dim_evidence = (EVIDENCE_OBSERVED
                            if any(c.evidence == EVIDENCE_OBSERVED for c in items)
                            else (EVIDENCE_UNOBSERVED
                                  if any(c.evidence == EVIDENCE_UNOBSERVED for c in items)
                                  else EVIDENCE_MISSING))
            ds = DimensionScore(name=dim, score=sum(scores) / len(scores),
                                evidence=dim_evidence, checks=items)
            self.dimensions.append(ds)

        # 总体 = 五维均值; 无任何维时记为 0
        self.overall = (sum(d.score for d in self.dimensions) / len(self.dimensions)
                        if self.dimensions else 0.0)
        return self

    # ── 五维检查项从 out 提炼 ──────────────────────────────────────────────
    # P2: 聚合头(consolidated.head_details) → 六维投影表. 每个维映射到一组 head,
    # 缺头生成 unobserved 检查(诚实标"该源未注册/未触发", 不伪造). 六维集合与
    # 既有断言 {task_understanding, ..., safety_authority} 天然一致.
    @classmethod
    def _project_consolidated(cls, cons: dict) -> list[CheckResult]:
        heads = {h["id"]: h for h in cons.get("head_details", [])}
        results: list[CheckResult] = []
        for dim, ids in _P2_DIM_HEADS.items():
            items = [heads[i] for i in ids if i in heads]
            if not items:
                results.append(CheckResult(
                    dim, _P2_DIM_LABEL[dim], EVIDENCE_UNOBSERVED, "unobserved",
                    detail="聚合头未注册该维来源", ref="out.consolidated"))
                continue
            for h in items:
                results.append(CheckResult(
                    dim, h.get("name", h["id"]),
                    h.get("evidence", EVIDENCE_UNOBSERVED),
                    h.get("outcome", "unobserved"),
                    detail=h.get("detail", ""), ref=h.get("ref", "")))
        return results

    @staticmethod
    def _task_understanding(out: Any) -> list[CheckResult]:
        results: list[CheckResult] = []
        if getattr(out, "plan_summary", None) is not None:
            results.append(CheckResult(
                "task_understanding", "需求拆解(planner 产出计划)",
                EVIDENCE_OBSERVED, "passed",
                detail=f"plan_summary={out.plan_summary!r}", ref="out.plan_summary"))
        else:
            results.append(CheckResult(
                "task_understanding", "需求拆解(planner 产出计划)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="未传 planner → plan_summary 为空", ref="out.plan_summary"))
        # 漏A · plan 修订门: 跑过 planner 且有执行后证据 → 显式复盘(锚定反制).
        rev = getattr(out, "plan_revision", None)
        if rev is not None:
            results.append(CheckResult(
                "task_understanding", "plan 修订门(新证据→显式复盘初始计划)",
                EVIDENCE_OBSERVED,
                "passed" if rev.get("verdict") == "plan_holds" else "failed",
                detail=f"plan_revision={rev!r}", ref="out.plan_revision"))
        return results

    @staticmethod
    def _controlled_execution(out: Any) -> list[CheckResult]:
        hits = getattr(out, "supervision_log", None)
        n = len(hits) if hits else 0
        if n > 0:
            return [
                CheckResult("controlled_execution", "受控执行(HITL/权限评审记录)",
                            EVIDENCE_OBSERVED, "passed",
                            detail=f"记录 {n} 条", ref="out.supervision_log"),
            ]
        return [
            CheckResult("controlled_execution", "受控执行(HITL/权限评审记录)",
                        EVIDENCE_UNOBSERVED, "unobserved",
                        detail="supervisor_every=0 或未触发人工评审", ref="out.supervision_log"),
        ]

    @staticmethod
    def _change_validation(out: Any) -> list[CheckResult]:
        results: list[CheckResult] = []
        # 声明门禁: verdict
        verdict = getattr(out, "verdict", "needs_grounding")
        passed = verdict in ("grounded", "pass", "accept", "confirmed")
        results.append(CheckResult(
            "change_validation", "声明门禁(claim_grounding)",
            EVIDENCE_OBSERVED if verdict != "needs_grounding" or getattr(out, "ungrounded", None)
            else EVIDENCE_UNOBSERVED,
            "passed" if passed else ("failed" if getattr(out, "ungrounded", None) else "unobserved"),
            detail=f"verdict={verdict}", ref="out.verdict"))
        # 结构闸门
        sg = getattr(out, "structural_gate", None)
        if sg is not None:
            ok = bool(sg.get("pass") or sg.get("aligned"))
            results.append(CheckResult(
                "change_validation", "结构闸门(交互等效/多元论审计)",
                EVIDENCE_OBSERVED, "passed" if ok else "failed",
                detail=str(sg), ref="out.structural_gate"))
        else:
            results.append(CheckResult(
                "change_validation", "结构闸门(交互等效/多元论审计)",
                EVIDENCE_UNOBSERVED, "unobserved", detail="未注入 structural_audit",
                ref="out.structural_gate"))
        return results

    @staticmethod
    def _reliable_delivery(out: Any) -> list[CheckResult]:
        results: list[CheckResult] = []
        cache = getattr(out, "cache", None)
        if cache:
            results.append(CheckResult(
                "reliable_delivery", "真实执行证据(缓存/expts 实录)",
                EVIDENCE_OBSERVED, "passed",
                detail=f"{len(cache)} 个实验", ref="out.cache"))
        else:
            results.append(CheckResult(
                "reliable_delivery", "真实执行证据(缓存/expts 实录)",
                EVIDENCE_UNOBSERVED, "unobserved", detail="cache 为空", ref="out.cache"))
        wv = getattr(out, "workspace_verified", None)
        if wv is not None:
            results.append(CheckResult(
                "reliable_delivery", "工作区广播门(C-Space 在场断言)",
                EVIDENCE_OBSERVED, "passed" if wv else "failed",
                detail=f"workspace_verified={wv}", ref="out.workspace_verified"))
        else:
            results.append(CheckResult(
                "reliable_delivery", "工作区广播门(C-Space 在场断言)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="未注入 workspace(可选) ", ref="out.workspace_verified"))
        return results

    @staticmethod
    def _learning_capture(out: Any) -> list[CheckResult]:
        # 能力自省/审计产出: 以各种审计 gate 的结果作为「下个任务可复用知识」的证据.
        # 主导信号是有没有留下审计/规划产物; 导入能力仅用于区分 unobserved/missing.
        audit_artifacts = ("structural_gate", "structural_aligned", "converred",
                           "supervision_log", "plan_summary")
        produced_audit = any(getattr(out, name, None) is not None for name in audit_artifacts)
        if produced_audit:
            return [
                CheckResult("learning_capture", "能力自省闭环(缺口→提案)",
                            EVIDENCE_OBSERVED, "passed",
                            detail="该 run 产生了可复用审计/规划产物", ref="out.structural_gate"),
            ]
        try:
            from huginn.capabilities.introspection import self_audit  # noqa: F401
        except Exception:  # noqa: BLE001 — 导入失败视为机制缺失/缺口
            return [
                CheckResult("learning_capture", "能力自省闭环(缺口→提案)",
                            EVIDENCE_MISSING, "missing",
                            detail="无法加载自省机制", ref="capabilities/introspection"),
            ]
        return [
            CheckResult("learning_capture", "能力自省闭环(缺口→提案)",
                        EVIDENCE_UNOBSERVED, "unobserved",
                        detail="自省已接线但本 run 无审计产物", ref="capabilities/introspection"),
        ]

    @staticmethod
    def _safety_authority(out: Any) -> list[CheckResult]:
        """第六维 —— 对齐 Physical AI 六格判据: "谁有下一步决定权".

        两项检查(都诚实标真, 配置存在≠能力可用):
          1. 外部安全否决器: 是否存在不共享策略参数、可独立验证的"叫停/改判"组件?
             物理 AI 洞察: 学习提出动作, 独立安全模块在外部拒绝 —— 是可分开验证的两个。
             → 我们的 structural_gate / claim_grounding / C-Space 广播门属此列。
          2. 世界模型真用?(误区二): predict 的后果是否真的参与当前决策选择?
             只在代码里有 law_model.py 不算; 只有预测结果改变选择的才是真深思 D。
        """
        results: list[CheckResult] = []
        # (1) 外部安全否决器 —— 独立门禁是否真实触发
        gate_attrs = ("structural_gate", "ungrounded", "workspace_verified")
        gate_fired = any(getattr(out, name, None) not in (None, []) for name in gate_attrs)
        if gate_fired:
            results.append(CheckResult(
                "safety_authority", "外部安全否决器(可独立验证的叫停组件)",
                EVIDENCE_OBSERVED,
                "passed" if getattr(out, "structural_aligned", None) is not False else "failed",
                detail="结构闸门/声明门/工作区门任一在本 run 真实触发",
                ref="out.structural_gate/out.ungrounded/out.workspace_verified"))
        else:
            results.append(CheckResult(
                "safety_authority", "外部安全否决器(可独立验证的叫停组件)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="本 run 未触发任何独立门禁(未注入 structural_audit/workspace)",
                ref="out.structural_gate/out.ungrounded/out.workspace_verified"))

        # (2) 世界模型真用? —— 误区二诚实审计: predict 是否参与决策而非仅在代码里
        # 判据: 决策/规划路径里有意义地消费了 predict/reconcile 的产物, 才算真 D。
        # 更进一步: 用了≠全对 —— 若 reconcile 对账存在但 borne_out_all=False, 如实标
        # observed-failed(预测被真实执行证伪, 这是发现不是失败), 不虚报"世界模型对啊"。
        wm_attrs = ("law_model_used", "model_based", "world_model_used", "planner_rollout")
        wm_evidence = next((getattr(out, a, None) for a in wm_attrs
                            if getattr(out, a, None) not in (None, False, [])), None)
        if wm_evidence is None:
            results.append(CheckResult(
                "safety_authority", "世界模型真用?(predict 参与决策 = 真深思 D)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="LawModel 存在但本 run 未见 predict/reconcile 产物 —— 能力未进决策路径",
                ref="out.law_model_used/out.planner_rollout"))
        else:
            wm_dict = wm_evidence if isinstance(wm_evidence, dict) else {}
            borne_out_all = wm_dict.get("borne_out_all")
            has_reconcile = bool(wm_dict.get("reconcile"))
            # 有对账且被证伪 → observed-failed(诚实的发现); 否则 passed(用了 or 无对账素材)
            outcome = ("failed" if (has_reconcile and borne_out_all is False) else "passed")
            detail = (f"predict 参与决策: {wm_evidence!r}; "
                      f"reconcile={wm_dict.get('reconcile', [])}"
                      if has_reconcile else f"predict 参与决策: {wm_evidence!r}")
            results.append(CheckResult(
                "safety_authority", "世界模型真用?(predict 参与决策 = 真深思 D)",
                EVIDENCE_OBSERVED, outcome, detail=detail, ref="out.law_model_used"))

        # 漏C · "得分≠使用"(看过≠用过, Jain & Wallace 镜像): 高分存活项是否真进
        # 最终报告 — 高 attention/高分 不等于 真被用于产出结论.
        ga = getattr(out, "grounding_audit", None)
        if ga is not None:
            results.append(CheckResult(
                "safety_authority", "得分≠使用(高分存活项真进最终报告?)",
                EVIDENCE_OBSERVED,
                "passed" if ga.get("verdict") == "proper_use" else "failed",
                detail=f"grounding_audit={ga!r}", ref="out.grounding_audit"))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_episode": self.task_episode,
            "goal": self.goal,
            "agent": self.agent,
            "machine": self.machine,
            "overall": round(self.overall, 3),
            "dimensions": [d.as_dict() for d in self.dimensions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def build_harness_report(goal: str, out: Any, *, agent: str = "", machine: str = "") -> HarnessReport:
    """便捷入口: 从 out 生成并评估一份 Self-Harness 报告."""
    return HarnessReport(goal=goal).assess(out, agent=agent, machine=machine)