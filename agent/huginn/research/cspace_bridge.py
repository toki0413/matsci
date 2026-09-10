"""deep_think → CSpace 桥 — 外部思维草稿进可证伪工作区.

对接 docs/deepthink-cspace-bridge-spec.md。一条红线:
  **deep_think 草稿 ≠ 在场。** 草稿经此桥进入 C-Space 只是候选
  (falsifiable=False, pending), 只有被证据证实 (promote) 才成为可引用在场
  (falsifiable=True)。

职责:
  - :class:`DeliberationBeing`  复用 ``CSpace.Being``, 加派生字段 ``phase``;
    id 稳定幂等 ``dt_ + sha1(origin+phase+claim)[:12]``。
  - :func:`enqueue_deliberation` / :func:`ingest_reasoning_trace`  批量把
    reasoning_trace 里的结构化记录灌成 candidate Being。
  - :func:`promote_to_at_hand`  门禁: 无证据证实的草稿永不确认在场。
  - :func:`reject` / :func:`confirmed_at_hand`  治理与"唯一可引用集合"。
  - :func:`reconcile_estimate`  M3 明星路径: 定量预判用 law_model.reconcile
    式数值对账, 吻合 → 强在场, 不符 → 如实 rejected 进治理账本。

失败模式一律 fail-open: memory=None → no-op; 空/套话 → rejected_empty 计数;
id 撞车 → 幂等合并; 访问器缺位 → 回退"无桥"行为。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from huginn.research.cspace import Being, CSpace

# 允许的推理阶段 (对齐 ReasoningPhase; 非法值回退 think)
PHASES = ("think", "plan", "pre_action", "reflect")
# 治理账本计数键 (暴露给 report/账本统计"套话率"/证伪率)
_GOV_KEYS = ("rejected_empty", "pending", "confirmed", "rejected")
# 数值抽取 (含科学计数法, 如 1.0e-9)
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class DeliberationBeing(Being):
    """从 deep_think 草稿派生的工作区在场.

    复用 CSpace.Being 的证据语义, 额外带 ``phase`` 阶段标记。草稿默认
    falsifiable=False → 只出现在 ``pending_no_source``, 绝不能进 ``at_hand``。
    """

    phase: str = "think"


def _being_id(origin: str, phase: str, claim: str) -> str:
    """稳定幂等 id: 'dt_' + sha1(origin+phase+claim)[:12] (同草稿重复想不撞车)."""
    import hashlib

    digest = hashlib.sha1(f"{origin}|{phase}|{claim}".encode()).hexdigest()[:12]
    return f"dt_{digest}"


def _norm_phase(phase: Any) -> str:
    s = str(phase or "")
    return s if s in PHASES else "think"


def _extract(record: Any, origin: str) -> tuple[str, str, str, str, str]:
    """从 ReasoningRecord / dict / 扁平 str 抽 (phase, claim, estimate, uncertainty, plan)."""
    if hasattr(record, "claim"):  # ReasoningRecord
        return (
            _norm_phase(getattr(record, "phase", "think")),
            str(getattr(record, "claim", "") or ""),
            str(getattr(record, "estimate", "") or ""),
            str(getattr(record, "uncertainty", "") or ""),
            str(getattr(record, "plan", "") or ""),
        )
    if isinstance(record, dict):
        claim = str(record.get("claim", record.get("text", "")) or "")
        return (
            _norm_phase(record.get("phase", "think")),
            claim,
            str(record.get("estimate", "") or ""),
            str(record.get("uncertainty", "") or ""),
            str(record.get("plan", "") or ""),
        )
    txt = str(record or "")
    return ("think", txt, "", "", "")


def _nonempty(s: Any) -> bool:
    return bool(str(s or "").strip())


def _gov(cspace: CSpace) -> dict[str, int]:
    """治理计数容器 (挂在 cspace 实例上, 避免跨工作区串账)."""
    g = getattr(cspace, "_bridge_gov", None)
    if g is None:
        g = dict.fromkeys(_GOV_KEYS, 0)
        cspace._bridge_gov = g
    return g


def _safe_call(fn: Callable[..., bool], *args: Any) -> bool:
    try:
        return bool(fn(*args))
    except Exception:  # noqa: BLE001 — 门禁回调异常按"不通过"处理, 不阻断主流程
        return False


def _safe_verify(verifier: Callable[[str, list[str]], dict], text: str,
                 trace: list[str]) -> dict:
    try:
        return verifier(text, trace)
    except Exception:  # noqa: BLE001 — 校验异常按"未证实"处理
        return {"verdict": "needs_grounding"}


def _has_numeric(text: str) -> bool:
    """是否含可作为证据对账的数值断言 (定性软引用不算证据在场)."""
    try:
        from huginn.validation.claim_grounding import extract_numeric_claims
        return bool(extract_numeric_claims(text or ""))
    except Exception:  # noqa: BLE001 — 数值抽取不可用 → 保守交给 verifier 判定
        return True


def _first_number(s: Any) -> float | None:
    m = _NUM_RE.search(str(s or ""))
    try:
        return float(m.group(0)) if m else None
    except (TypeError, ValueError):
        return None


# ── 契约: 入桥 ───────────────────────────────────────────────────


def enqueue_deliberation(cspace: CSpace, record: Any, *,
                         origin: str = "deep_think") -> Being | None:
    """从单条 ReasoningRecord/str 建 Being (falsifiable=False); 幂等同 id 不重复.

    cspace 为 None → fail-open 返回 None; 空/纯空白草稿 → rejected_empty 计数后返回 None.
    """
    if cspace is None:
        return None  # fail-open: memory 缺位不抛
    gov = _gov(cspace)
    phase, claim, estimate, uncertainty, plan = _extract(record, origin)
    if not _nonempty(claim):
        gov["rejected_empty"] += 1
        return None
    being_id = _being_id(origin, phase, claim)
    # 幂等合并: 同 origin+phase+claim 重复想 → 直接返回既有 Being, 不重复入场
    if being_id in cspace.beings:
        return cspace.beings[being_id]
    b = DeliberationBeing(
        id=being_id,
        kind="concept",
        payload={
            "phase": phase,
            "claim": claim,
            "estimate": estimate,
            "uncertainty": uncertainty,
            "plan": plan,
        },
        source=f"reasoning_trace:{origin}:{being_id}",  # 证据追溯指针
        falsifiable=False,
        phase=phase,
    )
    cspace.beings[being_id] = b
    return b


def ingest_reasoning_trace(cspace: CSpace, records: list[Any], *,
                           origin: str = "deep_think") -> list[Being]:
    """批量入桥: 每条 → candidate; 空/套话 → rejected_empty 计数并跳过.

    幂等: 同 id 草稿不重复入场; cspace 为 None → fail-open 返回空.
    """
    if cspace is None:
        return []
    out: list[Being] = []
    seen = set(cspace.beings)  # 已存在的 id → 重复草稿只幂等合并, 不重复上报
    for rec in records:
        b = enqueue_deliberation(cspace, rec, origin=origin)
        if b is not None and b.id not in seen:
            out.append(b)
            seen.add(b.id)
    return out


def ingest_from_memory(cspace: CSpace, memory_manager: Any, *,
                       origin: str = "deep_think") -> list[Being]:
    """M2 只读接线: 从 memory_manager 读回结构化推理记录入桥.

    fail-open: memory_manager 为 None 或访问器缺位 → no-op 返回空, 不抛.
    """
    if cspace is None or memory_manager is None:
        return []
    try:
        records = list(memory_manager.iter_reasoning_records())
    except Exception:  # noqa: BLE001 — 访问器缺位/读失败 → 无桥行为, flag 兜底为空
        records = []
    return ingest_reasoning_trace(cspace, records, origin=origin)


# ── 契约: 门禁 / 治理 ───────────────────────────────────────────


def promote_to_at_hand(cspace: CSpace, being_id: str, *,
                       corroborate: Callable[[Being, CSpace], bool] | None = None,
                       verify: Callable[[str, list[str]], dict] | None = None) -> dict:
    """门禁: 把 candidate 提升为 confirmed (falsifiable=True), 进 at_hand.

    默认门禁: 定性 claim 仅当其 claim/estimate 的数值断言能在 ``cspace.trace``
    找到真值才 confirmed (复用 :func:`grounding_verifier`)；否则保持 candidate
    并计数 pending。显式 ``corroborate`` 不通过 → 如实 rejected/falsified。

    verify 复用 product 单一门禁实现, 不另造 gate.
    """
    gov = _gov(cspace)
    b = cspace.beings.get(being_id)
    if b is None:
        gov["pending"] += 1
        return {"promoted": False, "state": "candidate", "reason": "unknown_being"}
    if b.falsifiable:
        # 已 confirmed, 幂等幂等不重复计数
        return {"promoted": True, "state": "confirmed"}
    verifier = verify or cspace.verify
    if corroborate is not None:
        if _safe_call(corroborate, b, cspace):
            b.falsifiable = True
            gov["confirmed"] += 1
            return {"promoted": True, "state": "confirmed"}
        # 显式证据对账失败 → 如实 rejected/falsified 进治理账本
        gov["rejected"] += 1
        return {"promoted": False, "state": "rejected", "reason": "gate_failed"}
    # 默认门禁 (定性 claim 的数值断言须能在 trace 找真值)
    claim_text = " ".join(
        p for p in (b.payload.get("claim", ""), b.payload.get("estimate", "")) if p
    )
    if not _has_numeric(claim_text):
        gov["pending"] += 1
        return {"promoted": False, "state": "candidate", "reason": "no_numeric_claim"}
    g = _safe_verify(verifier, claim_text, cspace.trace)
    if g.get("verdict") == "pass":
        b.falsifiable = True
        gov["confirmed"] += 1
        return {"promoted": True, "state": "confirmed"}
    gov["pending"] += 1
    return {"promoted": False, "state": "candidate", "reason": "no_evidence_in_trace"}


def reject(cspace: CSpace, being_id: str, *, reason: str = "") -> None:
    """把草稿降级为 rejected: 标 suppressed 并记治理计数 (不再可引用)."""
    gov = _gov(cspace)
    b = cspace.beings.get(being_id)
    if b is None:
        return
    b.suppressed = True
    b.falsifiable = False
    if reason:
        b.payload["reject_reason"] = reason
    gov["rejected"] += 1


def confirmed_at_hand(cspace: CSpace) -> list[Being]:
    """唯一可引用在场集合: falsifiable=True 且未被压制. 报告/决策只引这里."""
    if cspace is None:
        return []
    return [b for b in cspace.beings.values()
            if b.falsifiable and not b.suppressed]


def governance(cspace: CSpace) -> dict:
    """治理账本: {rejected_empty, pending, confirmed, rejected} 计数 (给 report/账本)."""
    if cspace is None:
        return dict.fromkeys(_GOV_KEYS, 0)
    return dict(_gov(cspace))


# ── M3: 定量预判 reconcile 门禁 ─────────────────────────────────


def contract_gate(quantities: dict | None = None, *,
                  objectives_key: str = "objectives",
                  strict_coverage: bool = False) -> Callable[[Being, CSpace], bool]:
    """构造一个 corroborate 门禁: 状态 Being 的 objectives 只有落在**域科学契约**
    (量纲/有效域) 内才 confirmed 进场.

    走 :func:`promote_to_at_hand` 的既有 ``corroborate`` 钩子, **不动 C-Space 内核**:
    C-Space 的"状态在场"从"文本溯源"升级为"可证伪且法律一致"。判定经
    ``external_validator.validate_scientific_contract``(非学习、与 harness 独立)。

      - ``Being.payload[objectives_key]`` 须为 {name: value}(状态数值);
      - 越出域声明有效域 → 硬拒(保持 candidate / 记 rejected);
      - coverage gap(未声明量, 有效域未知)默认宽容(与 external_validator 一致),
        传 ``strict_coverage=True`` 可把"未声明"也计作不过;
      - ``quantities`` 缺省时回退到 ``Being.payload["scientific_contract"]``
        (``cspace.add_state`` 已把 LawModel 治理卡片的域契约挂上), 二者皆空 → 不设限.
    """
    def _gate(b: Being, cspace: CSpace) -> bool:
        from huginn.research.external_validator import validate_scientific_contract
        obj = (b.payload or {}).get(objectives_key)
        if not isinstance(obj, dict) or not obj:
            return False   # 状态 Being 必须携带 objectives 才能按契约判定
        # 契约优先用显式传的 quantities; 缺省则用状态 Being 自带契约(卡/工作区灌入).
        q = quantities
        if not q:
            q = ((b.payload or {}).get("scientific_contract") or {}).get("quantities")
        if not q:
            return True   # 无任何契约声明 → 视为不设限
        ok, gaps = validate_scientific_contract(obj, q)
        if not ok:
            return False   # 有效域违反 → 硬拒
        if strict_coverage and any("未声明" in g for g in gaps):
            return False
        return True
    return _gate


def reconcile_estimate(actual: dict | float | None, *, tol: float = 0.03,
                       key: str = "value") -> Callable[[Being, CSpace], bool]:
    """构造一个 corroborate 门禁: 用真实执行 ``actual`` 对 pre_action 的 estimate
    预言做 law_model.reconcile 式数值对账.

    相对误差 |Δ|/|e| <= tol → 证实 (promote 进 confirmed 强在场); 否则不通过
    (promote 按 rejected/falsified 处理, 进治理计数)。对账结果回写
    ``being.payload["reconcile"]`` 供追溯。
    """
    def _corroborate(b: Being, cspace: CSpace) -> bool:
        pred = _first_number(b.payload.get("estimate", ""))
        if pred is None:
            return False
        if isinstance(actual, dict):
            y = _first_number(actual.get(key, actual.get("actual", "")))
        else:
            y = _first_number(actual)
        if y is None or y == 0.0:
            return False
        err = abs(pred - y) / abs(y)
        b.payload["reconcile"] = {
            "predicted": pred, "actual": y, "rel_err": round(err, 4),
            "borne_out": err <= tol, "tol": tol,
        }
        return err <= tol
    return _corroborate


__all__ = [
    "DeliberationBeing", "PHASES",
    "enqueue_deliberation", "ingest_reasoning_trace", "ingest_from_memory",
    "promote_to_at_hand", "reject", "confirmed_at_hand", "governance",
    "contract_gate",
    "reconcile_estimate",
]
