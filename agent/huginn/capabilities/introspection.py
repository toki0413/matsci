"""能力自省闭环 §1–§3 —— 研究 agent 对自己**能力面**的自审计与自提案.

把"开发期人/测试管碎片化"升级为"agent 运行时自审计"。设计对齐 `docs/capabilities-and
-registration-spec.md` 与 `huginn/research/tool_surface.py` 的契约, **不新增注册表**:

  §1 自省盘点  : 快照自己在 ToolRegistry 里的能力面 → CapabilitySurface
                 (registered 已注册 / discoverable LLM可见 / dangling 注册却不可见).
  §2 缺口诊断  : 规则为主 —— 目标(goal)里出现的领域关键词, 若无任何可见工具覆盖 → 缺口.
  §3 提案入轨迹: 缺口 → 能力提案 spec({name, description, parameters, handler_ref=None,
                 status=staged, origin=runtime-proposed}), 强转 canonical_tool_shape 后
                 序列化成**可证伪工件**进研究 trace —— 报告引用即可被 grounding 门禁核实.

诚实边界(遵循本仓库反杜撰纪律):
  - §3 只"提案 spec", **不实现 handler**(handler_ref=None) —— 落地归后续 §5 确认后实现,
    绝不把"缺失即拥有"伪装成能力。
  - rules-first 确定性; client 仅为可选增强, 失败自动回退规则。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


# ── 领域关键词 → 能力描述 的确定性期望表 (规则面, 可复现, 不靠 LLM) ──────────
# 命中任一 keyword(小写) 且 discoverable 无匹配工具 → 记为"覆盖缺口"。
# 每组的 keyword 同时含英文/拉丁与中文变体, 保证中英文目标都被规则命中。
CAPABILITY_EXPECT: list[tuple[tuple[str, ...], str]] = [
    (("topolog", "circulat", "vortex", "vorticit", "拓扑", "环流", "涡", "环量"),
     "高阶拓扑诊断(环流/Hodge 分解)"),
    (("insolat", "equilibrium temp", "radiation", "habitable", "irradian",
      "日晒", "平衡温度", "辐射", "宜居", "辐照"),
     "日晒/平衡温度第一性原理诊断"),
    (("convergence", "finite element", "finite-element", "iga", "order of accura",
      "refinement", "收敛", "有限元", "等几何"),
     "数值收敛率研究(有限元/等几何)"),
    (("conserv", "mass balance", "flux", "energy budget", "守恒", "质量平衡", "通量", "能量收支"),
     "守恒/通量收支诊断"),
    (("time series", "forecast", "extrapolat", "project", "时间序列", "预报", "外推"),
     "留出集外推预测与回算对账"),
]


@dataclass
class CapabilitySurface:
    """agent 运行时能力面快照 (§1)."""
    registered: list[str] = field(default_factory=list)    # ToolRegistry 已注册
    discoverable: list[str] = field(default_factory=list)  # LLM 可见(活性工具)名
    dangling: list[str] = field(default_factory=list)      # 注册却不可见(被隐藏/失活)

    def to_dict(self) -> dict:
        return {
            "registered": self.registered,
            "discoverable": self.discoverable,
            "dangling": self.dangling,
        }


def _registry_tool_names() -> list[str]:
    try:
        from huginn.tools.registry import ToolRegistry
        return ToolRegistry.list_tools()
    except Exception:  # noqa: BLE001 — 重栈不可用(轻量环境), 视为空
        return []


def _discoverable_tool_names() -> list[str]:
    try:
        from huginn.tools.registry import ToolRegistry
        return [s["function"]["name"] for s in ToolRegistry.get_all_schemas()]
    except Exception:  # noqa: BLE001
        return []


def diagnose_surface(registered: list[str] | None = None,
                     discoverable: list[str] | None = None) -> CapabilitySurface:
    """§1 自省盘点. 显式传入用于确定性命中测试; 缺省读真实 ToolRegistry(best-effort)."""
    reg = list(registered) if registered is not None else sorted(_registry_tool_names())
    disc = list(discoverable) if discoverable is not None else sorted(_discoverable_tool_names())
    dangling = sorted(set(reg) - set(disc))
    return CapabilitySurface(registered=reg, discoverable=disc, dangling=dangling)


def _covered(keywords: tuple[str, ...], discoverable: list[str]) -> bool:
    """discoverable 工具名/描述是否覆盖该关键词组 (规则匹配, 大小写不敏感)."""
    hay = " ".join(discoverable).lower()
    return any(k in hay for k in keywords)


def diagnose_gaps(goal: str, discoverable: list[str]) -> list[dict]:
    """§2 缺口诊断(规则). 返回命中且未被覆盖的能力缺口列表."""
    g = (goal or "").lower()
    out = []
    for keywords, desc in CAPABILITY_EXPECT:
        if any(k in g for k in keywords) and not _covered(keywords, discoverable):
            out.append({"reason": f"目标涉及『{desc}』但可见工具面无覆盖", "capability": desc})
    return out


def conform_proposal(raw: dict) -> dict:
    """把能力提案强转 canonical_tool_shape (复用工具面单一形状定义).

    raw 至少含 name/description; parameters 可缺省。返回带规范化 schema 的提案。
    """
    from huginn.research.tool_surface import canonical_tool_shape

    name = raw.get("name") or raw.get("capability", "").strip()
    if not name:
        raise ValueError("能力提案缺少 name")
    schema = canonical_tool_shape(
        name,
        raw.get("description") or f"能力缺口建议实现({name})",
        raw.get("parameters"),
    )
    return {
        "name": name,
        "description": schema["function"]["description"],
        "parameters": schema["function"]["parameters"],
        "handler_ref": raw.get("handler_ref"),   # 提案阶段 None —— 未实现, 不伪装
        "status": "staged",
        "origin": "runtime-proposed",
    }


def propose_capabilities(goal: str, discoverable: list[str],
                         client=None, model: str = "intern-s2-preview") -> list[dict]:
    """§2+§3 由缺口生成能力提案 (rules-first; client 可选增强, 失败回退)."""
    gaps = diagnose_gaps(goal, discoverable)
    if not gaps:
        return []
    # LLM 可选: 为每个缺口补一段更贴题的 description; 解析失败即用规则 desc。
    proposals = []
    for gap in gaps:
        name = _slug(gap["capability"])
        desc = gap["capability"]
        if client is not None:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content":
                               f"给能力『{gap['capability']}』写一句工具 description(中文,≤40字),"
                               "不要输出别的。"}],
                    max_tokens=60, temperature=0.0)
                d = (r.choices[0].message.content or "").strip()
                if d:
                    desc = d[:80]
            except Exception:  # noqa: BLE001
                pass
        proposals.append(conform_proposal({
            "name": name, "description": f"{gap['capability']}：{desc}"[:160]}))
    return proposals


def _slug(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s).strip("_")
    return f"proposed_{s}" if s else "proposed_capability"


def stage_to_trace(proposals: list[dict]) -> list[str]:
    """§3 能力提案序列化为**可证伪工件**, 供拼进研究 trace 作 grounding 证据."""
    return [json.dumps({
        "type": "capability_proposal",
        "status": "staged",
        "origin": "runtime-proposed",
        "handler_implemented": False,
        **_only(_p, ("name", "description", "parameters", "reason")),
    }, ensure_ascii=False) for _p in proposals]


def _only(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


def run_capability_self_audit(goal: str, client=None, model: str = "intern-s2-preview",
                              registered: list[str] | None = None,
                              discoverable: list[str] | None = None) -> dict:
    """执行 §1→§2→§3 整段, 返回 {surface, gaps, proposals, trace} 供上层/harness 使用."""
    surface = diagnose_surface(registered, discoverable)
    gaps = diagnose_gaps(goal, surface.discoverable)
    proposals = propose_capabilities(goal, surface.discoverable, client=client, model=model)
    trace = stage_to_trace(proposals)
    return {
        "surface": surface.to_dict(),
        "gaps": gaps,
        "proposals": proposals,
        "trace": trace,
    }


# ════ §4–§6: 提案生命周期 → 确认 → 实现(Journal) → 回滚 ════════════════════
# 延续 §3 的诚实边界: 提案默认是 staged spec(handler_implemented=False)。
# §4 确认门禁判定"是否值得落地"; §5 仅当拿到**真实 handler/后端** 才实现,
#    绝不把"缺失当拥有"(handler=None 一律拒绝); §6 依据 Journal 回滚, 恢复原能力面。
# 重依赖(CapabilityRegistry / ExternalCapability, 会拉 pydantic)一律 lazy + best-effort,
#    核心状态机纯标准库可测(与 §1–§3 同模式)。

PROPOSAL_STATES = ("staged", "approved", "implemented", "rejected")


@dataclass
class ProposalRecord:
    """能力提案的生命周期记录 (§4)."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    status: str = "staged"            # PROPOSAL_STATES 之一
    reason: str = ""                  # 缺口依据 (create)/ 拒绝/驳回说明
    handler_implemented: bool = False
    auth: str = "runtime-proposed"    # §4: 落地须显式确认(默认未确认)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RealizationRecord:
    """§5 实现的 Journal 记录 —— 承载 §6 回滚所需的完整 provenance.

    prev_registered: 实现前注册表里已有的能力名快照(实现后新增名字据此可精确移除).
    registered:      是否真正落入 CapabilityRegistry(轻量环境下可能仅 journal).
    """
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    prev_registered: list = field(default_factory=list)
    registered: bool = False
    handler_implemented: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RollbackRecord:
    """§6 回滚的结果: 移除了哪个落地能力, 剩余表面恢复成实现前快照."""
    name: str
    removed: bool = False
    restored: list = field(default_factory=list)
    journal_only: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def create_proposal(name: str, description: str, parameters: dict | None = None,
                    reason: str = "") -> ProposalRecord:
    """§4 新建提案: 初始恒为 staged(未确认=不可落地)."""
    p = parameters if isinstance(parameters, dict) else {}
    return ProposalRecord(name=name, description=description, parameters=p, reason=reason)


def confirm_proposal(rec: ProposalRecord, authorized_by: str = "gate") -> ProposalRecord:
    """§4 确认门禁: 仅 staged 可 → approved (拿到落地许可)."""
    if rec.status == "implemented":
        raise ValueError(f"proposal '{rec.name}' 已实现, 无需重复确认")
    if rec.status == "rejected":
        raise ValueError(f"proposal '{rec.name}' 已驳回, 先 reactivate 再确认")
    rec.status = "approved"
    rec.auth = authorized_by or "gate"
    return rec


def reject_proposal(rec: ProposalRecord, reason: str = "") -> ProposalRecord:
    """§4 驳回: staged/approved → rejected; 已实现不能直接驳回(先回滚)."""
    if rec.status == "implemented":
        raise ValueError(f"proposal '{rec.name}' 已实现, 请先 rollback 再驳回")
    rec.status = "rejected"
    rec.reason = reason or rec.reason
    return rec


def reactivate_proposal(rec: ProposalRecord, *,
                        reset_auth: bool = True) -> ProposalRecord:
    """把 rejected 提案重新拉回 staged(允许修改后再走确认门禁)."""
    if rec.status != "rejected":
        raise ValueError(f"只有 rejected 可 reactivate, 当前 '{rec.status}'")
    rec.status = "staged"
    if reset_auth:
        rec.auth = "runtime-proposed"
    return rec


def realize_proposal(rec: ProposalRecord, handler, *,
                     registry=None, reason: str = "") -> RealizationRecord:
    """§5 落地实现: 仅 approved 提案 + **真实 handler/后端** 可实现.

    诚实边界(核心): ``handler`` 为 None → 一律拒绝, 绝不把 staged spec"包装成已实现"
    假信息。真实 handler 可以是可调用(HuginnTool.backend / fn)或显式传入的 registry
    能力装配。重依赖(CapabilityRegistry) lazy + best-effort: 装不进注册表时仅记
    journal(registered=False), 仍保留 provenance 供 §6 回滚。
    """
    if rec.status != "approved":
        raise ValueError(
            f"proposal '{rec.name}' 未确认(auth='{rec.auth}'), 须先 confirm_proposal")
    if handler is None:
        raise ValueError(
            f"proposal '{rec.name}' 缺真实 handler —— 未实现不伪装(拒绝空实现)")
    prev = sorted(registry.list_capabilities()) if registry is not None else []
    registered = False
    if registry is not None:
        try:
            registry.register(_assemble_capability(rec, handler))
            registered = True
        except Exception:  # noqa: BLE001 — 注册面失败不吞 Job, 仅 journal
            registered = False
    rec.status = "implemented"
    rec.handler_implemented = True
    rec.reason = reason or rec.reason
    return RealizationRecord(
        name=rec.name, description=rec.description, parameters=rec.parameters,
        prev_registered=prev, registered=registered, reason=rec.reason,
    )


def _assemble_capability(rec: ProposalRecord, handler):
    """把 approved 提案装成可注册的 Capability (ExternalCapability / 函数包装).

    lazy 拉取(会带出 pydantic 重栈), 由调用方 try 兜底 —— 轻量/沙箱环境无注册表
    时本函数不会被调用, 因此重依赖只在真正要落注册表时才加载。
    """
    from huginn.capabilities.base import Capability, CapabilityResult
    from huginn.capabilities.intents import ExternalCapability
    from huginn.core_types import ToolContext

    if isinstance(handler, Capability):
        return handler

    if callable(handler):
        # 本地子类: 把纯函数 handler 包成 Capability, 供注册表/编排器统一契约使用
        class _FunctorCapability(Capability):
            name = ""
            category = "runtime-proposed"

            def __init__(self, _rec_outer, _fn_outer):
                self.name = _rec_outer.name
                self.description = _rec_outer.description
                self._fn = _fn_outer

            def is_available(self) -> bool:
                return True

            async def _run(self, args, context: ToolContext | None) -> CapabilityResult:
                import inspect
                try:
                    if context is None:
                        context = ToolContext(session_id="capability", workspace=".")
                    eats_ctx = "context" in inspect.signature(self._fn).parameters
                    res = self._fn(args, context) if eats_ctx else self._fn(args)
                    return CapabilityResult.ok(res)
                except Exception as exc:  # noqa: BLE001
                    return CapabilityResult.fail(f"{type(exc).__name__}: {exc}",
                                                 code="execution_error")

        return _FunctorCapability(rec, handler)

    return ExternalCapability(
        rec.name, description=rec.description, backend=handler, read_only=True)


def rollback_realization(rec: RealizationRecord, *, registry=None) -> RollbackRecord:
    """§6 回滚: 依据 Journal 移除新落地能力, 恢复实现前快照.

    - 注册表可用: 精确 unregister 该能力名(只动自己新增的, 不动 prev_registered).
    - 轻量环境: registered=False, 仅移除 journal(移除能力未真正落注册表).
    """
    removed = False
    journal_only = not rec.registered
    if registry is not None and rec.registered:
        try:
            registry.unregister(rec.name)
            removed = True
        except Exception:  # noqa: BLE001
            removed = False
    return RollbackRecord(
        name=rec.name, removed=removed, restored=rec.prev_registered, journal_only=journal_only)


def evolution_run(goal: str, *, handler_store: dict[str, Any] | None = None,
                  registry=None, authorized_by: str = "gate",
                  discoverable: list[str] | None = None) -> dict:
    """§1→§6 闭环一键跑通; 返回 {surface, proposals, confirmed, realized, rollbacks}."""
    surface = diagnose_surface(discoverable=discoverable)
    proposals = propose_capabilities(goal, surface.discoverable)
    confirmed, realized, rollbacks = [], [], []
    for p in proposals:
        rec = create_proposal(**{k: p[k] for k in ("name", "description", "parameters")},
                              reason=p.get("reason", ""))
        confirm_proposal(rec, authorized_by=authorized_by)
        confirmed.append(rec)
        handler = (handler_store or {}).get(rec.name)
        try:
            j = realize_proposal(rec, handler, registry=registry)
            realized.append(j)
        except ValueError as exc:
            # 无 handler -> 保持 approved(未实现), 记入可见提案
            rollbacks.append({"name": rec.name, "error": str(exc)})
    return {
        "surface": surface.to_dict(),
        "proposals": [r.to_dict() for r in confirmed],
        "realized": [r.to_dict() for r in realized],
        "pending_no_handler": rollbacks,
    }