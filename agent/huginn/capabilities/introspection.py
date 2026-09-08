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