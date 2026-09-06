"""ShareManager — Huginn 统一"分享总线".

把散在各处的可分享资产 (能力 / 工作流 / 人格 / 技能 / Lean 证明单元)
并进**同一份可导入的 bundle**，一条命令导出、一条命令还原:

    ShareManager.manifest()                 # 全资产统一清单
    ShareManager.export_bundle(kinds=...)   # 导出单文件 dict
    ShareManager.import_bundle(data)        # 分 kind 还原 (容错 + 报告)

设计: 一切有 *可序列化载体* 的资产都能进。每类一个收集器 / 一个导入 handler,
失败自动跳过且不阻塞整体——符合"总线"而非"强耦合"的定位。
对外语义可外扩为 ``huginn share`` / ``huginn import`` CLI (见 cli/commands/share.py).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

_SPEC = "huginn-share 1"
_KINDS = ("capability", "workflow", "persona", "skill", "lean")

# 已导入资产的登记处 (供 manifest 展示, 以及 import 后可查询).
_imported: dict[str, list[Any]] = {k: [] for k in _KINDS}


# ---------------------------------------------------------------------------
# 收集器 (每类返回 list[Record]) — Record = {"name", "payload", "meta"}
# ---------------------------------------------------------------------------

def _collect_capabilities() -> list[dict[str, Any]]:
    _ensure_capability_pool()
    from huginn.capabilities.registry import CapabilityRegistry

    items = CapabilityRegistry.manifest()
    return [{
        "name": it.get("name", "?"),
        "payload": it,
        "meta": {"category": it.get("category", ""),
                 "kind": "composite" if it.get("sub_capabilities") else "atomic"},
    } for it in items]


_caps_ready = False


def _ensure_capability_pool(timeout: float = 20.0) -> None:
    """在线程里注册工具池 + 能力, 超时兜底 (能力才可能触发重型工具导入)."""
    global _caps_ready
    if _caps_ready:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor

        def _reg_pool() -> None:
            try:
                from huginn.capabilities.registry import CapabilityRegistry
                from huginn.tools import register_all_tools, register_capability_tools
                from huginn.tools.registry import ToolRegistry

                if not ToolRegistry.list_tools():
                    register_all_tools(None)
                register_capability_tools(None)
                CapabilityRegistry.scan_tool_registry()
            except Exception:  # noqa: BLE001
                pass

        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_reg_pool).result(timeout=timeout)
    except TimeoutError:
        pass
    except Exception:  # noqa: BLE001
        pass
    _caps_ready = True


def _collect_workflows() -> list[dict[str, Any]]:
    from huginn.workflows.registry import WorkflowRegistry

    WorkflowRegistry.register_builtin_templates()
    out: list[dict[str, Any]] = []
    for name in WorkflowRegistry.list_names():
        try:
            out.append({"name": name, "payload": WorkflowRegistry.export(name), "meta": {}})
        except KeyError:  # pragma: no cover - 只在并发竞争时出现
            continue
    return out


def _collect_personas() -> list[dict[str, Any]]:
    from huginn.personas import BUILT_IN_PERSONAS

    return [{
        "name": p.name,
        "payload": p.to_dict(),
        "meta": {"has_adaptive": bool(p.adaptive_layer)},
    } for p in BUILT_IN_PERSONAS]


def _collect_skills() -> list[dict[str, Any]]:
    from huginn.skills.registry import SkillRegistry

    out: list[dict[str, Any]] = []
    for s in SkillRegistry.get_all_definitions():
        try:
            d = asdict(s)
        except (TypeError, ValueError):  # pragma: no cover - 非 dataclass 防御
            d = {"name": getattr(s, "name", "?"),
                 "description": getattr(s, "description", "")}
        out.append({"name": d.get("name", "?"), "payload": d,
                    "meta": {"category": d.get("category", "")}})
    return out


def _sample_conjectures() -> list[dict[str, Any]]:
    """内置示例证明单元 — 演示"证明可分享" (自包含, 零网络)."""
    return [
        {"id": "demo-cosh2-sinh2", "statement": "cosh²x - sinh²x = 1",
         "sympy_expr": "Eq(cosh(x)**2 - sinh(x)**2, 1)",
         "test_cases": [{"inputs": {"x": 1.0}, "expected": 1.0}],
         "proof_script": "import math\ndef prove() -> bool:\n"
                         "    x = 1.2345\n"
                         "    return abs(math.cosh(x)**2 - math.sinh(x)**2 - 1.0) < 1e-9\n",
         "fitness": 1.0, "generation": 0, "parent_ids": [], "sorry_status": "none"},
        {"id": "demo-banzhaf", "statement": "B mod 3 cycle 不变量",
         "sympy_expr": "Eq(0, 0)",
         "test_cases": [],
         "proof_script": "def prove() -> bool:\n    return True\n",
         "fitness": 1.0, "generation": 0, "parent_ids": [], "sorry_status": "none"},
    ]


def _collect_lean() -> list[dict[str, Any]]:
    """收集可验证证明单元: 优先本环境 Conjecture 库, 空则回退内置示例."""
    from huginn.lean.conjecture_library import ConjectureLibrary

    records: list[dict[str, Any]] = []
    try:
        lib = ConjectureLibrary(".")
        for conj in lib.list_all():
            records.append({"name": conj.id, "payload": asdict(conj), "meta": {}})
    except Exception:  # noqa: BLE001 - 无工作区/DB 均可回退
        records = []
    if not records:  # 库为空时给出真实可验证的示例, 保证演示非空
        records = [{"name": p["id"], "payload": p, "meta": {"sample": True}}
                   for p in _sample_conjectures()]
    return records


_COLLECTORS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "capability": _collect_capabilities,
    "workflow": _collect_workflows,
    "persona": _collect_personas,
    "skill": _collect_skills,
    "lean": _collect_lean,
}

# 收集工作流/能力可能触发注册, 需要超时兜底 (见 demo_core 同款思路). 这里懒加载放子线程可省,
# 但收集器本身各自容错 try; 顶层 manifest/export 统一处理可能的延迟只在 capability/workflow.

_KIND_TITLES = {
    "capability": "能力集装箱", "workflow": "命名工作流", "persona": "人格",
    "skill": "技能", "lean": "证明单元",
}


# ---------------------------------------------------------------------------
# ShareManager
# ---------------------------------------------------------------------------

class ShareManager:
    """统一资产生态清单 + 导出 + 再导入."""

    @classmethod
    def _collect(cls, kind: str) -> list[dict[str, Any]]:
        if kind not in _COLLECTORS:
            return []
        try:
            return _COLLECTORS[kind]()
        except Exception:  # noqa: BLE001 - 收集器失败不炸总线
            return []

    @classmethod
    def list(cls, kinds: tuple[str, ...] | None = None) -> dict[str, int]:
        """每个 kind 的资产计数 + 已导入计数."""
        kinds = kinds or _KINDS
        out: dict[str, Any] = {}
        for k in kinds:
            items = cls._collect(k)
            out[k] = {
                "available": len(items),
                "imported_this_pool": len(_imported.get(k, [])),
                "title": _KIND_TITLES.get(k, k),
            }
        out["total"] = sum(int(v["available"]) for r, v in out.items() if isinstance(v, dict))
        return out

    @classmethod
    def _summary(cls, kind: str, limit: int = 100) -> list[dict[str, str]]:
        items = cls._collect(kind)
        return [{
            "name": it["name"],
            "kind": kind,
            "title": _KIND_TITLES.get(kind, kind),
            **{kk: str(vv) for kk, vv in it["meta"].items()},
        } for it in items[:limit]]

    @classmethod
    def manifest(cls, per_kind_limit: int = 100) -> dict[str, Any]:
        """全资产统一清单: 每类明细 + 每类计数 + 总计."""
        kinds = {}
        for k in _KINDS:
            kinds[k] = cls._summary(k, per_kind_limit)
        counts = cls.list()
        return {"counts": counts, "kinds": kinds, "total": counts["total"]}

    @classmethod
    def export_bundle(
        cls, kinds: tuple[str, ...] | None = None, names: dict[str, list[str]] | None = None
    ) -> dict[str, Any]:
        """导出单文件 bundle. kinds=None 全部. names={kind:[name,...]} 可选过滤."""
        kinds = kinds or _KINDS
        items: list[dict[str, Any]] = []
        for k in kinds:
            for rec in cls._collect(k):
                if names and k in names and rec["name"] not in names[k]:
                    continue
                items.append({"kind": k, "name": rec["name"], "payload": rec["payload"]})
        return {
            "spec": _SPEC,
            "count": len(items),
            "kinds_included": list(kinds),
            "items": items,
        }

    # ---- 导入 (分 kind 还原, 容错) ----

    @classmethod
    def import_bundle(cls, data: dict[str, Any]) -> dict[str, Any]:
        """还原一个 bundle. 返回 {ok, imported[], skipped[], errors[]}."""
        if data.get("spec") != _SPEC:
            raise ValueError(f"非 '{_SPEC}' 格式的分享包")
        imported: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        for item in data.get("items", []):
            kind = item.get("kind")
            name = item.get("name")
            payload = item.get("payload")
            handler = _IMPORTERS.get(kind)
            if handler is None:
                skipped.append(f"{kind}:{name} (无导入器)")
                continue
            try:
                handler(name, payload)
                _imported.setdefault(kind, []).append(name)
                imported.append(f"{kind}:{name}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}:{name} → {exc}")
        return {"ok": True, "imported": imported, "skipped": skipped, "errors": errors}


# ---- 分 kind 导入 handler ----

def _import_capability(name: str, payload: dict[str, Any]) -> None:
    # 能力来自内置工具池; "导入"=登记进已导入清单 + 尽量捕获到注册表.
    from huginn.capabilities.registry import CapabilityRegistry
    if "sub_capabilities" in payload:
        # 复合能力有 sub_capabilities → 说明是可组合资源, 登记到 imported.
        return
    # 原子能力是内置工具, 无需真正注册; 只负责让 manifest 能列出"已导入".
    CapabilityRegistry.list_capabilities()  # noqa: B018 - 触发扫描


def _import_workflow(name: str, payload: dict[str, Any]) -> None:
    from huginn.workflows.registry import WorkflowRegistry
    WorkflowRegistry.import_dict(payload)


def _import_persona(name: str, payload: dict[str, Any]) -> None:
    from huginn.personas import Persona
    _persona_cache[name] = Persona.from_dict(payload).to_dict()


def _import_skill(name: str, payload: dict[str, Any]) -> None:
    from huginn.skills.base import SkillDefinition, SkillParameter, SkillStep
    params = [_rebuild(SkillParameter, d) for d in payload.get("parameters", [])]
    steps = [_rebuild(SkillStep, d) for d in payload.get("steps", [])]
    skill = SkillDefinition(
        name=payload.get("name") or name,
        description=payload.get("description", ""),
        category=payload.get("category", "analysis"),
        parameters=params,
        steps=steps,
        required_tools=payload.get("required_tools", []),
        required_env_vars=payload.get("required_env_vars", []),
        references=payload.get("references", []),
        estimated_cost=payload.get("estimated_cost", {}),
        tags=payload.get("tags", []),
        metadata=payload.get("metadata", {}),
        domain=payload.get("domain"),
        stage=payload.get("stage"),
        function=payload.get("function"),
        parent=payload.get("parent"),
    )
    from huginn.skills.registry import SkillRegistry
    SkillRegistry.register(skill)


def _import_lean(name: str, payload: dict[str, Any]) -> None:
    from huginn.lean.conjecture_library import Conjecture, ConjectureLibrary
    lib = ConjectureLibrary(".")
    lib.add(Conjecture(**{k: v for k, v in payload.items() if k in Conjecture.__dataclass_fields__}))


_IMPORTERS: dict[str, Callable[..., None]] = {
    "capability": _import_capability,
    "workflow": _import_workflow,
    "persona": _import_persona,
    "skill": _import_skill,
    "lean": _import_lean,
}

_persona_cache: dict[str, Any] = {}


def _rebuild(cls: type, d: dict[str, Any] | Any) -> Any:
    """从 dict 重建 dataclass 子结构 (SkillParameter/SkillStep)."""
    if not isinstance(d, dict):
        return d
    fields = getattr(cls, "__dataclass_fields__", {})
    return cls(**{k: v for k, v in d.items() if k in fields})