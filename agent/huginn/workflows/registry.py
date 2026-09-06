"""WorkflowRegistry — 工作流集装箱层: 统一注册 / 导出 / 再导入命名工作流.

让"工作流"和 CapabilityRegistry 形成对称的集装箱化能力:
  Workflow ≈ 货柜, WorkflowRegistry ≈ 堆场, export/import ≈ 分享搬运,
  ``workflows-mcp`` (后续) ≈ 码头(把命名工作流暴露给任意 MCP host).

支持三种可封装形态 (kind):
  - stages   : 拓扑模板 (list[ComputationalStage], 走 WorkflowEngine)
  - script   : 动态并行脚本 (WorkflowScript, 走 WorkflowOrchestrator)
  - research : 报告管线 (ResearchWorkflowConfig → DeliAutoResearch)
  - template : 惰性模板占位 (工厂函数 + 参数 schema, 需参才渲染 stages)

分享语义 = 注册 → manifest() 罗列 → export(name) 得单文件 dict →
外地 import_dict() 注册还原 → render_stages/run 复用. 全纯函数/脱网可测.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# 可与 warp 对称的 kind 全集
_KINDS = ("stages", "script", "research", "template")

# ComputationalStage 骨架序列化时要剔除的运行时状态字段
_STAGE_RUNTIME_FIELDS = ("status", "result", "attempts", "started_at", "completed_at")


# ---------------------------------------------------------------------------
# 序列化小工具 (仅取可以跨实例搬运的骨架字段, 剔除运行时状态)
# ---------------------------------------------------------------------------

def _stage_to_dict(stage: Any) -> dict[str, Any]:
    d = asdict(stage) if hasattr(stage, "__dataclass_fields__") else dict(stage)
    for k in _STAGE_RUNTIME_FIELDS:
        d.pop(k, None)
    return d


def _as_stage_dicts(stages: list[Any]) -> list[dict[str, Any]]:
    return [_stage_to_dict(s) for s in stages]


# ---------------------------------------------------------------------------
# 统一封装包
# ---------------------------------------------------------------------------

@dataclass
class WorkflowPackage:
    """一个可命名、可导出、可再导入的工作流封装."""

    name: str
    kind: str
    description: str = ""
    category: str = ""
    stages: list[dict[str, Any]] | None = None
    script: dict[str, Any] | None = None
    research: dict[str, Any] | None = None
    params: dict[str, Any] | None = None       # template 的参数 schema (name->default)
    source: str = ""                            # 来源模块/工厂, 仅供溯源
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"spec": "huginn/workflow 1", **asdict(self)}


# ---------------------------------------------------------------------------
# 集装箱注册表
# ---------------------------------------------------------------------------

class WorkflowRegistry:
    """登记命名工作流, 提供清单 / 导出 / 再导入 / 模板渲染."""

    _packages: dict[str, WorkflowPackage] = {}
    _template_factories: dict[str, Callable] = {}

    # ---- 注册 ----

    @classmethod
    def register_package(cls, pkg: WorkflowPackage) -> str:
        if pkg.kind not in _KINDS:
            raise ValueError(f"未知工作流 kind: {pkg.kind} (可选 {_KINDS})")
        cls._packages[pkg.name] = pkg
        return pkg.name

    @classmethod
    def register_stages(
        cls,
        name: str,
        stages: list[Any],
        description: str = "",
        category: str = "",
        **meta: Any,
    ) -> str:
        """注册一个已渲染的拓扑工作流 (list[ComputationalStage])."""
        return cls.register_package(WorkflowPackage(
            name=name, kind="stages", description=description, category=category,
            stages=_as_stage_dicts(stages), **({"meta": meta} if meta else {}),
        ))

    @classmethod
    def register_script(
        cls, name: str, script: Any, description: str = "", category: str = ""
    ) -> str:
        """注册一个动态并行脚本 (WorkflowScript 或等价 dict)."""
        script_dict = script if isinstance(script, dict) else {
            "id": getattr(script, "id", None),
            "objective": getattr(script, "objective", ""),
            "max_concurrent": getattr(script, "max_concurrent", 8),
            "subtasks": [
                {"id": s.id, "tool": s.tool_name, "args": s.args,
                 "description": getattr(s, "description", "")}
                for s in getattr(script, "subtasks", [])
            ],
        }
        return cls.register_package(WorkflowPackage(
            name=name, kind="script", description=description, category=category,
            script=script_dict,
        ))

    @classmethod
    def register_template(
        cls, name: str, factory: Callable, description: str = "", category: str = ""
    ) -> str:
        """注册一个惰性模板 (工厂函数, 运行时才渲染 stages).

        参数 schema 从 factory 签名提取 (参数名->默认值), 供调用方知道要填什么.
        """
        sig = inspect.signature(factory)
        params: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            params[pname] = None if p.default is inspect.Parameter.empty else p.default
        cls._template_factories[name] = factory
        return cls.register_package(WorkflowPackage(
            name=name, kind="template", description=description, category=category,
            params=params, source=f"{factory.__module__}.{factory.__qualname__}",
        ))

    # ---- 清单 / 读取 ----

    @classmethod
    def manifest(cls) -> list[dict[str, Any]]:
        """统一清单: 每个工作流的 name/kind/category/stages数/subtask数."""
        out = []
        for name in sorted(cls._packages):
            pkg = cls._packages[name]
            out.append({
                "name": pkg.name,
                "kind": pkg.kind,
                "category": pkg.category,
                "description": pkg.description,
                "n_stages": len(pkg.stages or []),
                "n_subtasks": len((pkg.script or {}).get("subtasks", [])),
                "params": sorted((pkg.params or {}).keys()),
            })
        return out

    @classmethod
    def get(cls, name: str) -> WorkflowPackage | None:
        return cls._packages.get(name)

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._packages)

    # ---- 导出 / 再导入 (分享) ----

    @classmethod
    def export(cls, name: str) -> dict[str, Any]:
        """导出单文件交换格式 (spec + 完整骨架), 无运行时状态."""
        pkg = cls.get(name)
        if pkg is None:
            raise KeyError(f"工作流 '{name}' 未注册")
        return pkg.to_dict()

    @classmethod
    def import_dict(cls, data: dict[str, Any]) -> str:
        """从一个导出 dict 还原并注册, 返回注册名 (幂等: 同名覆盖)."""
        if data.get("spec") != "huginn/workflow 1":
            raise ValueError("非本格式的工作流导出包")
        return cls.register_package(WorkflowPackage(
            name=str(data["name"]),
            kind=str(data["kind"]),
            description=str(data.get("description", "")),
            category=str(data.get("category", "")),
            stages=data.get("stages"),
            script=data.get("script"),
            research=data.get("research"),
            params=data.get("params"),
            source=str(data.get("source", "import")),
            meta=data.get("meta") or {},
        ))

    @classmethod
    def export_all(cls) -> list[dict[str, Any]]:
        """批量导出全部注册工作流 (用于整体备份/分享)."""
        return [cls.export(n) for n in cls.list_names()]

    # ---- 执行 (仅模板渲染, 真正的 run 交给 WorkflowEngine/Orchestrator) ----

    @classmethod
    def render_stages(cls, name: str, **params: Any) -> list[dict[str, Any]]:
        """调模板工厂渲染 stages; 非模板/失败抛 ValueError."""
        factory = cls._template_factories.get(name)
        if factory is None:
            pkg = cls.get(name)
            if pkg and pkg.kind == "stages":
                return pkg.stages or []
            raise ValueError(f"'{name}' 不是可渲染的模板或 stages 工作流")
        return _as_stage_dicts(factory(**params))

    @classmethod
    def register_builtin_templates(cls) -> list[str]:
        """扫描 workflows.templates 里全部 *_workflow 工厂并注册为惰性模板."""
        mod = importlib.import_module("huginn.workflows.templates")
        names: list[str] = []
        for mname, fn in inspect.getmembers(mod, inspect.isfunction):
            if mname.endswith("_workflow"):
                names.append(cls.register_template(
                    mname, fn,
                    description=((fn.__doc__ or "").strip().splitlines() or [mname])[0],
                ))
        return names