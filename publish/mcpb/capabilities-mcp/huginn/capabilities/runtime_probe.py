"""能力可运行性面 (runtime_probe): L0 依赖探针 + 报告组装.

与 introspection 的分工: introspection 管**名字面** (工具是否登记/可见),
本模块管**依赖面** (依赖是否可加载 → 能力能否真跑)。纯静态扫描 + 单次
``find_spec``, 不 import 工具模块本体, 保证 weightless、可注入、可测、确定性。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# huginn/capabilities/runtime_probe.py -> parents[2] == agent/.
AGENT_ROOT = Path(__file__).resolve().parents[2]

STATUS_RUNNABLE = "runnable"
STATUS_MISSING = "missing_deps"
STATUS_NOT_INSTANTIABLE = "not_instantiable"
STATUS_HIDDEN = "hidden"

# 全部就绪时注入 context 的一行 (空串表示"不注入", 省 token).
ALL_OK_LINE = "Tool capabilities: all runnable."


@dataclass(frozen=True)
class ToolSpec:
    """一条待探测声明: (模块, 类, 可选源码路径)."""

    module: str
    cls: str
    path: Path | None = None
    kind: str = "tool"


@dataclass
class ProbeResult:
    name: str
    kind: str
    status: str
    missing_deps: list[str] = field(default_factory=list)
    install_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "missing_deps": list(self.missing_deps),
            "install_hint": self.install_hint,
        }


@dataclass
class CapabilityRuntimeReport:
    l0: list[ProbeResult] = field(default_factory=list)
    l1: list = field(default_factory=list)  # list[SmokeResult], 避免循环导入

    def degraded(self) -> list[ProbeResult]:
        return [r for r in self.l0 if r.status != STATUS_RUNNABLE]


def _filter_deps(deps: set[str]) -> list[str]:
    """去掉 stdlib 与 huginn 内部导入, 只留第三方依赖名 (排序)."""
    stdlib = sys.stdlib_module_names
    return sorted(d for d in deps if not d.startswith("huginn") and d not in stdlib)


def _scan(path: Path | None) -> tuple[bool, list[str]]:
    """静态扫描: 返回 (是否可解析, 缺失的第三方依赖)."""
    if path is None or not path.is_file():
        return False, []
    try:
        from huginn.tools import _collect_top_level_imports  # 惰性, 免包级循环导入

        deps = _filter_deps(_collect_top_level_imports(str(path)))
    except Exception:  # 非致命: 解析失败 = 无信息
        logger.debug("dep scan failed: %s", path, exc_info=True)
        return False, []
    missing = [d for d in deps if importlib.util.find_spec(d) is None]
    return True, missing


def classify(
    *,
    missing: list[str],
    imported: bool = True,
    visible: bool | None = None,
    constructed: bool | None = None,
) -> str:
    """纯判据: 由事实推 status. 顺序即优先级."""
    if not imported or missing:
        return STATUS_MISSING
    if constructed is False:
        return STATUS_NOT_INSTANTIABLE
    if visible is False:
        return STATUS_HIDDEN
    return STATUS_RUNNABLE


# 少数 import 名与其 PyPI 分发名不一致 —— 直接拼 `pip install <import 名>` 会装错包
# 或找不到包. 只登记本仓库工具集里确认的映射; 未登记的按 pip 的归一化规则
# (下划线/连字符/点号等价) 原样输出.
_IMPORT_TO_DIST: dict[str, str] = {
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "mace": "mace-torch",
    "fairchem": "fairchem-core",
    "tavily": "tavily-python",
    "z3": "z3-solver",
}


def _dist_name(import_name: str) -> str:
    return _IMPORT_TO_DIST.get(import_name, import_name)


def install_hint(missing: list[str]) -> str:
    if not missing:
        return ""
    return f"pip install {' '.join(_dist_name(m) for m in missing)}"


def _module_path(module: str) -> Path | None:
    rel = module.replace(".", "/")
    for cand in (AGENT_ROOT / f"{rel}.py", AGENT_ROOT / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def default_tool_specs() -> list[ToolSpec]:
    """从既有 builtin 工具清单派生 spec (惰性 import, 只取名字 + 静态解析路径)."""
    from huginn.tools import _CORE_MODULES, _OPTIONAL_MODULES

    return [
        ToolSpec(module=m, cls=c, path=_module_path(m))
        for m, c in (*_CORE_MODULES, *_OPTIONAL_MODULES)
    ]


def probe_runtime(
    specs: list[ToolSpec] | None = None,
    *,
    available_names: set[str] | None = None,
) -> list[ProbeResult]:
    """对每条 spec 探测依赖可运行性. 任何异常收敛为 status, 绝不抛."""
    specs = specs if specs is not None else default_tool_specs()
    out: list[ProbeResult] = []
    for spec in specs:
        try:
            imported, missing = _scan(spec.path)
            visible = None if available_names is None else (spec.cls in available_names)
            out.append(
                ProbeResult(
                    name=spec.cls,
                    kind=spec.kind,
                    status=classify(
                        missing=missing, imported=imported, visible=visible
                    ),
                    missing_deps=missing,
                    install_hint=install_hint(missing),
                )
            )
        except Exception:  # 探针自身绝不抛
            logger.debug("probe failed for %s", spec, exc_info=True)
            out.append(
                ProbeResult(name=spec.cls, kind=spec.kind, status=STATUS_MISSING)
            )
    return sorted(out, key=lambda r: (r.kind, r.name))


def build_report(
    l0: list[ProbeResult] | None = None, l1: list | None = None
) -> CapabilityRuntimeReport:
    return CapabilityRuntimeReport(l0=list(l0 or []), l1=list(l1 or []))


def summary_line(report: CapabilityRuntimeReport) -> str:
    """一行汇总, 供注入 system prompt. 全就绪时返回 ALL_OK_LINE."""
    bad = report.degraded()
    if not bad:
        return ALL_OK_LINE
    deps = sorted({d for r in bad for d in r.missing_deps})
    names = ", ".join(r.name for r in bad[:8])
    more = "" if len(bad) <= 8 else f" (+{len(bad) - 8})"
    hint = install_hint(deps)
    tail = f" Install with: {hint}." if hint else ""
    return f"Tool capabilities degraded: {len(bad)} ({names}{more}).{tail}"


def capability_note() -> str:
    """能力面汇总块 (供注入 system prompt): 有降级能力时返回一段, 否则空串.

    每次调用重算 (不缓存), 保证装完依赖后不注入陈旧状态 (AC-4)。
    绝不抛 —— 探测失败回落空串 (不注入), 保证调用方永不崩 (fail-open)。
    诚实边界: 只陈述「哪些能力因缺依赖不可用、可 pip 装什么」, 不暗示能力已可用。
    """
    try:
        line = summary_line(build_report(probe_runtime()))
    except Exception:  # 非致命: 能力盘点失败则不注入
        logger.debug("capability runtime probe failed", exc_info=True)
        return ""
    if line == ALL_OK_LINE:
        return ""  # 全就绪不注入, 省 token
    return f"\n\n# Tool Capabilities\n{line}"
