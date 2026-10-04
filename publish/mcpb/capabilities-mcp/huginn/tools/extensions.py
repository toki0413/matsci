"""运行时工具注入 —— Pi 式"自扩展"原语的基础设施层.

哲学对齐 Pi: 不预装一整个工具动物园, 而是让模型缺什么就**自己写一个**工具,
``ToolRegistry.register`` 热注 + schema 缓存失效, 下一步就能被调用.
本模块是底层机制; 模型侧入口是 :mod:`huginn.tools.make_tool`。

契约:
  - 模型提供 snake_case 工具名 + Python 源码 (定义 HuginnTool 子类);
  - 校验: 名字模式 // 源码可编译 // 确有 HuginnTool 子类 // 声明的 ``name`` 与请求一致;
  - 持久化到运行时扩展目录 (重启后仍可被 :func:`scan_extensions` 收回);
  - fail-open: 校验失败不注册, 返回明确错误, 不破坏主流程。

诚实边界: make_tool 授权模型写**可执行代码**. 这与 hugging 里其他原语 (bash_tool/
code_tool) 同等级别的能力, 由既有权限门禁 (check_permissions) 约束; 本模块不做
"更安全的 python", 只做名字/可编译性/类属校验 + 明确落盘位置, 便于审计.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry
from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

# 工具名: 小写 snake_case, 长度 2-64, 只允许 [a-z0-9_]
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
# 类名: PascalCase
_CLASS_RE = re.compile(r"^[A-Z][A-Za-z0-9_]{0,127}$")

# 显式注册过的扩展工具名 (distinguish from 官方工具, 供 list/审计)
_EXTENSION_NAMES: set[str] = set()
_EXTENSIONS_DIR: Path | None = None


def extensions_dir(base: Path | None = None) -> Path:
    """运行时扩展目录 (默认 ~/.huginn/extensions, 可用 base 覆盖)."""
    global _EXTENSIONS_DIR
    if _EXTENSIONS_DIR is None:
        _EXTENSIONS_DIR = (base or get_runtime_home()) / "extensions"
        _EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _EXTENSIONS_DIR


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid tool name {name!r}: use lower snake_case, 2-64 chars [a-z0-9_]"
        )
    return name


def _find_tool_class(module: Any, want_name: str) -> type[HuginnTool]:
    """在模块里找一个 HuginnTool 子类, 且其 ``name`` 与请求一致."""
    matches: list[type[HuginnTool]] = []
    for _, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, HuginnTool) and obj is not HuginnTool:
            matches.append(obj)
    if not matches:
        raise ValueError(
            "source defines no HuginnTool subclass; make sure your code declares "
            "`class MyTool(HuginnTool)` with a `name` and a `call`/`_execute`"
        )
    for cls in matches:
        declared = (cls.name or "").strip()
        if declared == want_name:
            return cls
    # 未精匹配: 若只有一个候选且其 name 为空 → 直接采用请求名
    if len(matches) == 1 and not (matches[0].name or "").strip():
        return matches[0]
    found = ", ".join(c.name or "(unnamed)" for c in matches)
    raise ValueError(f"source declares name(s) {found!r}, expected {want_name!r}")


def _load_module(name: str, source: str, dirpath: Path) -> Any:
    """把源码落盘并用 importlib 装载 (sys.modules 复用 + 覆盖)."""
    module_name = f"_huginn_ext_{name}"
    path = dirpath / f"{name}_tool.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot create module spec for {name!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — 模型代码异常展开给调用方, 由 make_tool 转 ToolResult
        raise ValueError(f"extension module failed to import: {type(exc).__name__}: {exc}") from exc
    return mod


def build_extension(
    name: str,
    code: str,
    *,
    category: str = "self",
    description: str | None = None,
    base: Path | None = None,
) -> dict[str, Any]:
    """编译 + 落盘 + 导入 + 实例化 + 热注册一个模型自写的工具.

    Returns: {"registered", "name", "class", "module", "path", "active"}
    Raises: ValueError (名字/可编译/类属/名称不一致) → 不注册.
    """
    name = _validate_name(name)
    source = (code or "").strip()
    if not source:
        raise ValueError("code must not be empty")

    # 1) 先编译 (AST/compile 校验语法), 语法错在写入前就暴露; SyntaxError 归一成
    #    ValueError, 让 make_tool 的统一 except ValueError 覆盖.
    try:
        compile(source, f"<extension:{name}>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"extension source syntax error: {exc.msg} (line {exc.lineno})") from exc

    dirpath = extensions_dir(base)
    mod = _load_module(name, source, dirpath)
    cls = _find_tool_class(mod, name)

    tool = cls()
    # 统一 name/category/description 到请求值 (description/category 允许实例覆盖)
    tool.name = name
    if category:
        tool.category = str(category)
    if description:
        tool.description = str(description)

    ToolRegistry.register(tool)   # 注册即 _schemas_cache=None → 下一步 LLM 可见
    _EXTENSION_NAMES.add(name)
    path = dirpath / f"{name}_tool.py"
    return {
        "registered": True,
        "name": name,
        "class": type(tool).__name__,
        "module": path.name,
        "path": str(path),
        "active": tool.active,
    }


def list_extensions() -> list[dict[str, Any]]:
    """已注册的扩展工具 + 基本元数据."""
    out: list[dict[str, Any]] = []
    for name in sorted(_EXTENSION_NAMES):
        tool = ToolRegistry.get(name)
        out.append({
            "name": name,
            "registered": tool is not None,
            "class": type(tool).__name__ if tool else "",
            "active": tool.active if tool else False,
        })
    return out


def scan_extensions(base: Path | None = None) -> list[dict[str, Any]]:
    """重启后收回扩展目录里已落盘的工具 (幂等: 已注册跳过).

    供 server 启动时把上一 session 造的扩展恢复进 registry.
    """
    dirpath = extensions_dir(base)
    restored: list[dict[str, Any]] = []
    for f in sorted(dirpath.glob("*_tool.py")):
        name = f.name.removesuffix("_tool.py")
        if ToolRegistry.get(name) is not None:
            continue
        try:
            mod = _load_module(name, f.read_text(encoding="utf-8"), dirpath)
            cls = _find_tool_class(mod, name)
            tool = cls()
            tool.name = name
            ToolRegistry.register(tool)
            _EXTENSION_NAMES.add(name)
            restored.append({"name": name, "restored": True})
        except Exception:  # noqa: BLE001 — 单一坏扩展不影响其余恢复
            logger.warning("failed to restore extension %r", name, exc_info=True)
    return restored


__all__ = [
    "build_extension", "list_extensions", "scan_extensions",
    "extensions_dir", "_EXTENSION_NAMES",
]
