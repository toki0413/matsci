"""模型面契约门 —— 表示层语义 + 组装结构不变量 + 披露双向核对.

为什么有这扇门: 模型面此前是字符串, 各组装根各自 `+=`, 段名在拼接处丢失,
于是门只能断言「生产者声明了什么」, 不能断言「模型实际收到什么」. 本门把
表示层 (model_surface) 的语义与主路径的组装结构固定下来, 并核对 Catalog 的
披露声明是否真的落到文本里.

离线部分只依赖标准库; 运行时部分 importorskip langchain.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from huginn.agent.mechanism_census import KIND_DISCLOSURE, census_text, disclosure_level, disclosure_reason
from huginn.agent.model_surface import Surface, SurfaceSection, compose

_AGENT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONTEXT = _AGENT_ROOT / "huginn" / "agent" / "context.py"
_CENSUS = _AGENT_ROOT / "huginn" / "agent" / "mechanism_census.py"

# 主路径允许的段来源 (新增来源须在此登记, 否则视为未申报的接线).
_ALLOWED_SOURCES = frozenset({
    "context.mode_prompt",
    "prompt_segments",
    "context.system",
    "context.project",
    "context.project_memory",
    "personalization",
    "runtime.backends",
    "capabilities.runtime_probe",
    "memory.stable_principles",
    "catalog",
    "phases",
    "prompt_builder",
})


def _class_method(tree: ast.Module, cls_name: str, method: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
    raise AssertionError(f"{cls_name}.{method} not found")


def _module_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# ── 表示层语义 ────────────────────────────────────────────────────────────

def test_render_joins_in_declared_order() -> None:
    surface = compose([
        SurfaceSection("a", "src:a", "first"),
        SurfaceSection("b", "src:b", "second"),
    ])
    assert surface.render() == "first\n\nsecond"
    assert surface.names() == ("a", "b")
    assert surface.char_count() == len("first\n\nsecond")


def test_empty_sections_are_absent_not_empty_strings() -> None:
    surface = compose([
        SurfaceSection("a", "src:a", "x"),
        SurfaceSection("empty", "src:e", ""),
        SurfaceSection("b", "src:b", "y"),
    ])
    assert surface.text_of("empty") == ""
    assert surface.render() == "x\n\ny"


def test_duplicate_section_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate surface section"):
        compose([
            SurfaceSection("dup", "src:a", "1"),
            SurfaceSection("dup", "src:b", "2"),
        ])


def test_section_requires_name_and_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SurfaceSection("", "src", "text")
    with pytest.raises(ValueError, match="missing source"):
        SurfaceSection("name", " ", "text")


def test_sources_are_deduped_and_ordered() -> None:
    surface = compose([
        SurfaceSection("a", "x", "1"),
        SurfaceSection("b", "y", "2"),
        SurfaceSection("c", "x", "3"),
    ])
    assert surface.sources() == ("x", "y")


# ── 组装结构不变量 ────────────────────────────────────────────────────────

def test_main_path_composes_surface_without_string_concat() -> None:
    tree = ast.parse(_CONTEXT.read_text(encoding="utf-8"))
    compose_fn = _module_function(tree, "compose_main_surface")
    render_fn = _class_method(tree, "ContextMixin", "_effective_system_prompt")

    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "compose"
        for n in ast.walk(compose_fn)
    ), "compose_main_surface 必须通过 compose() 组装; 直接拼字符串会让段名丢失"
    assert not any(isinstance(n, ast.AugAssign) for n in ast.walk(compose_fn)), (
        "compose_main_surface 内不得用 += 拼 prompt; 拼接权必须收归 Surface.render()"
    )
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "render"
        for n in ast.walk(render_fn)
    ), "_effective_system_prompt 必须调用 Surface.render()"
    assert not any(isinstance(n, ast.AugAssign) for n in ast.walk(render_fn)), (
        "_effective_system_prompt 内不得拼接字符串"
    )


# ── 披露双向核对 ──────────────────────────────────────────────────────────

def test_disclosure_table_covers_all_catalog_kinds() -> None:
    from huginn.catalog.models import KINDS

    missing = [kind for kind in KINDS if kind not in KIND_DISCLOSURE]
    assert not missing, f"Catalog kind 缺披露声明 (默认不可见 = 复发): {missing}"


def test_hidden_kinds_must_state_a_reason() -> None:
    for kind in KIND_DISCLOSURE:
        if disclosure_level(kind) == "hidden":
            assert disclosure_reason(kind), f"hidden kind {kind!r} 必须写理由"


def test_always_kinds_are_named_in_census_text() -> None:
    text = census_text()
    if not text:
        pytest.skip("Catalog 不可用或无条目, 该段缺席 (fail-open)")
    from huginn.catalog.manager import CatalogManager

    entries = CatalogManager().discover_all()
    for kind in KIND_DISCLOSURE:
        if disclosure_level(kind) != "always":
            continue
        names = sorted(e.name for e in entries if e.kind == kind and e.enabled)
        if not names:
            continue
        assert names[0] in text, f"always 级 kind {kind!r} 的条目未出现在普查段"


# ── 运行时: 真实主路径 ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def main_surface() -> Surface:
    pytest.importorskip("langchain.agents.middleware")
    from huginn.agent.context import ContextMixin

    class _PM:
        def prompt_prefix(self) -> str:
            return ""

        def tool_filter(self):
            return None

    class _Ctx(ContextMixin):
        def __init__(self) -> None:
            self.system_prompt = "base prompt"
            self._mode = "default"
            self.workspace = "/tmp"
            self._csm = None
            self._phase_manager = _PM()

    return _Ctx()._effective_surface()


def test_runtime_surface_declares_catalog_source(main_surface: Surface) -> None:
    assert "catalog" in main_surface.sources(), "主路径未登记 catalog 披露段"


def test_runtime_sources_are_all_declared(main_surface: Surface) -> None:
    undeclared = sorted(set(main_surface.sources()) - _ALLOWED_SOURCES)
    assert not undeclared, f"主路径出现未申报的段来源: {undeclared}"
