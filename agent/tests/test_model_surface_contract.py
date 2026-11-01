# governance-gate
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

from huginn.agent.mechanism_census import (
    KIND_DISCLOSURE,
    census_text,
    disclosure_level,
    disclosure_reason,
)
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


# ── Surface 段名台账 (改名/删除即红) ─────────────────────────────────────

def _surface_section_literals(tree: ast.Module) -> list[tuple[str, str]]:
    """收集 `SurfaceSection("<name>", "<source>", ...)` 的常量字面量."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SurfaceSection"
        ):
            continue
        if len(node.args) < 2:
            continue
        name, source = node.args[0], node.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str) \
                and isinstance(source, ast.Constant) and isinstance(source.value, str):
            out.append((name.value, source.value))
    return out


def test_every_literal_surface_section_is_registered() -> None:
    """段名/来源必须登记在 SURFACE_SPECS; 改名或新增未登记 → 红."""
    from huginn.agent.context import SURFACE_SPECS

    literals = _surface_section_literals(ast.parse(_CONTEXT.read_text(encoding="utf-8")))
    fixed = [(n, s) for n, s in literals if s != "prompt_segments"]
    for name, source in fixed:
        assert name in SURFACE_SPECS, (
            f"段 {name!r} 未在 context.SURFACE_SPECS 登记 —— 改名/新增段必须显式登记"
        )
        declared = SURFACE_SPECS[name][0]
        assert declared == source, (
            f"段 {name!r} 来源漂移: 源码 {source!r} vs 台账 {declared!r}"
        )
    produced = {n for n, _ in fixed}
    stale = set(SURFACE_SPECS) - produced
    assert not stale, f"台账有而源里已无生产点: {sorted(stale)} —— 段被删? 同步 SURFACE_SPECS."


def test_required_surface_sections_are_unconditional() -> None:
    """标 required 的段必须在 compose_main_surface 的基础段列表里无条件登记."""
    from huginn.agent.context import SURFACE_SPECS

    required = {n for n, (_src, req) in SURFACE_SPECS.items() if req}
    tree = ast.parse(_CONTEXT.read_text(encoding="utf-8"))
    fn = _module_function(tree, "compose_main_surface")
    unconditional: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if (
            not isinstance(target, ast.Name)
            or target.id != "sections"
            or not isinstance(value, ast.List)
        ):
            continue
        for elt in value.elts:
            if (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "SurfaceSection"
                and elt.args
                and isinstance(elt.args[0], ast.Constant)
            ):
                unconditional.add(elt.args[0].value)
    missing = sorted(required - unconditional)
    assert not missing, (
        f"required 段未在基础段列表无条件登记: {missing} —— 被移进条件分支 = 悄悄不发送"
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


def _synthetic_entries(pairs: list[tuple[str, str]]) -> list[object]:
    from huginn.catalog.models import CatalogEntry

    return [
        CatalogEntry(id=f"{kind}:{name}", kind=kind, name=name, origin="builtin")
        for kind, name in pairs
    ]


_ALL_KINDS_SAMPLE = [
    ("skill", "alpha"),
    ("workflow", "wf-beta"),
    ("plugin", "plug-gamma"),
    ("tool", "t"),
    ("mcp", "srv"),
    ("model", "m"),
    ("prompt", "p"),
]


def test_census_projects_every_kind_by_disclosure_level() -> None:
    """合成条目覆盖全部 kind: 三档披露按声明落地, 不再依赖真 Catalog 的偶发内容.

    这条门直接锁住 Critical 的形态: mcp/model/plugin 曾经结构上不可能出现.
    """
    from huginn.catalog.models import KINDS

    entries = _synthetic_entries(_ALL_KINDS_SAMPLE)
    text = census_text(entries)
    assert text.strip(), "census_text 返回空串 —— 披露退化成静默"
    for kind in KINDS:
        level = disclosure_level(kind)
        if level == "hidden":
            assert f"- {kind} (" not in text, f"hidden 级 kind {kind!r} 不应出现在普查段"
            continue
        assert f"- {kind} (" in text, f"{level} 级 kind {kind!r} 未出现在普查段"
    # always 级列名; on_demand 级只给数量与取用入口 (看不见 ≠ 不存在).
    for name in ("alpha", "wf-beta", "plug-gamma"):
        assert name in text, f"always 级条目 {name!r} 未列名"
    assert "tool_search" in text, "on_demand 级缺取用入口说明"


def test_census_truncation_points_to_entry_point() -> None:
    """截断时必须给取用入口, 否则模型只看到「+N」却不知去哪找."""
    entries = _synthetic_entries([("skill", f"s{i:02d}") for i in range(25)])
    text = census_text(entries)
    assert " (+5 more; use `tool_search` to look one up)" in text


def test_always_kinds_are_named_in_census_text() -> None:
    """真 Catalog: always 级条目名必须落到文本; census 永不返回空串 (无死 skip)."""
    from huginn.catalog.manager import CatalogManager

    entries = CatalogManager().discover_all()
    text = census_text(entries)
    assert text.strip(), "census_text 返回空串 —— 「声明该段生效」与实际渲染对不上"
    for kind in KIND_DISCLOSURE:
        if disclosure_level(kind) != "always":
            continue
        names = sorted(e.name for e in entries if e.kind == kind and e.enabled)
        if not names:
            continue
        assert names[0] in text, f"always 级 kind {kind!r} 的条目未出现在普查段"


# ── 组装根台账 (只减不增) ────────────────────────────────────────────────
# 扫描范围: 参与主对话/子代理/自主循环的包. 台账收录「构造 SystemMessage(content=...)
# **且未用 Surface 组装**」的模块 —— 在那里模型面仍是一根字符串, 段名在发送时丢失.
# 用 compose() 组装并 render() 的根不在台账内 (另由 test_surface_backed_roots_render
# 防止「调用 compose 却仍拼字符串」蒙混); 新增字符串根 = 红, 迁移一个就摘除一个.
_LEGACY_SURFACE_ROOTS: dict[str, str] = {
    "huginn/agent/core.py": "主会话图装配与 pre-model 钩子注入",
    "huginn/agent/middlewares.py": "规划/前沿任务 SystemMessage 注入",
    "huginn/agent/reflection.py": "反思总结器 (子 LLM)",
    "huginn/agent/streaming.py": "流式回合消息/预算/风格/收尾总结注入",
    "huginn/agents/subagent.py": "子代理输出总结器 (子 LLM)",
    "huginn/autoloop/conjecture.py": "LLM 猜想生成/重构 (子 LLM)",
    "huginn/autoloop/engine_act.py": "autoloop 主对话 (persona+GRILL+插件段)",
    "huginn/autoloop/engine_control.py": "旁支问题应答 (子 LLM)",
    "huginn/autoloop/engine_reflect.py": "LLM 判定 (子 LLM)",
    "huginn/autoloop/hypothesis_loop.py": "假设精炼 (子 LLM)",
    "huginn/autoloop/red_team.py": "红队发现 (子 LLM)",
    "huginn/coder/loop.py": "Coder 循环",
    "huginn/coder/refactor_engine.py": "重构计划 (子 LLM)",
    "huginn/interaction/clarification.py": "澄清提问 (子 LLM)",
}
_SCANNED_PACKAGES = ("agent", "autoloop", "agents", "coder", "interaction")


def _scan_candidates() -> list[tuple[str, ast.Module]]:
    """扫描范围内的全部模块 (rel path + 已解析 AST)."""
    out: list[tuple[str, ast.Module]] = []
    for path in sorted((_AGENT_ROOT / "huginn").rglob("*.py")):
        rel = path.relative_to(_AGENT_ROOT).as_posix()
        if not any(rel.startswith(f"huginn/{pkg}/") for pkg in _SCANNED_PACKAGES):
            continue
        try:
            out.append((rel, ast.parse(path.read_text(encoding="utf-8"))))
        except SyntaxError:
            continue
    return out


def _has_system_message(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "SystemMessage"
        and any(k.arg == "content" for k in n.keywords)
        for n in ast.walk(tree)
    )


def _calls(tree: ast.Module, name: str) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
        for n in ast.walk(tree)
    )


def _calls_render(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "render"
        for n in ast.walk(tree)
    )


def _surface_backed_root_files() -> set[str]:
    """构造 SystemMessage 且已用 compose() 组装的模块 (已迁移的根)."""
    return {
        rel for rel, tree in _scan_candidates()
        if _has_system_message(tree) and _calls(tree, "compose")
    }


def _system_message_root_files() -> set[str]:
    """构造 SystemMessage **却未用 Surface 组装**的模块 —— 未迁移的字符串根."""
    return {
        rel for rel, tree in _scan_candidates()
        if _has_system_message(tree) and not _calls(tree, "compose")
    }


def test_surface_roots_ledger_is_exact() -> None:
    """台账双向核对: 新增组装根红, 已迁移/已删未摘除也红."""
    found = _system_message_root_files()
    declared = set(_LEGACY_SURFACE_ROOTS)
    added = sorted(found - declared)
    removed = sorted(declared - found)
    assert not added, (
        "出现未登记的模型面组装根 (默认不可见 = 复发):\n" + "\n".join(added) +
        "\n要么走 Surface/compose, 要么在 _LEGACY_SURFACE_ROOTS 登记迁移理由"
    )
    assert not removed, "组装根已迁移, 请从 _LEGACY_SURFACE_ROOTS 摘除: " + ", ".join(removed)


def test_surface_backed_roots_render() -> None:
    """compose() 组装的根必须真的 render() —— 否则段名仍会在发送前消失."""
    backed = _surface_backed_root_files()
    assert backed, "扫描范围内未发现 compose() 组装的根 (表示层未被使用?)"
    missing_render = sorted(rel for rel, tree in _scan_candidates()
                            if rel in backed and not _calls_render(tree))
    assert not missing_render, (
        f"以下根调用 compose() 却未 render() Surface: {missing_render}"
    )


def test_every_declared_root_file_exists() -> None:
    missing = [f for f in _LEGACY_SURFACE_ROOTS if not (_AGENT_ROOT / f).is_file()]
    assert not missing, f"台账引用了不存在的文件: {missing}"


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


def test_runtime_surface_discloses_mechanisms(main_surface: Surface) -> None:
    assert "## MECHANISMS" in main_surface.render(), "主路径未披露机制清单"


def test_mechanisms_segment_covers_all_paths() -> None:
    from huginn.plugins.prompt_segments import segment_applicability

    paths = set(segment_applicability().get("mechanisms", ()))
    assert {"main", "s7", "autoloop"} <= paths, f"mechanisms 段未覆盖全部生产路径: {paths}"


def test_codeact_path_discloses_mechanisms() -> None:
    """第 4 条生产路径 (CodeAct) 也必须披露机制清单, 不能靠被 main 段覆盖."""
    try:
        from huginn.agent.code_act_loop import _build_surface
    except ImportError as exc:  # 离线环境缺依赖时跳过, 不制造假红
        pytest.skip(f"code_act_loop 依赖不可用: {exc}")

    class _Tool:
        description = "demo tool"
        input_schema = None

    surface = _build_surface({"alpha": _Tool()})
    assert surface.names() == ("codeact", "atomworld", "cognitive_map", "mechanisms")
    assert "## MECHANISMS" in surface.render(), "CodeAct 路径未披露机制清单"


def test_runtime_sources_are_all_declared(main_surface: Surface) -> None:
    undeclared = sorted(set(main_surface.sources()) - _ALLOWED_SOURCES)
    assert not undeclared, f"主路径出现未申报的段来源: {undeclared}"
