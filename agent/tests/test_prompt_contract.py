"""prompt 面「声明 ↔ 生效」一致性门 (prompt contract).

治理层把 prompt 段建模为"注册面 ↔ 生效面": `prompt_segments.register_prompt_segment`
注册段并声明 `applies_to` (段须在哪些生产路径生效), 各生产路径按声明渲染. 本门把该
契约固化为硬不变量, 闭合 prompt 面的缺口族:

- **声明齐全** — 每个 registered 段都有非空生效声明 (缺声明 = 孤儿段 → 红).
- **声明 = 渲染** — main / s7 / autoloop 三路径**实际渲染的段名集合**与声明逐路径相等
  (入口被改回硬编码单段 → 红).
- **关键段 floor** — 反防御写作 / 多模态 / 外部思考 / 安全 等不得从注册或声明中消失.
- **插件段显式声明** — 非内置段名的 `register_prompt_segment` 调用必须显式给
  `applies_to` (不得靠默认静默).
- **重复源同步** — safety 规范两份来源 (`prompts.py` §8 ↔ `prompt_builder.safety_segment`)
  语义锚点同步 (对齐 `MATH_DEPTH` 范式).
- **主路径源级守卫** — 主路径入口不得回退到"只渲染 thinking"的硬编码.

weightless: 按"命名空间包 + 文件路径"加载 `prompt_segments` / `prompt_builder`, 不触发
`huginn.plugins` / `huginn.agent` 的重依赖 `__init__` (不拉 langchain/pydantic/rich).

诚实边界: 本门只断言"段在某路径**生效**"(被渲染), **不**暗示"规范一定被模型**遵守**"
—— prompt 是引导, 非保证.

钩子事件面「声明 ↔ 触发 ↔ 注册」的一致性已由 `tests/test_contract_audit.py` 的
`build_hook_contract` 全量覆盖 (逐事件判 wired / trigger-only / dead), 本门不重复.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

_AGENT = pathlib.Path(__file__).resolve().parents[1]


def _ensure_namespace_pkg(name: str, path: pathlib.Path) -> None:
    """把 `name` 装成命名空间包 (只给 `__path__`), 绕过其重依赖 `__init__`.

    已加载的真实包不覆盖 (全量套件里其它测试可能已 import 真实包 —— 那正是我们想要的).
    """
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_namespace_pkg("huginn", _AGENT / "huginn")
_ensure_namespace_pkg("huginn.plugins", _AGENT / "huginn" / "plugins")
_ensure_namespace_pkg("huginn.agent", _AGENT / "huginn" / "agent")

# 导入 prompt_builder 时其模块尾的 `_register_builtin_segments()` 已注册 14 个内置段.
import huginn.agent.prompt_builder as pb  # noqa: E402
import huginn.plugins.prompt_segments as ps  # noqa: E402

_VALID_PATHS = frozenset({"main", "s7", "autoloop"})

# 关键段 floor (不得从注册/声明中消失). 与治理台账 `PROMPT_SPECS` 同源 (下方交叉核对).
_REQUIRED_SEGMENTS = (
    "persona", "core_principles", "behavioral_discipline", "response_format",
    "first_principles", "mode", "phase", "metacog", "tools",
    "multimodal", "writing", "multi_agent_protocol", "thinking", "safety",
)

# safety 规范两份来源必须共同覆盖的语义锚点 (允许措辞微差, 只锁语义集).
_SAFETY_ANCHORS = ("Physics precheck", "Data integrity", "No fabrication")

# base prompt (prompts.py) 中曾寄生在 persona 单源、现已**拆成独立段**的行为规范:
# §3 Core Principles / §4 Behavioral Discipline / §7 Response Format. 三节作为段
# (core_principles/behavioral_discipline/response_format) 在主/s7/autoloop 三路统一供给
# (文本单一事实来源 `huginn.prompts.PROMPT_SECTIONS`). 段门虽保证"段存在且生效", 却仍
# 无法定位到"工具调用纪律子句被删了"或"证据强度条款被悄悄改弱" —— 故另用**声明式锚点门**
# 锁定关键子句 (镜像 `_SAFETY_ANCHORS` 范式): 删除/削弱任一锚点 = 红, 须显式改本表 (可见决策).
_PRINCIPLES_ANCHORS = (
    "Mathematical Rigor",
    "Physical Validation",
    "Epistemic Honesty",
    "Adversarial Review",
)
_DISCIPLINE_ANCHORS = (
    "Tool use & failure handling",
    "Least-effort path",
    "Retry discipline",
    "Benchmark alignment",
    "Literature discipline",
)
_FORMAT_ANCHORS = (
    "Response Format",
    "confidence assessment",
)


def _applicability() -> dict[str, tuple[str, ...]]:
    return ps.segment_applicability()


def _invoked_names(monkeypatch, runner) -> set[str]:
    """包裹 registry 的段函数, 记录**实际被调用**的段名 (含返回空串的段).

    测的是"路径入口真的调了哪些段", 而非只看声明表 —— 能抓住"入口硬编码单段"
    这类回归.
    """
    seen: list[str] = []
    real_ordered = ps._registry.ordered

    def spy():
        out = []
        for name, prio, fn in real_ordered():
            def wrapped(*a, __n=name, __f=fn, **k):
                seen.append(__n)
                return __f(*a, **k)

            out.append((name, prio, wrapped))
        return out

    monkeypatch.setattr(ps._registry, "ordered", spy)
    runner()
    return set(seen)


# ──────────────────────── 声明齐全性 ────────────────────────


def test_every_registered_segment_has_applies_to():
    """AC-1: 每个 registered 段都有非空生效声明, 且路径取值合法."""
    app = _applicability()
    registered = set(ps.registered_prompt_segments())
    assert registered, "注册表为空 —— 内置段未注册?"
    missing = sorted(n for n in registered if not app.get(n))
    assert missing == [], (
        f"段缺生效声明 (孤儿段): {missing} —— 内置段补进 `_APPLIES_TO`, 第三方插件段在"
        f" register 时显式给 applies_to (可见决策)."
    )
    bad = sorted(
        f"{n}:{sorted(set(v) - _VALID_PATHS)}"
        for n, v in app.items()
        if set(v) - _VALID_PATHS
    )
    assert bad == [], f"生效路径声明含非法路径 (仅限 main/s7/autoloop): {bad}"


# ──────────────────── 声明面 = 生效面 (逐路径) ────────────────────


def test_main_path_renders_exactly_declared_segments(monkeypatch):
    """AC-2/AC-3: 主路径实际渲染的段集合 == 声明含 main 的段集合."""
    declared = {n for n, v in _applicability().items() if "main" in v}
    invoked = _invoked_names(
        monkeypatch,
        lambda: ps.render_prompt_segments_for_path("main", "chat", "execute", "s0_blank"),
    )
    assert invoked == declared, (
        f"主路径渲染集合 {sorted(invoked)} != 声明集合 {sorted(declared)} —— "
        f"render_prompt_segments_for_path 的过滤与 _APPLIES_TO 声明脱节."
    )


def test_s7_path_renders_all_registered_segments(monkeypatch):
    """s7 全量组装渲染**全部**注册段 ⇒ 每个注册段都须声明 s7 (否则声明撒谎)."""
    declared_s7 = {n for n, v in _applicability().items() if "s7" in v}
    registered = set(ps.registered_prompt_segments())
    invoked = _invoked_names(
        monkeypatch, lambda: ps.assemble_prompt_segments("chat", "execute", "s0_blank")
    )
    assert invoked == registered, (
        f"s7 组装渲染集合 {sorted(invoked)} != 全部注册段 {sorted(registered)}."
    )
    assert registered == declared_s7, (
        f"以下段在 s7 会被渲染却未声明 s7: {sorted(registered - declared_s7)} —— "
        f"给这些段补上 applies_to 里的 's7'."
    )


def test_autoloop_path_declares_every_plugin_segment(monkeypatch):
    """autoloop 消费**全部非框架段** ⇒ 每个非框架(插件)段都须声明 autoloop."""
    declared_auto = {n for n, v in _applicability().items() if "autoloop" in v}
    plugin = {
        n for n in ps.registered_prompt_segments() if n not in ps._FRAMEWORK_SEGMENTS
    }
    invoked = _invoked_names(
        monkeypatch,
        lambda: ps.assemble_plugin_prompt_segments("chat", "execute", "s0_blank"),
    )
    assert invoked == plugin, (
        f"autoloop 组装渲染集合 {sorted(invoked)} != 非框架段 {sorted(plugin)}."
    )
    assert plugin == declared_auto, (
        f"以下插件段在 autoloop 会被渲染却未声明 autoloop: {sorted(plugin - declared_auto)}."
    )


def test_unknown_path_rejected():
    """未知 path 立即报错, 不静默."""
    with pytest.raises(ValueError):
        ps.render_prompt_segments_for_path("s8", "chat", "execute", "s0_blank")


# ──────────────────────── 关键段 floor ────────────────────────


def test_required_segments_declared_and_registered():
    """floor: 关键段不得从注册/声明中消失 (静默萎缩)."""
    app = _applicability()
    registered = set(ps.registered_prompt_segments())
    for name in _REQUIRED_SEGMENTS:
        assert name in registered, f"关键段 `{name}` 从注册表消失."
        assert app.get(name), f"关键段 `{name}` 的生效声明被清空."


def test_main_path_really_injects_writing_and_multimodal():
    """AC-3 强断言: 主路径真实产出含写作/多模态/多智能体/拆段三节 (非仅声明)."""
    out = ps.render_prompt_segments_for_path("main", "chat", "execute", "s0_blank")
    assert "## WRITING" in out, "主路径未注入反防御写作段 (G-2)."
    assert "## MULTIMODAL" in out, "主路径未注入多模态引导段 (G-1)."
    assert "## MULTI-AGENT PROTOCOL" in out, "主路径未注入多智能体协作协议段."
    # §3/§4/§7 拆出后, 主路径必须由段补齐这三节 (persona 基线已不含).
    assert "## 3. Core Principles" in out, "主路径未注入 §3 Core Principles 段."
    assert "## 4. Behavioral Discipline" in out, "主路径未注入 §4 Behavioral Discipline 段."
    assert "## 7. Response Format" in out, "主路径未注入 §7 Response Format 段."


# ──────── 拆段 (§3/§4/§7) 的声明式锚点门 + 段生效断言 ────────


def test_base_prompt_spec_anchors_locked():
    """AC-6: prompts.py 中 §3 Core Principles / §4 Behavioral Discipline / §7 Response
    Format 的关键子句不得被静默删除或削弱, 且须经拆出的段真正生效.

    三节已从 persona 串剥离为独立段, 文本单一事实来源为 `PROMPT_SECTIONS` (由
    `HUGINN_SYSTEM_PROMPT` 确定性切分). 本门: (1) 锁 prompts.py 源文本的关键子句锚点;
    (2) 断言三段函数渲染出的文本仍含这些锚点 —— 切分漂移或段被清空即红; (3) 断言拆段后
    `HUGINN_BASE_PROMPT` 里这些节确已移除 (不再与段重复注入). 任一失败即红, 须显式改本表/
    切分或段实现 (可见决策). 镜像 safety 范式.
    """
    import huginn.prompts as pr

    src = (_AGENT / "huginn" / "prompts.py").read_text(encoding="utf-8")
    section_segment = {
        "§3 Core Principles": ("core_principles", _PRINCIPLES_ANCHORS),
        "§4 Behavioral Discipline": ("behavioral_discipline", _DISCIPLINE_ANCHORS),
        "§7 Response Format": ("response_format", _FORMAT_ANCHORS),
    }
    for section, (seg_key, anchors) in section_segment.items():
        for anchor in anchors:
            assert anchor in src, (
                f"prompts.py {section} 缺锚点 '{anchor}' —— 该行为规范被删除或削弱? "
                f"若确为有意改写, 显式更新 test_prompt_contract 的锚点集 (可见决策)."
            )
            seg_text = pr.PROMPT_SECTIONS[seg_key]
            assert anchor in seg_text, (
                f"拆出的段 `{seg_key}` 缺锚点 '{anchor}' —— 切分漂移或段被清空?"
            )
    # 拆段后 persona 基线不得再内联这三节 (否则主/s7/autoloop 会重复注入).
    for heading in ("## 3. Core Principles", "## 4. Behavioral Discipline", "## 7. Response Format"):
        assert heading not in pr.HUGINN_BASE_PROMPT, (
            f"HUGINN_BASE_PROMPT 仍含 '{heading}' —— 拆段未剥离干净, 会与段重复注入."
        )


# ──────────────────── 插件注册必须显式声明 ────────────────────


def _register_calls() -> list[tuple[pathlib.Path, ast.Call]]:
    calls: list[tuple[pathlib.Path, ast.Call]] = []
    for p in (_AGENT / "huginn").rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_prompt_segment"
            ):
                calls.append((p, node))
    return calls


def test_plugin_registrations_declare_applies_to():
    """AC-4: 非内置段名的 register 调用必须显式声明 applies_to (不得靠默认静默).

    内置段 (prompt_builder 里以字面量段名注册的 14 段) 声明来自内置 `_APPLIES_TO`;
    其余调用 (第三方插件段) 段名非内置 → 必须显式给 `applies_to=`.
    """
    builtin = frozenset(_REQUIRED_SEGMENTS)
    offenders: list[str] = []
    for path, call in _register_calls():
        name_arg = call.args[0] if call.args else None
        is_builtin_literal = (
            isinstance(name_arg, ast.Constant)
            and isinstance(name_arg.value, str)
            and name_arg.value in builtin
        )
        if is_builtin_literal:
            continue
        if not any(kw.arg == "applies_to" for kw in call.keywords):
            offenders.append(f"{path.relative_to(_AGENT)}:{call.lineno}")
    assert offenders == [], (
        f"以下 register_prompt_segment 调用未显式声明 applies_to: {offenders} —— "
        f"非内置段必须声明生效路径, 否则被判为孤儿段."
    )


def test_asd_ste100_declares_all_paths():
    """AC-4 具体锚点: asd_ste100 插件段的 applies_to 覆盖 main/s7/autoloop."""
    src = (
        _AGENT / "huginn" / "plugins" / "asd_ste100" / "main.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "register_prompt_segment"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            assert "applies_to" in kw, "asd_ste100 未声明 applies_to."
            paths = ast.literal_eval(kw["applies_to"])
            assert set(paths) == _VALID_PATHS, (
                f"asd_ste100 声明路径 {sorted(paths)} != 全部生产路径 {sorted(_VALID_PATHS)}."
            )
            return
    raise AssertionError("asd_ste100 未找到 register_prompt_segment 调用.")


# ──────────────── 重复规范源同步 (MATH_DEPTH 范式) ────────────────


def test_safety_spec_source_synchronized():
    """AC-5: safety 规范两份来源语义同步.

    safety 段是主路径 (`prompts.py` §8 承载) 与 s7 组装 (`prompt_builder.safety_segment`)
    的**重复源**; 照 MATH_DEPTH 范式加同步门: 两处必须覆盖同一组语义锚点, 不得各自漂移
    (一处改了另一处没改 → 红). 不做逐字相等 (允许措辞微差), 只锁语义锚点集.
    """
    prompts_text = (_AGENT / "huginn" / "prompts.py").read_text(encoding="utf-8")
    assert "## 8. Safety" in prompts_text, "prompts.py 缺 §8 Safety 节."
    for anchor in _SAFETY_ANCHORS:
        assert anchor in prompts_text, f"prompts.py §8 缺 safety 锚点: {anchor}"

    seg = pb.safety_segment()
    assert seg.startswith("## SAFETY")
    for anchor in _SAFETY_ANCHORS:
        assert anchor in seg, f"prompt_builder.safety_segment 缺 safety 锚点: {anchor}"


# ──────────────── 主路径源级守卫 + 台账交叉核对 ────────────────


def test_main_path_source_uses_declaration_renderer():
    """源级守卫 (G-1): 主路径入口须调用按声明渲染, 不得回退到只渲染 thinking 的硬编码."""
    src = (_AGENT / "huginn" / "agent" / "context.py").read_text(encoding="utf-8")
    assert "render_prompt_segments_for_path(" in src, (
        "context.py 不再使用 render_prompt_segments_for_path —— 主路径段注入被改回硬编码?"
    )
    assert 'render_prompt_segment("thinking"' not in src, (
        "context.py 回退到只渲染 thinking 的硬编码 (孤儿段回归)."
    )


def test_governance_prompt_specs_match_registry():
    """台账 `PROMPT_SPECS` 段名清单与真实注册表一致 (声明面单一来源不分叉)."""
    spec = importlib.util.spec_from_file_location(
        "governance_audit_pc", _AGENT / "huginn" / "cli" / "governance_audit.py"
    )
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    assert set(ga.PROMPT_SPECS) == set(ps.registered_prompt_segments()), (
        f"台账 PROMPT_SPECS={sorted(ga.PROMPT_SPECS)} 与注册表 "
        f"{sorted(ps.registered_prompt_segments())} 不一致 —— 段增删须两边同步."
    )
