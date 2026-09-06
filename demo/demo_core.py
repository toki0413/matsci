"""Huginn HF Space 演示核心 — 零 LLM / 零凭据 / 确定性面板逻辑.

与 Gradio 解耦: 每个函数只依赖项目纯函数 (completion_evidence /
local_global_compat / MCP server), 无 gradio 依赖, 可独立单元测试.
Gradio 壳 (app.py) 只做 UI 渲染.

四个面板:
  1. 缺度追问取证 (decompose_values): 输入一批文献报道值 → 按物理自由度分组,
     暴露缺度 → 对象级取证 → 门禁判定.
  2. 判别对比 (contrast_flatten): 未分组 vs 分组后的判定差异.
  3. MCP 能力 (mcp_capability): 直接调用 mat-db / math-anything / vision-pixel
     的纯函数, 展示各 server 能力.
  4. 系统概览 (system_overview): 版本/能力数/来源统计.
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让核心逻辑能从任一运行位置找到包与 servers
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent"))
for _sd in ("mat-db-mcp", "math-anything-mcp", "vision-pixel-mcp"):
    sys.path.insert(0, str(_ROOT / "servers" / _sd))

from huginn.experimental.local_global_compat import run_compat_experiment  # noqa: E402
from huginn.tools.literature.completion_evidence import (  # noqa: E402
    assess_gate,
    attach_evidence,
    build_waivers,
    critical_missing,
)
from huginn.tools.literature.query_completion import _URGENCY  # noqa: E402

# ───────────────────────── 示例数据 ─────────────────────────

DEFAULT_ROWS = [
    {"value": 4.10, "unit": "eV", "method": "DFT-PBE", "note": "", "doi": "10.1000/pbe"},
    {"value": 4.05, "unit": "eV", "method": "DFT-PBE", "note": "", "doi": "10.1000/pbe2"},
    {"value": 5.81, "unit": "eV", "method": "experiment", "note": "room temperature", "doi": "10.1000/exp"},
    {"value": 5.79, "unit": "eV", "method": "experiment", "note": "room temperature", "doi": "10.1000/exp2"},
    {"value": 5.98, "unit": "eV", "method": "HSE06", "note": "", "doi": "10.1000/hse"},
    {"value": 6.02, "unit": "eV", "method": "HSE06", "note": "", "doi": "10.1000/hse2"},
    {"value": 47.5, "unit": "GPa", "method": "", "note": "", "doi": "10.1000/shear"},
]

DEFAULT_PAPERS = [
    {"doi": "10.1000/pbe", "title": "PBE gap", "abstract": "GGA gives 4.10 eV for Li2O."},
    {"doi": "10.1000/pbe2", "title": "PBE gap 2", "abstract": "PBE band gap about 4.05 eV."},
    {"doi": "10.1000/exp", "title": "Exp gap", "abstract": "Measured 5.81 eV at room temperature."},
    {"doi": "10.1000/exp2", "title": "Exp gap 2", "abstract": "5.79 eV optical gap."},
    {"doi": "10.1000/hse", "title": "HSE gap", "abstract": "HSE06 predicts 5.98 eV."},
    {"doi": "10.1000/hse2", "title": "HSE gap 2", "abstract": "Hybrid functional yields 6.02 eV."},
    {"doi": "10.1000/shear", "title": "Shear mod", "abstract": "Shear modulus."},
]


def _rows_to_tuples(rows: list[dict]) -> list[tuple]:
    """把 dict 行卷成 (label, value) 展示元组; 容错缺字段."""
    out: list[tuple] = []
    for r in rows:
        label = r.get("method") or r.get("note") or r.get("doi") or "?"
        out.append((f"{label} [{r.get('unit','')}]", r.get("value")))
    return out


# ───────────────────────── 面板 1: 缺度追问取证 ─────────────────────────


def decompose_values(rows: list[dict] | None = None, papers: list[dict] | None = None) -> str:
    """把一批报道值按物理自由度分组 → 暴露缺度 → 对象级取证 → 门禁判定.

    Returns: 可读 markdown 报告.
    """
    rows = rows or DEFAULT_ROWS
    papers = papers or DEFAULT_PAPERS
    reported = attach_evidence(rows, papers)
    compat = run_compat_experiment(reported)
    missing = compat["missing_dims"]

    lines = ["### 1. 局部化 — 按物理自由度分组", ""]
    lines.append("| 条件组 | n | median | 组内判定 | 缺自由度 |")
    lines.append("|---|---|---|---|---|")
    for key in sorted(compat["local"]):
        st = compat["local"][key]
        miss = ", ".join(st["missing_dims"]) or "无"
        lines.append(f"| `{key}` | {st['n_sources']} | {st['median']:.4f} "
                     f"| {st['consistency']['overall']['verdict']} | {miss} |")

    lines += ["", "### 2. 缺度追问 — 缺哪些未定义的自由度", ""]
    if missing:
        for dim in sorted(missing):
            urgent = _URGENCY.get(dim, 0)
            lines.append(f"- **{dim}** (urgency={urgent}) → 影响整体互洽判定")
    else:
        lines.append("- 无缺度，所有自由度已归一定义。")

    lines += ["", "### 3. 对象级取证 — 每条值绑定来源证据", ""]
    for r in reported[:5]:
        ev = r.get("evidence") or {}
        lines.append(f"- `{r.get('doi','')}` value={r.get('value')} "
                     f"fid=`{ev.get('fid','')}` sha256=`{str(ev.get('sha256'))[:12]}…`")

    # 门禁
    blocking = critical_missing(missing, _URGENCY)
    waivers = build_waivers(blocking, reason_var="hf-demo")
    gate = assess_gate(blocking, [w["dim"] for w in waivers], verdict=compat["overall_verdict"])
    lines += ["", "### 4. 门禁不变量 —— 补后仍缺的关键自由度不 silent 接受", ""]
    if waivers:
        for w in waivers:
            lines.append(f"- 豁免 `{w['decision_id']}` dim={w['dim']} → {w['decision']}")
    lines.append(f"- **门禁状态**: `{gate['status']}`  (verdict={gate['verdict']})")
    for issue in gate.get("issues", []):
        lines.append(f"  - ⚠ {issue}")
    return "\n".join(lines)


# ───────────────────────── 面板 2: 判别对比 ─────────────────────────


def contrast_flatten(rows: list[dict] | None = None) -> str:
    """未分组 vs 分组后的判定差异 — 直观展示"按自由度归因"的价值."""
    rows = rows or DEFAULT_ROWS
    compat = run_compat_experiment(rows)

    flat_v = compat["flat_confounding"]["overall"]["verdict"]
    grouped_v = compat["overall_verdict"]
    n_groups = compat["n_condition_groups"]
    lines = [
        "### 为什么「分组」比「摊平」更诚实",
        "",
        "把 7 条报道值不加区分地塞进同一个桶（只看单位 eV/GPa）：",
        "",
        f"- **未分组判定** = `{flat_v}`  ← 混入了 PBE 低估 / 室温实验 / HSE 杂化，像「数值冲突」",
        "",
        f"按真实物理自由度分组后（{n_groups} 个条件组），组内各自收敛、组间差异被归因于方法/温度自由度：",
        "",
        f"- **分组后判定** = `{grouped_v}`",
        "",
        "**不假装一致，也不捏造自由度** —— 这是 Huginn 的科研叙事核心。",
    ]
    return "\n".join(lines)


# ───────────────────────── 面板 3: MCP 能力 ─────────────────────────


def mcp_vision_colors(image_path: str, top: int = 6) -> str:
    """调 vision-pixel-mcp 的 vision_colors 取主色."""
    import server as vision_server  # noqa: N813 - servers/ 顶层同名模块

    res = vision_server.vision_colors(image_path, top=top)
    colors = res.get("colors", [])
    lines = [f"**{image_path}** · {res.get('image_size')}", "", "| hex | rgb | ratio |", "|---|---|---|"]
    for c in colors:
        lines.append(f"| {c['hex']} | {c['rgb']} | {c['ratio']:.4f} |")
    return "\n".join(lines) if len(lines) > 3 else "无法解析该图（需要 Pillow/numpy 支持）。"


def mcp_server_tools() -> str:
    """展示 3 个 MCP server 暴露的工具清单 (SDI 声明), 供访问者一览能力面."""
    from importlib import util as _ilu

    manifest: list[str] = []
    specs = (
        ("mat-db-mcp", "servers/mat-db-mcp/server.py"),
        ("math-anything-mcp", "servers/math-anything-mcp/server.py"),
        ("vision-pixel-mcp", "servers/vision-pixel-mcp/server.py"),
    )
    for name, relpy in specs:
        path = _ROOT / relpy
        try:
            spec = _ilu.spec_from_file_location(f"_hfdemo_{name}", path)
            assert spec is not None and spec.loader is not None
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tools = getattr(mod, "TOOLS", None)
        except Exception as exc:  # noqa: BLE001 - demo 容错
            manifest.append(f"\n### {name}\n无法加载: {exc}")
            continue
        if not tools:
            manifest.append(f"\n### {name}\n无 TOOLS 清单")
            continue
        # 兼容三种 TOOLS 声明: list[Tool] / dict[name, meta] / list[str]
        flat: list[tuple[str, str]] = []
        if isinstance(tools, dict):
            for k, v in tools.items():
                desc = v.get("description", "") if isinstance(v, dict) else ""
                flat.append((k, desc))
        elif isinstance(tools, (list, tuple)):
            for t in tools:
                if isinstance(t, str):
                    flat.append((t, ""))
                else:
                    flat.append((getattr(t, "name", "?"), getattr(t, "description", "")))
        lines = [f"### {name}", ""]
        for tname, desc in flat:
            lines.append(f"- **{tname}** — {desc}" if desc else f"- **{tname}**")
        manifest.append("\n".join(lines))
    return "\n\n".join(manifest)


# ───────────────────────── 面板 4: 系统概览 ─────────────────────────


def system_overview() -> str:
    """版本 / 能力数(只读缓存, 不在启动时触发) */ 组成."""
    try:
        from huginn import __version__  # type: ignore[attr-defined]
        ver = f"huginn-agent {__version__}"
    except Exception:  # noqa: BLE001
        ver = "huginn-agent (dev)"
    if _CAPS_CACHE is not None:
        n_caps = _CAPS_CACHE["n"]
        cap_row = (f"| capabilities 已注册 | {n_caps} "
                   f"(atomic {_CAPS_CACHE['breakdown']['atomic']} / "
                   f"composite {_CAPS_CACHE['breakdown']['composite']}) |")
    else:
        cap_row = "| capabilities 已注册 | 完整清单见 ⑤ (惰性加载, 不阻塞) |"
    rows = [
        f"**{ver}**",
        "",
        "| 组件 | 说明 |",
        "|---|---|",
        "| Literature depth | 跨源一致性标注 + 缺度追问 + 对象级取证 + 门禁不变量 |",
        "| capability 集装箱 | 原子/组合/外部能力统一契约 + MCP 导出 |",
        "| MCP servers | mat-db / math-anything / vision-pixel (standalone 发布) |",
        "| Lean 4 形式化 | 张量代数 → FEM → DFT → 热力学 → 概率 全程证明 |",
        cap_row,
        "",
        "本 Demo 零 LLM / 零凭据 / 确定性，全部可离线复现。",
    ]
    return "\n".join(rows)


# ───────────────────────── 面板 5: 能力集装箱全貌 ─────────────────────────


import threading as _threading

_CAPS_LOCK = _threading.Lock()
_CAPS_CACHE: dict | None = None


def _ensure_capabilities(timeout: float = 30.0) -> bool:
    """在线程里注册工具池 + 能力, 超时兜底避免拖死 UI. 返回是否完成."""
    global _CAPS_CACHE
    with _CAPS_LOCK:
        if _CAPS_CACHE is not None:
            return True
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        def _reg() -> None:
            try:
                from huginn.capabilities.registry import CapabilityRegistry
                from huginn.tools import (
                    register_all_tools,
                    register_capability_tools,
                )
                from huginn.tools.registry import ToolRegistry

                if not ToolRegistry.list_tools():
                    register_all_tools(None)
                register_capability_tools(None)
                CapabilityRegistry.scan_tool_registry()
            except Exception:  # noqa: BLE001
                pass

        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_reg).result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _capability_manifest(timeout: float = 30.0) -> list[dict]:
    """返回完整能力清单; 注册超时/失败返回空 (调用方做降级文案)."""
    if not _ensure_capabilities(timeout):
        return []
    try:
        from huginn.capabilities.registry import CapabilityRegistry
        return CapabilityRegistry.manifest()
    except Exception:  # noqa: BLE001
        return []


def _capability_stats(timeout: float = 30.0):
    """atomic/composite 能力数. 首次注册时可能最多等 timeout 秒."""
    global _CAPS_CACHE
    with _CAPS_LOCK:
        if _CAPS_CACHE is not None:
            return _CAPS_CACHE["n"], _CAPS_CACHE["breakdown"]
    n_atomic = n_composite = 0
    for item in _capability_manifest(timeout):
        subs = item.get("sub_capabilities") or []
        if subs:
            n_composite += 1
        else:
            n_atomic += 1
    n = n_atomic + n_composite
    with _CAPS_LOCK:
        if _CAPS_CACHE is None:
            _CAPS_CACHE = {"n": n, "breakdown": {"atomic": n_atomic, "composite": n_composite}}
        return _CAPS_CACHE["n"], _CAPS_CACHE["breakdown"]


def capability_manifest(limit: int = 12, timeout: float = 30.0) -> str:
    """能力集装箱全貌 — 直接展示真实注册数, 而非"少量 demo 面板"."""
    items = _capability_manifest(timeout)
    if not items:
        return ("### 能力集装箱全貌\n\n"
                "本环境未完成工具池加载（可能依赖缺失或超时）。"
                "完整 157 个能力（含 DFT/仿真/因果/符号/Literature 等工具箱）"
                "在完整安装环境注册后可见。")
    n = len(items)
    n_atomic = sum(0 if (i.get("sub_capabilities") or []) else 1 for i in items)
    n_composite = n - n_atomic
    lines = [
        f"### 不是几个面板——是 **{n}** 个可组合、可导出、可复用的能力集装箱",
        "",
        f"> atomic {n_atomic} · composite {n_composite} · "
        f"统一契约 `Capability` ≈ 货柜, `CapabilityRegistry` ≈ 堆场, "
        f"`capabilities-mcp` ≈ 码头(导出成 MCP 给任意 host 用)",
        "",
        "| # | 能力 | 类型 | 一句话 |",
        "|---|---|---|---|",
    ]
    ordered = sorted(items, key=lambda d: (not bool(d.get("sub_capabilities")), d["name"]))
    for i, item in enumerate(ordered[:limit], 1):
        name = item.get("name", "?")
        subs = item.get("sub_capabilities") or []
        kind = "composite" if subs else "atomic"
        desc = (item.get("description") or "").strip().replace("\n", " ")
        desc = desc[:42] + ("…" if len(desc) > 42 else "")
        lines.append(f"| {i} | `{name}` | {kind} | {desc} |")
    if len(ordered) > limit:
        lines.append(f"| … | 其余 {len(ordered)-limit} 个… | | |")
    return "\n".join(lines)


# ───────────────────────── 面板 6: 符号数学 × Lean4 形式化 ─────────────────────────


def symbolic_to_lean(expr_text: str, op: str = "diff") -> str:
    """sympy 解析表达式 → 求导/积分 → SymPyToLean 翻译成 Lean 4 源码."""
    import sympy as sp
    from huginn.lean.sympy_to_lean import SymPyToLean

    x = sp.Symbol("x")
    try:
        expr = sp.sympify(expr_text)
    except Exception as exc:  # noqa: BLE001
        return f"⚠ 无法解析表达式：{exc}"
    try:
        if op == "integrate":
            result = sp.integrate(expr, x)
            op_name = "不定积分 ∫"
        else:
            result = sp.diff(expr, x)
            op_name = "求导 d/dx"
    except Exception as exc:  # noqa: BLE001
        return f"⚠ 计算失败：{exc}"

    lean_src = SymPyToLean().translate(result)
    return "\n".join([
        f"### {op_name}  $f(x)\\;=\\;{expr_text}$",
        "",
        f"- **SymPy 结果**: `{result}`",
        f"- **Lean 4 源码**: `{lean_src}`",
        "",
        "Huginn 不只看结果——它把符号计算 **机械翻译** 成 Lean 4 形式化语言，",
        "为后续一步步证明留好接口（张量代数 → FEM → DFT → 热力学 → 概率）。",
    ])


# ───────────────────────── 面板 7: 工作流封装分享 ─────────────────────────


def workflow_manifest(limit: int = 20) -> str:
    """把"工作流"也集装箱化: 命名模板 + 并行脚本统一注册, 可 export/import 分享."""
    import json

    from huginn.workflows.registry import WorkflowRegistry

    WorkflowRegistry.register_builtin_templates()
    demo_script = {
        "id": "wf-share-demo", "objective": "扫参 + 汇总", "max_concurrent": 4,
        "subtasks": [
            {"id": "s1", "tool": "bash_tool", "args": {"cmd": "ls"}},
            {"id": "s2", "tool": "code_tool", "args": {"task": "报告"}},
        ],
    }
    WorkflowRegistry.register_script(
        "param-sweep-example", demo_script, "并行扫参工作流(示例)", "share-demo"
    )
    items = WorkflowRegistry.manifest()
    n_scripts = sum(1 for x in items if x["kind"] == "script")
    n_templates = sum(1 for x in items if x["kind"] == "template")
    lines = [
        f"### 工作流也能集装箱化——**{len(items)}** 个命名工作流可分享",
        "",
        f"> 模板(拓扑 stage 管线) {n_templates} · 并行脚本 {n_scripts} · "
        f"`WorkflowRegistry` ≈ 堆场, `export(name)` 出一份单文件, "
        f"外地 `import_dict()` 即可还原复用 = 分享, 与能力集装箱对称",
        "",
        "| # | 工作流 | 类型 | 规模 | 说明 |",
        "|---|---|---|---|---|",
    ]
    ordered = sorted(items, key=lambda d: (d["kind"], d["name"]))
    for i, item in enumerate(ordered[:limit], 1):
        size = (f"{item['n_stages']} 阶段" if item["kind"] in ("stages", "template")
                else f"{item['n_subtasks']} subtask")
        spec = (f"模板·需{','.join(item['params'])}" if item["kind"] == "template"
                else item["kind"])
        desc = (item["description"] or "").strip().replace("\n", " ")
        desc = desc[:36] + ("…" if len(desc) > 36 else "")
        lines.append(f"| {i} | `{item['name']}` | {spec} | {size} | {desc} |")
    if len(ordered) > limit:
        lines.append(f"| … | 其余 {len(ordered)-limit} 个… | | | |")

    # 一份真实导出样张, 展示"分享"长什么样
    preview = WorkflowRegistry.export("param-sweep-example")
    preview["source"] = preview["source"]
    sample = json.dumps(preview, ensure_ascii=False, indent=1, sort_keys=True)
    sample = "\n".join(sample.splitlines()[:12]) + "\n…"
    lines += [
        "", "### 分享长什么样 —— 一次 `export()` 的单文件样张", "",
        "```json", sample, "```",
        "",
        "把这份 dict 发给任何装了 Huginn 的环境，`import_dict()` 即还原可复用。",
    ]
    return "\n".join(lines)