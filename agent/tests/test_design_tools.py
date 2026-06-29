"""Tests for design 类 7 个工具 — 行为测试, 不依赖 LLM.

覆盖: GapAnalysis / DOE / Debugger / DesignPlan / Nudge / DesignAtom / GenerativeDesign
风格参考 test_clarification_tool.py: async def + await tool.call + 断言 result.success/data.
单例工具 (_PlanStore / _NudgeStore) 用 autouse fixture 重置 _instance 避免跨测试污染.
GenerativeDesignTool 用 FakeSandbox + tmp_path 注入, 不真跑代码.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.tools.design.gap_analysis_tool import GapAnalysisTool
from huginn.tools.design.doe_tool import DOEInput, DOETool
from huginn.tools.design.debugger_tool import DebuggerInput, DebuggerTool
from huginn.tools.design.design_plan_tool import DesignPlanTool, _PlanStore
from huginn.tools.design.nudge_tool import NudgeTool, _NudgeStore
from huginn.tools.design.design_atom_tool import DesignAtomTool
from huginn.tools.design.generative_design_tool import GenerativeDesignTool


# ── 单例重置 fixture ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_design_singletons():
    """每个测试前清掉 _PlanStore / _NudgeStore 单例, 防止状态污染."""
    _PlanStore._instance = None
    _NudgeStore._instance = None
    yield
    _PlanStore._instance = None
    _NudgeStore._instance = None


# ── FakeSandbox for GenerativeDesignTool ─────────────────────────────


class FakeSandbox:
    """假 sandbox, run() 按预设返回. 记录调用参数供断言."""

    def __init__(self, result=None, raises=None):
        # result 可以是 dict 或对象 (有 stdout/stderr/returncode 属性)
        self._result = result or {"stdout": "", "stderr": "", "returncode": 0}
        self._raises = raises
        self.calls: list[dict] = []

    def run(self, code, work_dir=None, timeout=None):
        if self._raises:
            raise self._raises
        self.calls.append({"code": code, "work_dir": work_dir, "timeout": timeout})
        return self._result


class _FakeRunResult:
    """带属性的对象形态 sandbox 返回, 测 hasattr 分支."""

    def __init__(self, stdout="ok", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ════════════════════════════════════════════════════════════════════
# GapAnalysisTool
# ════════════════════════════════════════════════════════════════════


async def test_gap_analyze_gaps_empty_papers():
    """空 papers 短路返回 summary, 不报错."""
    tool = GapAnalysisTool()
    result = await tool.call(
        {"action": "analyze_gaps", "topic": "perovskite", "papers": []},
        context=None,
    )
    assert result.success is True
    assert result.data["gaps"] == []
    assert "没有论文可分析" in result.data["summary"]


async def test_gap_analyze_gaps_finds_missing_combinations():
    """3 篇论文覆盖 method×material, 第 3 个组合未出现应被识别为 missing_combination."""
    tool = GapAnalysisTool()
    papers = [
        {"title": "p1", "methods": ["DFT"], "materials": ["TiO2"], "results": ""},
        {"title": "p2", "methods": ["MD"], "materials": ["TiO2"], "results": ""},
        {"title": "p3", "methods": ["DFT"], "materials": ["Si"], "results": ""},
    ]
    result = await tool.call(
        {"action": "analyze_gaps", "topic": "oxides", "papers": papers},
        context=None,
    )
    assert result.success is True
    gaps = result.data["gaps"]
    missing = [g for g in gaps if g["type"] == "missing_combination"]
    # MD × Si 应该是未出现的组合
    combos = {(g["method"], g["material"]) for g in missing}
    assert ("md", "si") in combos


async def test_gap_compare_methods_returns_overlap_info():
    """compare_methods 对多方法论文分组, 返回 comparison 列表."""
    tool = GapAnalysisTool()
    papers = [
        {"title": "a", "methods": ["DFT"], "results": "improved performance", "tags": ["metal"]},
        {"title": "b", "methods": ["MD"], "results": "decreased accuracy", "tags": ["metal"]},
    ]
    result = await tool.call(
        {"action": "compare_methods", "topic": "x", "papers": papers},
        context=None,
    )
    assert result.success is True
    assert "comparison" in result.data
    assert len(result.data["comparison"]) >= 2  # DFT + MD 两组


async def test_gap_assess_novelty_max_jaccard():
    """1 假设 + 3 已知论文, novelty_score = 1 - max_jaccard."""
    tool = GapAnalysisTool()
    papers = [
        {"title": "DFT study of TiO2", "abstract": "density functional theory titanium oxide"},
        {"title": "MD of Si", "abstract": "molecular dynamics silicon"},
        {"title": "experimental GaN", "abstract": "gallium nitride synthesis"},
    ]
    # 假设几乎和第 1 篇一样, novelty 应该很低
    hypothesis = "density functional theory titanium oxide study"
    result = await tool.call(
        {
            "action": "assess_novelty",
            "topic": "x",
            "papers": papers,
            "hypotheses": [hypothesis],
        },
        context=None,
    )
    assert result.success is True
    assessment = result.data["assessments"][0]
    # 工具内部 round(novelty, 3), 这里用同样 round 对齐避免浮点抖
    assert assessment["novelty_score"] == round(
        1.0 - assessment["max_similarity"], 3
    )
    assert assessment["max_similarity"] > 0.0


async def test_gap_assess_novelty_empty_hypotheses_returns_empty():
    """缺 hypotheses 时不报错, 返回空 assessments + 提示 summary."""
    tool = GapAnalysisTool()
    result = await tool.call(
        {"action": "assess_novelty", "topic": "x", "papers": [], "hypotheses": []},
        context=None,
    )
    assert result.success is True
    assert result.data["assessments"] == []
    assert "没有待评估" in result.data["summary"]


# ════════════════════════════════════════════════════════════════════
# DOETool
# ════════════════════════════════════════════════════════════════════


async def test_doe_factorial_cartesian_product():
    """2 因子 × 2 水平 → 4 runs, 含全组合."""
    tool = DOETool()
    args = DOEInput(
        action="factorial",
        factors=[
            {"name": "T", "levels": [300, 400]},
            {"name": "P", "levels": [1, 2]},
        ],
        randomize=False,
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["n_runs"] == 4
    combos = {(r["T"], r["P"]) for r in result.data["design_matrix"]}
    assert combos == {(300, 1), (300, 2), (400, 1), (400, 2)}


async def test_doe_fractional_resolution_lookup():
    """k=4, p=1 → 查表 resolution=IV."""
    tool = DOETool()
    args = DOEInput(
        action="fractional",
        design_type="half",
        factors=[
            {"name": "A", "levels": [-1, 1]},
            {"name": "B", "levels": [-1, 1]},
            {"name": "C", "levels": [-1, 1]},
            {"name": "D", "levels": [-1, 1]},
        ],
        randomize=False,
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["resolution"] == "IV"
    assert result.data["n_runs"] == 8  # 2^(4-1)


async def test_doe_orthogonal_l8_table():
    """选 L8 表, 断言 8 runs + 自动选表."""
    tool = DOETool()
    args = DOEInput(
        action="orthogonal",
        factors=[{"name": f"f{i}", "levels": [1, 2]} for i in range(4)],
        randomize=False,
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["table_name"] == "L8"
    assert result.data["n_runs"] == 8


async def test_doe_rsm_ccd_alpha():
    """CCD 设计 α = (2^k)^0.25, 验证 alpha 字段."""
    tool = DOETool()
    args = DOEInput(
        action="rsm",
        design_type="CCD",
        factors=[
            {"name": "x", "low": -1, "high": 1},
            {"name": "y", "low": -1, "high": 1},
        ],
        center_points=3,
        randomize=False,
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    expected_alpha = (2.0 ** 2) ** 0.25  # k=2
    assert abs(result.data["alpha"] - expected_alpha) < 1e-6
    # 2^k(角点) + 2k(轴点) + center = 4 + 4 + 3 = 11
    assert result.data["n_runs"] == 11


async def test_doe_unknown_action_returns_error():
    """未知 action 走 except 分支, success=False."""
    tool = DOETool()
    # 用 dict 绕过 Literal 校验, 触发运行时 ValueError
    args = DOEInput(action="factorial", factors=[{"name": "x", "levels": [1, 2]}])
    args.action = "bogus"  # 强行改字段绕 Literal
    result = await tool.call(args, context=None)
    assert result.success is False
    assert "未知 action" in result.error


# ════════════════════════════════════════════════════════════════════
# DebuggerTool
# ════════════════════════════════════════════════════════════════════


async def test_debugger_parse_traceback_multi_frame():
    """多帧 traceback, 解析后 frames 长度 ≥ 2, error_type 非空."""
    tool = DebuggerTool()
    text = (
        'Traceback (most recent call last):\n'
        '  File "a.py", line 10, in foo\n    x.bar()\n'
        '  File "b.py", line 20, in baz\n    return x\n'
        "AttributeError: 'NoneType' object has no attribute 'bar'"
    )
    args = DebuggerInput(action="parse_traceback", traceback_text=text)
    result = await tool.call(args, context=None)
    assert result.success is True
    parsed = result.data["parsed"]
    assert len(parsed["frames"]) >= 2
    assert parsed["error_type"] == "AttributeError"


async def test_debugger_parse_traceback_syntax_error():
    """SyntaxError 特殊正则, 无 'in func' 也能抽到帧."""
    tool = DebuggerTool()
    text = (
        'Traceback (most recent call last):\n'
        '  File "x.py", line 5\n    print("hi"\n'
        "SyntaxError: unexpected EOF while parsing"
    )
    args = DebuggerInput(action="parse_traceback", traceback_text=text)
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["parsed"]["error_type"] == "SyntaxError"


async def test_debugger_analyze_root_cause_unknown_falls_back():
    """未知异常类型走 _DEFAULT_ENTRY, severity=medium."""
    tool = DebuggerTool()
    args = DebuggerInput(
        action="analyze_root_cause",
        error_message="WeirdCustomException: something odd happened",
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    analysis = result.data["analysis"]
    # 未知类型走默认条目
    assert analysis["severity"] == "medium"
    assert "未分类" in analysis["root_cause"] or analysis["root_cause"]


async def test_debugger_suggest_fix_module_not_found():
    """error_message 含 'no module named' → 反推 ModuleNotFoundError + pip install 建议."""
    tool = DebuggerTool()
    args = DebuggerInput(
        action="suggest_fix",
        error_message="No module named 'numpy'",
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    suggestions = result.data["suggestions"]
    assert len(suggestions) >= 1
    # 主修复建议应含 pip install numpy
    main_fix = suggestions[0]["fix"]
    assert "numpy" in main_fix


# ════════════════════════════════════════════════════════════════════
# DesignPlanTool
# ════════════════════════════════════════════════════════════════════


async def test_plan_propose_generates_pending_plan():
    """propose 返回 plan_id 且 status=pending."""
    tool = DesignPlanTool()
    result = await tool.call(
        {"action": "propose", "goal": "run VASP", "steps": ["prep", "submit"]},
        context=None,
    )
    assert result.success is True
    assert result.data["plan_id"].startswith("plan-")
    assert result.data["status"] == "pending"
    assert result.data["plan"]["goal"] == "run VASP"


async def test_plan_confirm_unknown_returns_false():
    """不存在的 plan_id confirm 报错."""
    tool = DesignPlanTool()
    result = await tool.call(
        {"action": "confirm", "plan_id": "plan-nonexistent"},
        context=None,
    )
    assert result.success is False
    assert "不存在" in result.error


async def test_plan_reject_clears_confirmed():
    """reject 后 has_confirmed 应为 False (清掉对应 thread 的 confirmed)."""
    tool = DesignPlanTool()
    # 先 propose + confirm
    prop = await tool.call(
        {"action": "propose", "goal": "g", "thread_id": "t1"},
        context=None,
    )
    pid = prop.data["plan_id"]
    await tool.call({"action": "confirm", "plan_id": pid, "thread_id": "t1"}, context=None)
    # reject
    await tool.call(
        {"action": "reject", "plan_id": pid, "thread_id": "t1", "reject_reason": "bad"},
        context=None,
    )
    # 查状态
    status = await tool.call(
        {"action": "status", "thread_id": "t1"}, context=None
    )
    assert status.data["has_confirmed"] is False


async def test_plan_thread_isolation():
    """A thread confirm 不应放行 B thread."""
    tool = DesignPlanTool()
    prop = await tool.call(
        {"action": "propose", "goal": "g", "thread_id": "tA"},
        context=None,
    )
    pid = prop.data["plan_id"]
    await tool.call({"action": "confirm", "plan_id": pid, "thread_id": "tA"}, context=None)
    # 查 B thread 的状态
    status_b = await tool.call(
        {"action": "status", "thread_id": "tB"}, context=None
    )
    assert status_b.data["has_confirmed"] is False
    # A thread 仍是 confirmed
    status_a = await tool.call(
        {"action": "status", "thread_id": "tA"}, context=None
    )
    assert status_a.data["has_confirmed"] is True


async def test_plan_status_no_plan_id_returns_summary():
    """status 不带 plan_id 返回 total_plans/last_confirmed/pending 摘要."""
    tool = DesignPlanTool()
    await tool.call({"action": "propose", "goal": "g1"}, context=None)
    await tool.call({"action": "propose", "goal": "g2"}, context=None)
    result = await tool.call({"action": "status"}, context=None)
    assert result.success is True
    assert result.data["total_plans"] == 2
    assert "pending" in result.data
    assert isinstance(result.data["pending"], list)


# ════════════════════════════════════════════════════════════════════
# NudgeTool
# ════════════════════════════════════════════════════════════════════


async def test_nudge_expose_params_returns_task_id():
    """expose_params 返回 task_id."""
    tool = NudgeTool()
    result = await tool.call(
        {
            "action": "expose_params",
            "task_name": "VASP run",
            "params": [{"name": "ENCUT", "current_value": 400}],
        },
        context=None,
    )
    assert result.success is True
    assert result.data["task_id"].startswith("task-")


async def test_nudge_unknown_param_returns_error():
    """nudge 不存在的 param 报错 (不是返回 None)."""
    tool = NudgeTool()
    reg = await tool.call(
        {
            "action": "expose_params",
            "task_name": "t",
            "params": [{"name": "ENCUT", "current_value": 400}],
        },
        context=None,
    )
    tid = reg.data["task_id"]
    result = await tool.call(
        {"action": "nudge", "task_id": tid, "param_name": "NONEXIST", "new_value": 500},
        context=None,
    )
    assert result.success is False
    assert "不存在" in result.error


async def test_nudge_restore_returns_checkpoint():
    """restore 返回 checkpoint_state + current_params."""
    tool = NudgeTool()
    reg = await tool.call(
        {
            "action": "expose_params",
            "task_name": "t",
            "params": [{"name": "ENCUT", "current_value": 400}],
            "checkpoint_state": {"input_file": "POSCAR"},
        },
        context=None,
    )
    tid = reg.data["task_id"]
    result = await tool.call({"action": "restore", "task_id": tid}, context=None)
    assert result.success is True
    assert result.data["checkpoint_state"] == {"input_file": "POSCAR"}
    assert result.data["current_params"]["ENCUT"] == 400


async def test_nudge_list_tasks_field_filtering():
    """list_tasks 只返回摘要字段, 不暴露内部 history."""
    tool = NudgeTool()
    await tool.call(
        {
            "action": "expose_params",
            "task_name": "t1",
            "params": [{"name": "x", "current_value": 1}],
        },
        context=None,
    )
    result = await tool.call({"action": "list_tasks"}, context=None)
    assert result.success is True
    tasks = result.data["tasks"]
    assert len(tasks) == 1
    # 摘要字段
    assert "task_id" in tasks[0]
    assert "task_name" in tasks[0]
    assert "param_count" in tasks[0]
    # 不应暴露完整 history
    assert "history" not in tasks[0]


# ════════════════════════════════════════════════════════════════════
# DesignAtomTool
# ════════════════════════════════════════════════════════════════════


async def test_atom_list_atoms_returns_all_16():
    """list_atoms 返回 16 个原子 + 4 个 category."""
    tool = DesignAtomTool()
    result = await tool.call({"action": "list_atoms"}, context=None)
    assert result.success is True
    assert result.data["total"] == 16
    cats = set(result.data["categories"])
    assert cats == {"layout", "style", "geometry", "dataviz"}


async def test_atom_render_atom_unknown_returns_comment():
    """未知 atom_name 走 _render_one 兜底, 返回注释占位 (不报错)."""
    tool = DesignAtomTool()
    # 先 validate 会拦, 直接调 _render_one 验证兜底
    code = tool._render_one("unknown.atom", {})
    assert "未知原子" in code


async def test_atom_compose_renders_each_atom():
    """compose 多原子, snippets 长度匹配."""
    tool = DesignAtomTool()
    result = await tool.call(
        {
            "action": "compose",
            "atoms": [
                {"atom_name": "layout.grid", "params": {"columns": 3}},
                {"atom_name": "style.palette", "params": {"primary": "#000"}},
            ],
        },
        context=None,
    )
    assert result.success is True
    assert result.data["count"] == 2
    assert len(result.data["snippets"]) == 2


async def test_atom_preview_auto_detects_python():
    """preview auto 模式 + dataviz 原子 → 检测为 python."""
    tool = DesignAtomTool()
    result = await tool.call(
        {
            "action": "preview",
            "atoms": [{"atom_name": "dataviz.bar", "params": {}}],
            "output_format": "auto",
        },
        context=None,
    )
    assert result.success is True
    assert result.data["format"] == "python"
    assert "matplotlib" in result.data["content"]


# ════════════════════════════════════════════════════════════════════
# GenerativeDesignTool
# ════════════════════════════════════════════════════════════════════


async def test_generative_render_only_no_file_no_run():
    """render_only 不写文件不执行 sandbox."""
    sandbox = FakeSandbox()
    tool = GenerativeDesignTool(sandbox=sandbox)
    result = await tool.call(
        {
            "action": "render_only",
            "atoms": [{"atom_name": "layout.grid", "params": {}}],
        },
        context=None,
    )
    assert result.success is True
    assert "code" in result.data
    assert len(sandbox.calls) == 0  # 没调 sandbox


async def test_generative_render_and_run_html_writes_file_no_run(tmp_path):
    """render_and_run html 模式只写文件返回路径, 不调 sandbox.run."""
    sandbox = FakeSandbox()
    tool = GenerativeDesignTool(sandbox=sandbox)
    result = await tool.call(
        {
            "action": "render_and_run",
            "atoms": [{"atom_name": "layout.grid", "params": {}}],
            "work_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["format"] == "html"
    assert "file_path" in result.data
    # 文件确实写了
    assert Path(result.data["file_path"]).exists()
    # 没调 sandbox (html 不执行)
    assert len(sandbox.calls) == 0


async def test_generative_render_and_run_python_sandbox_dict(tmp_path):
    """render_and_run python 模式 + sandbox 返回 dict → 解析 stdout/stderr/returncode."""
    sandbox = FakeSandbox(
        result={"stdout": "ok", "stderr": "", "returncode": 0, "output_files": []}
    )
    tool = GenerativeDesignTool(sandbox=sandbox)
    result = await tool.call(
        {
            "action": "render_and_run",
            "atoms": [{"atom_name": "dataviz.bar", "params": {"data": [["a", 1]]}}],
            "work_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["returncode"] == 0
    assert result.data["stdout"] == "ok"
    assert len(sandbox.calls) == 1


async def test_generative_render_and_run_sandbox_object_response(tmp_path):
    """sandbox.run 返回对象 (有 stdout 属性), 同样能解析."""
    sandbox = FakeSandbox(result=_FakeRunResult(stdout="hello", returncode=0))
    tool = GenerativeDesignTool(sandbox=sandbox)
    result = await tool.call(
        {
            "action": "render_and_run",
            "atoms": [{"atom_name": "dataviz.line", "params": {}}],
            "work_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["stdout"] == "hello"
    assert result.data["returncode"] == 0


async def test_generative_run_from_code_no_sandbox_degrades(tmp_path):
    """无 sandbox 时 run_from_code 降级: 写文件 + success=True + message 提示."""
    tool = GenerativeDesignTool(sandbox=None)
    result = await tool.call(
        {
            "action": "run_from_code",
            "code": "print('hi')",
            "language": "python",
            "work_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert "file_path" in result.data
    assert Path(result.data["file_path"]).exists()
