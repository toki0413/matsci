"""架构整洁度硬门槛 —— 把"立规降熵"变成每次跑测试都执行的机械检查.

对标上轮审计结论: 仓库治理在接缝处是干净的, 熵主要来自静默吞异常、并行重复管线、
超宽依赖面。本模块给**自研接缝**(huginn/research/ + exploration/ + validation/ +
autoloop/ + agent/ + agents/ + coder/ + rag/ + tools/sim/ + capabilities 关键文件 +
server_core)与**展示入口**立三条不靠人自觉的栅栏:

  1. 静默 catch-all 禁用: `except Exception:`/`except:`(无 `as` 绑定) 必须带 `— 原因`
     注释, 否则视为"吞异常"拦截 (禁止新的盲 except)。
  2. 深研入口唯一化: 示例/展示必须走 `huginn.research`(program/planning/science_team/
     law_model), 不得新引 `deli_research` 或再造平行管线。
  3. 依赖白名单: `[project].dependencies` 里新增顶层依赖必须先显式加进本文件白名单
     —— 阻止新可选依赖滚雪球。

这些测试是给 CI 读的硬门槛, 不是文档口号。
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # agent/
_HUGINN = _ROOT / "huginn"
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# 自研接缝: 深研运行时 + 核心认知循环 + agent 主循环 + 团队编排 + 代码/检索子系统
# + 科学计算工具封装 + 运行时 server 核心 (对这些强制"带原因的 catch-all"纪律).
# 已机械治理的模块才能进接缝 —— 新增目录同样须先清零静默盲 except, 否则本门会拦.
_OWNED_DIRS = ["research", "exploration", "validation", "autoloop", "agent",
               "agents", "coder", "rag", "tools/sim"]

_OWNED_FILES = ["huginn/capabilities/introspection.py",
                "huginn/capabilities/mcp_export.py",
                "huginn/capabilities/registry.py",
                "huginn/capabilities/world_model.py",
                "huginn/server_core.py"]

# 静态 catch-all 检测: 命中 `except Exception:` / `except BaseException:` / 裸 `except:`
_CATCHALL = re.compile(r"^\s*except\s*(BaseException|Exception)?\s*:")
# 但排除 `except Exception as exc:`(绑定了变量 → 身体会用) 以及 `except` 带 `as`
_CATCHALL_WITH_AS = re.compile(r"^\s*except\s*(BaseException|Exception)?\s+as\b")


def _owned_py_files() -> list[Path]:
    files = [_ROOT / f for f in _OWNED_FILES]
    for d in _OWNED_DIRS:
        base = _HUGINN / d
        if base.exists():
            files.extend(sorted(base.glob("*.py")))
    return [f for f in files if f.exists()]


def _has_reason(line: str) -> bool:
    """该 except 行是否带"真实原因"(中文/`—`/`→`), 而非光秃秃的 `# noqa`."""
    if "—" in line or "→" in line:
        return True
    return any("\u4e00" <= ch <= "\u9fff" for ch in line)


def test_no_silent_catchall_except_in_owned() -> None:
    """自研接缝里不得出现"不带原因的静默 catch-all"."""
    offenders: list[str] = []
    for f in _owned_py_files():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _CATCHALL_WITH_AS.match(line):
                continue                                  # 带 as 绑定, 允许
            if _CATCHALL.match(line) and not _has_reason(line):  # 吞异常且无原因
                offenders.append(f"{f.relative_to(_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "自研接缝存在静默 catch-all(未写 `— 原因`)。请补一句原因注释, 或改用 `as exc` "
        f"并在身体内处理。违规:\n" + "\n".join(offenders))


def test_no_blanket_notimplemented_in_owned() -> None:
    """自研接缝不得用 raise NotImplementedError 当占位(stub)。"""
    offenders = [f for f in _owned_py_files()
                 if "raise NotImplementedError" in f.read_text(encoding="utf-8")]
    assert not offenders, f"自研接缝不应有 NotImplementedError 占位: {offenders}"


def test_deep_research_entrypoints_use_canonical_seam() -> None:
    """展示/示例必须走 canonical 深研入口, 不得另引 deli 或再造管线."""
    assert (_HUGINN / "research/program.py").exists(), "canonical 深研入口应存在"
    # 深研展示入口(白名单), 每个都必须依赖 canonical seam 或团队成员
    showcases = ["ai4s_product_demo.py", "ai4s_arena_demo.py",
                 "ai4s_fullchain_demo.py", "ai4s_worldmodel_demo.py"]
    for name in showcases:
        f = _EXAMPLES / name
        assert f.exists(), f"展示入口缺失: {name}"
        src = f.read_text(encoding="utf-8")
        assert "research_workflow" not in src, f"{name} 不得引在线 WS 模式"
        assert "deli_research" not in src, f"{name} 不得引 Deli 实现"
        assert any(marker in src for marker in (
            "run_research_program", "ScienceTeam", "ModelBasedScienceTeam")), \
            f"{name} 未通过 canonical 深研入口 (run_research_program/ScienceTeam)"


def test_no_new_root_parallel_research_pipeline() -> None:
    """禁止在根级新增平行深研管线模块 (research_* 顶层文件)。

    根级 `research_*.py` 分两类, 白名单只放"允许留在根级的既有成员":
      - 在线 WS 研究模式入口 `research_workflow.py` (流式事件, 服务在线会话);
      - 遗留支撑基础设施 `research_budget.py` / `research_log.py` (前者限制昂贵
        工具调用次数, 后者记录猜想演化树 —— 都是**共享 IO/账本辅助**, 不是
        假说→实验→报告 的深研管线, 无自己的 pitch-science 闭环)。

    新增任何根级 `research_*.py` 都不在白名单 → 必须在此评审: 若是深研管线入口
    (pitch/实验/综合/报告任何一环) 则拒绝, 要求规划进 `huginn/research/` (program/
    planning/science_team/law_model) 统一收敛; 若是复用的共享辅助, 也应尽量并入相同
    职责模块, 而非再起一个"以 research_ 打头的平行文件"。
    """
    allowed_root = {"research_workflow.py", "research_budget.py", "research_log.py"}
    root_research = {f.name for f in _HUGINN.glob("research_*.py")}
    unexpected = sorted(root_research - allowed_root)
    assert not unexpected, (
        "根级不允许新增平行深研模块。深研能力应放进 `huginn/research/`(program/"
        "planning/science_team/law_model)。遗留 shared infra 已在白名单(research_budget/"
        "research_log); 新增文件即便只是复用辅助, 也请先并入既有职责模块。违规: "
        + ", ".join(unexpected))


def test_dependency_allowlist_blocks_unbounded_new_deps() -> None:
    """pyproject 新增顶层依赖必须先显式加进白名单(防依赖面滚雪球)。"""
    import tomllib
    with open(_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    declared = cfg["project"]["dependencies"]
    # 显式白名单: 当前已批准的顶层依赖 (新依赖需先评审、再在此显式登记)
    allowlist = {
        "pydantic", "langchain", "langchain-core", "langchain-openai",
        "langgraph", "langgraph-checkpoint-sqlite", "deepagents", "click",
        "rich", "networkx", "numpy", "scipy", "sympy", "z3-solver", "aiohttp",
        "python-dotenv", "cryptography", "pyyaml", "toml", "mcp", "fastapi",
        "uvicorn", "sse-starlette", "python-multipart", "httpx", "websockets",
        "requests", "tenacity", "Pillow",
    }

    def _pkg(dep: str) -> str:
        m = re.match(r"\s*(?:([A-Za-z0-9_.\-]+)\[[^\]]*\])|([A-Za-z0-9_.\-]+)", dep)
        raw = m.group(2) or m.group(1)
        return raw

    names = {_pkg(d) for d in declared}
    unknown = sorted(n for n in names if n and n not in allowlist)
    assert not unknown, (
        "发现未在白名单里的顶层依赖。若要引入, 请先评审其在轻量/离线/沙箱的导入成本, "
        f"再显式加入 test_arch_cleanliness.allowlist。未批准: {unknown}")


# ── 双接缝边界门: 文献层(academic/deli_research) 与 计算层(research/) 各司其职 ──
# 审计确认(2026): deli_research 是"文献→综述→论文→评审"的文档研究管线, program 是
# "假说→真实数值实验→Pareto"的可复现计算管线 —— 是两条不同产品层, 不应互相串层。
# 计算接缝已唯一化(research/), 这里再钉死"文献层不产计算、计算层不产文献"的方向性,
# 防未来新 API 误把计算深研塞进文献人才, 或把文献写作塞进计算接缝。
def test_no_academic_literature_layer_imports_compute_research() -> None:
    """文献层 (academic/) 不得 import 计算深研接缝 (huginn.research.*)。

    文献流水线依赖检索/编排内核(empty autoloop.engine、kg/rag)做 LLM 协同写作与引用校验;
    它一旦 import research.program/science_team/law_model, 就是想把"可复现数值闭环"揉进
    文档产出的第一信号 —— 串层。计算能力应单独对接, 不内嵌进文献人才。
    """
    import re as _re
    academic_dir = _HUGINN / "academic"
    compute_seam = _re.compile(r"from\s+huginn\.research\b|import\s+huginn\.research\b")
    offenders = []
    for f in sorted(academic_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        if compute_seam.search(f.read_text(encoding="utf-8")):
            offenders.append(f.relative_to(_ROOT).as_posix())
    assert not offenders, (
        "文献层 (academic/) 串了计算深研接缝 (huginn.research.*)。"
        "两条产品层应各司其职: 文献层做文档/评审, 计算层做可复现数值闭环。"
        "新增计算能力应单独对接, 不要 import 计算接缝进文献人才文件。违规: "
        + ", ".join(offenders))


def test_no_compute_research_imports_academic_literature() -> None:
    """计算深研接缝 (research/) 不得 import 文献层 (huginn.academic.*)。

    research/ 是可复现科学计算的纯接缝(零相对导入、近叶子); 一旦它 import
    academic/deli_research, 就背着"文档写作"的隐重, 破坏其纯计算可移植性。
    """
    import re as _re
    compute_research_dir = _HUGINN / "research"
    lit_seam = _re.compile(r"from\s+huginn\.academic\b|import\s+huginn\.academic\b")
    offenders = []
    for f in sorted(compute_research_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        if lit_seam.search(f.read_text(encoding="utf-8")):
            offenders.append(f.relative_to(_ROOT).as_posix())
    assert not offenders, (
        "计算深研接缝 (research/) 串了文献层 (huginn.academic.*)。"
        "计算接缝是纯计算叶子, 不应背文档写作的隐重。违规: "
        + ", ".join(offenders))