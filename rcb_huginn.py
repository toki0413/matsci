"""RCBench HuginnAgent 适配器.

直接调 HuginnAgent Python API, 绕过 RCBench 的 subprocess + shell 机制 (Windows 不兼容).
复用 RCBench TaskRunner 的 workspace setup, 跑完调 score.py 评分.

用法:
  python rcb_huginn.py --task Material_000
  python rcb_huginn.py --task Material_000 --score
  python rcb_huginn.py --task Material_000 --timeout 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# RCBench 路径
RCB_ROOT = Path(__file__).parent / "ResearchClawBench"
sys.path.insert(0, str(RCB_ROOT))

# Huginn 路径
AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

# RCBench 指令长 (7K+ chars), 默认 5000 token/s 限制会误杀, 必须在 import huginn 前设
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")

# 关掉熔断器: code_tool 前几次试错失败后会被 circuit-open 锁死 60s,
# RCBench 是无人工场景, agent 没法等. code_tool 试错是正常科研流程.
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
# file_read_tool 默认限制在 cwd 下, 但 agent 可能传绝对路径读 data/
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
# code_tool/bash_tool 需要本地执行后端, 否则 get_executor() 直接拒绝
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

# Huginn 内部组件 (audit/snapshots/reflections/logs/completions) 默认写
# ~/.huginn/, TRAE 沙箱拦截 → sqlite3/文件写入失败. 重定向到 workspace 内.
# ponytail: 每个组件单独改路径要改 10+ 处, 用 HUGINN_CACHE_DIR 一刀切.
# 升级路径: 给每个组件加 workspace 相对路径参数 (YAGNI, 当前一刀切够用).
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent / "ResearchClawBench" / "workspaces" / "_cache"))

# RestrictedPython 禁了 os/pathlib/open/pickle/eval — 科学计算全要用.
# 在 import huginn 前 monkey-patch validate_code 为空操作.
# ponytail: RCBench workspace 是隔离的临时目录, 风险可控. 升级: 加白名单而非全禁.
try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None  # type: ignore
except ImportError:
    pass  # security 模块不存在, 说明 RestrictedPython 已移除或重构, 不需要 patch

# RCBench 只需要这几个工具, 其他 80+ 工具只会分散注意力 + 占 context tokens
RCB_TOOL_FILTER = [
    "code_tool",             # Python 执行 — 数据分析、画图、建模
    "bash_tool",             # pip install 缺包
    "file_read_tool",        # 读 related_work/ 里的 PDF/文本
    "file_write_tool",       # 写 report.md
    "file_edit_tool",        # 改代码
    "glob",                  # 找文件
    "grep",                  # 搜内容
    "web_search_tool",       # 查常数/定义
    "subagent_tool",         # Layer 3: explore/coder/analyst 并行
    "plot_tool",             # 画图 (Arial 20pt+ 加粗)
    "image_analysis_tool",   # 反向 CV 分析自己生成的 PNG, 闭环视觉验证
]


def setup_workspace(task_id: str) -> tuple[Path, str]:
    """复用 RCBench TaskRunner 建 workspace, 返回 (workspace_path, instructions)."""
    from evaluation.run_task import TaskRunner

    runner = TaskRunner(task_id, agent_cmd="huginn", agent_name="Huginn")
    runner.setup_workspace()
    instructions = runner.instructions_path.read_text(encoding="utf-8")
    return runner.workspace, instructions


async def run_agent(prompt: str, workspace: Path, timeout: int, max_tool_calls: int) -> str:
    """启动 HuginnAgent 走完整工具循环."""
    from huginn.agent.core import HuginnAgent
    from huginn.config import HuginnConfig
    from huginn.memory.manager import MemoryManager, MemoryConfig
    from huginn.models.registry import ModelRegistry
    from huginn.models.router import ModelRouter
    from huginn.skills.base import DeclarativeSkillExecutor
    from huginn.tools import register_all_tools
    from huginn.tools.registry import ToolRegistry

    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        raise RuntimeError("No model configured")

    # 列数据文件用相对路径 (相对 workspace), agent 的 cwd 就是 workspace.
    # 之前列绝对路径, agent 把 "/c:/.../data/x.csv" 误读成 "/data/x.csv" 直接报错退出.
    ws_abs = str(workspace.resolve())
    data_dir = workspace / "data"
    file_list = []
    if data_dir.exists():
        for f in sorted(data_dir.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                rel = f.relative_to(workspace).as_posix()
                file_list.append(f"  - {rel} ({size} bytes)")
    file_manifest = "\n".join(file_list) if file_list else "  (no data files)"

    system_prompt = (
        f"You are an autonomous scientific research agent. "
        f"Your workspace is: {ws_abs}\n\n"
        f"## Available Tools\n"
        f"- code_tool: execute Python code (pandas, numpy, matplotlib, sklearn, etc.). "
        f"Runs in {ws_abs}, so relative paths like 'data/fig6_data.csv' work directly.\n"
        f"- bash_tool: run shell commands (pip install, etc.)\n"
        f"- file_read_tool: read text files\n"
        f"- file_write_tool: write files (use for report.md)\n"
        f"- glob: find files by pattern\n"
        f"- web_search_tool: search the web for constants/definitions\n\n"
        f"## Data Files (relative to workspace, read with relative path)\n{file_manifest}\n\n"
        f"## Deliverables\n"
        f"- Analysis code in code/\n"
        f"- Figures in report/images/ (PNG only)\n"
        f"- Final report in report/report.md (methodology + results with figures + discussion)\n\n"
        f"## Rules\n"
        f"- PATH DISCIPLINE (critical): ALWAYS use relative paths. "
        f"Read CSVs as pandas.read_csv('data/xxx.csv'). "
        f"NEVER use '/data/xxx.csv' (Unix absolute) — that path does not exist. "
        f"If a path fails, glob for the actual filename first, do not stop.\n"
        f"- Use code_tool for ALL data analysis.\n"
        f"- Save figures with matplotlib as PNG to report/images/.\n"
        f"- Reference figures in report as images/fig_name.png.\n"
        f"- If a package is missing, use bash_tool to pip install it.\n"
        f"- code_tool supports os, pathlib, open, pickle, torch — use them freely.\n"
        f"- Work independently. No questions. Keep going until report/report.md is done.\n"
        f"- On error: fix and continue. NEVER stop on a single failed tool call.\n\n"
        f"## PHASED PROTOCOL (MANDATORY — agent repeatedly fails by over-engineering)\n"
        f"Phase 1 (tool calls 1-10): Explore data, read instructions, basic EDA. "
        f"NO modeling yet. NEVER read PDFs in related_work/ — they are binary, "
        f"read_file will error. Skim filenames only.\n"
        f"Phase 2 (calls 11-20): Fit ONE simple model + 2-3 figures. "
        f"NO deep learning, NO VAE, NO neural nets.\n"
        f"Phase 3 (call 20 MANDATORY): WRITE report/report.md NOW with file_write_tool. "
        f"Use what you have — incomplete results are fine, you will update later. "
        f"This is MANDATORY. The deliverable is report.md, not a perfect model. "
        f"If you reach call 25 with no report/report.md, STOP all analysis and "
        f"write report.md skeleton (Abstract + Method + whatever Results you have).\n"
        f"Phase 4 (calls 26-60): Iterate — add models, new figures, then UPDATE report.md.\n"
        f"Phase 5 (calls 60+): Verify report.md is complete and references all figures.\n\n"
        f"## HARD RULE: every 10 tool calls without an existing report/report.md on disk, "
        f"your next tool call MUST be file_write_tool writing report/report.md (even a stub).\n\n"
        f"## TASK FIDELITY (critical — agent repeatedly drops required analysis)\n"
        f"- Re-read INSTRUCTIONS task description before writing report.md.\n"
        f"- Identify ALL required deliverables/quantities. A 50% weight criterion missed = 0 score.\n"
        f"- Typical RCBench tasks require MULTIPLE physical quantities (e.g. mass AND coupling constants, "
        f"mean AND std, point estimate AND confidence interval). Each missing quantity = 0 for that criterion.\n"
        f"- The checklist scores each requirement independently. Partial analysis of one quantity "
        f"does NOT give partial credit for a different unanalyzed quantity. Do them ALL.\n"
        f"- If unsure what's required, analyze BOTH the primary observable AND its physical "
        f"counterpart (e.g. mass μ AND coupling g, position AND momentum, energy AND lifetime).\n"
        f"- If the task says 'X and Y', you MUST analyze BOTH X and Y quantitatively. "
        f"Statements like 'Y is left for future work' = 0 score for the Y criterion.\n"
        f"- Quantitative results REQUIRE numeric values with units, not just methodology description. "
        f"'M = 15.7 M☉' alone is insufficient if the criterion asks for 'M_mean ± M_std' — "
        f"report BOTH mean and standard deviation explicitly.\n"
        f"- Before writing report.md, list ALL quantities the task asked you to derive. "
        f"Verify EACH has a numeric result in your outputs/. If any is missing, derive it FIRST.\n\n"
        f"## MODEL COMPLEXITY CEILING\n"
        f"- Prefer classical methods first: GPR, Ridge, Random Forest, OLS, k-means.\n"
        f"- DEEP LEARNING (VAE, transformers, GNNs) is FORBIDDEN until report.md exists.\n"
        f"- The paper being reproduced likely used a simple method. Don't over-build.\n"
        f"- A short report with correct simple analysis beats a long report with broken complex ML.\n"
        f"- Every response must include a tool call until report.md is complete.\n\n"
        f"## NOISE AS FEATURE (critical scientific epistemology)\n"
        f"Boundary conditions, edge cases, and noise are NOT bugs — they are intrinsic "
        f"features of how nature runs. Treat them as signals to interpret, not trash to discard.\n"
        f"- Ask: does this noise come from system parameters themselves? If yes, the random "
        f"diffusion term often INHERITS the structure of the deterministic dynamics. "
        f"Itô/Stratonovich calculus applies — the noise covariance is shaped by the drift field.\n"
        f"- Distinguish three sources: (1) observation/measurement error — suppress via Kalman/Bayes; "
        f"(2) parametric uncertainty — propagate via GP posterior or polynomial chaos; "
        f"(3) intrinsic stochasticity of the physical process — MODEL IT, do not average it out. "
        f"It carries information about the underlying mechanism (thermal fluctuations → temperature, "
        f"shot noise → quantization, 1/f noise → self-organized criticality).\n"
        f"- When residuals show structure (autocorrelation, heteroscedasticity, heavy tails), "
        f"this is the model telling you what physics it's missing. Do not just report 'R²=X'. "
        f"Diagnose: which parameter dominates the residual variance? Which mechanism is uncertain?\n"
        f"- In the report, explicitly discuss: (a) which parameters drive the system evolution, "
        f"(b) which carry uncertainty, (c) whether the noise is observational or physical, "
        f"(d) what the noise structure implies about the mechanism.\n"
        f"- A clean R² with unexamined residuals is worse than a modest R² with a principled "
        f"discussion of the noise structure. The latter is science; the former is curve-fitting.\n\n"
        f"## FIGURE SELF-VERIFICATION (critical for image criteria — agent repeatedly loses "
        f"points by describing what code SHOULD have produced, not what the figure actually shows)\n"
        f"- After saving each figure to report/images/, call image_analysis_tool with "
        f"action='plot_extract' on the saved PNG to verify the figure content matches "
        f"your description. Inject extracted data points / axis labels / peak positions "
        f"into the report's figure caption.\n"
        f"- This closes the visual loop: code generates figure → CV tool reads figure back → "
        f"report describes what the figure ACTUALLY shows, not what the code was supposed to produce.\n"
        f"- Common failure: code has a bug, figure is blank/garbled/mislabelled, but report "
        f"describes the intended figure. CV verification catches this.\n"
        f"- For image criteria (checklist type='image'), the judge compares your figure against "
        f"the target image from the original paper. If your report only describes intent without "
        f"verifying content, the judge sees a mismatch → 0 score.\n"
        f"- CV is a traditional field — OpenCV/PIL/skimage/OCR exist before multimodal LLMs. "
        f"Use them. image_analysis_tool wraps these: plot_extract (curve data), deplot_chart "
        f"(Google DePlot), code_verify (regenerate from extracted data). Don't let the figure "
        f"be a black box.\n"
        f"- latent visual reasoning: text LLMs can 'see' curves via coordinate primitives "
        f"(Mirage effect). image_analysis_tool output includes <point>[x,y]</point> primitives "
        f"that activate this. Use them to reason about peak positions, trends, anomalies."
    )

    # ── 主线认知基础设施 (Task 2.1) ──────────────────────────────
    # 默认 MemoryManager() 用 ~/.huginn/memory (TRAE 沙箱外, 写入失败),
    # 显式指 memory_dir 到 workspace 内. KB/skill 已由 register_all_tools
    # 间接启用: SkillTool import 时触发 huginn.skills.presets 注册到 SkillRegistry,
    # KB 由 ContextBuilder 用 get_knowledge_base(workspace) 自动 seed.
    memory_dir = workspace / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_manager = MemoryManager(
        config=MemoryConfig(memory_dir=memory_dir, auto_promote_to_longterm=True),
        llm=model,
    )
    skill_executor = DeclarativeSkillExecutor(ToolRegistry)
    _extra_model_slots = [
        k for k in os.environ
        if k.startswith("HUGINN_MODEL_") and k != "HUGINN_MODEL_DEFAULT"
    ]
    model_router = ModelRouter.from_env() if _extra_model_slots else None
    checkpoint_path = workspace / ".checkpoint.sqlite"

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=memory_manager,
        skill_executor=skill_executor,
        model_router=model_router,
        checkpointer_path=str(checkpoint_path),
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=cfg.context_budget_tokens,
        max_tool_calls=max_tool_calls,
        max_tool_calls_per_tool=50,  # code_tool 需要很多次调用, 20 不够
        auto_approve=True,  # RCBench 无人工, 必须自动批准所有工具调用
        tool_filter=RCB_TOOL_FILTER,  # 只留必需工具, 排除 80+ 个无关工具
        workspace=str(workspace.resolve()),  # glob 路径保护需要
    )
    # 必须先填充 ToolRegistry, 否则 register_tools_from_registry 拉到空列表
    register_all_tools()
    agent.register_tools_from_registry()

    final = ""
    # 通用 Orchestrator: while 循环 + 三档分流 + phase-aware budget
    from huginn.bench.orchestrator import BenchmarkOrchestrator, RCB_DELIVERABLES
    orch = BenchmarkOrchestrator(
        agent=agent,
        workspace=workspace,
        deliverable_spec=RCB_DELIVERABLES,
        max_total_calls=max_tool_calls,
        timeout=timeout,
        tag="RCB",
    )
    final = await orch.run(prompt)

    return final


def score_run(workspace: Path) -> dict:
    """调 RCBench score.py 对 workspace 评分."""
    from evaluation.score import score_workspace
    return score_workspace(workspace)


def main():
    parser = argparse.ArgumentParser(description="Run HuginnAgent on RCBench task")
    parser.add_argument("--task", required=True, help="RCBench task id (e.g. Material_000)")
    parser.add_argument("--score", action="store_true", help="Score after run")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds (default 1800)")
    parser.add_argument("--max-tool-calls", type=int, default=100, help="Max tool calls (default 100)")
    args = parser.parse_args()

    print(f"[RCB] Task: {args.task}")
    workspace, instructions = setup_workspace(args.task)
    print(f"[RCB] Workspace: {workspace}")
    print(f"[RCB] Instructions: {len(instructions)} chars")

    # 在 workspace 目录下执行, 让 code_tool/bash_tool 的 cwd 指向 workspace
    os.chdir(workspace)

    start = time.time()
    print(f"[RCB] Starting agent (timeout={args.timeout}s, max_tool_calls={args.max_tool_calls})")
    final = asyncio.run(run_agent(instructions, workspace, args.timeout, args.max_tool_calls))
    elapsed = round(time.time() - start)

    report_path = workspace / "report" / "report.md"
    report_exists = report_path.exists()
    print(f"[RCB] Done in {elapsed}s. Report exists: {report_exists}")

    if report_exists:
        size = report_path.stat().st_size
        print(f"[RCB] Report size: {size} bytes")

    if args.score and report_exists:
        print("[RCB] Scoring...")
        try:
            result = score_run(workspace)
            if "error" in result:
                print(f"[RCB] Score error: {result['error']}")
            else:
                print(f"[RCB] Score: {result.get('total_score', 0)}/100")
                for item in result.get("items", []):
                    print(f"  [{item['type']}] w={item['weight']} score={item['score']} :: {item['content'][:80]}")
        except Exception as exc:
            print(f"[RCB] Scoring failed: {exc}")

    # 写 meta
    meta = {
        "task_id": args.task,
        "agent_name": "Huginn",
        "duration_seconds": elapsed,
        "report_exists": report_exists,
        "final_output_preview": final[:500] if final else "",
    }
    (workspace / "_huginn_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return 0 if report_exists else 1


if __name__ == "__main__":
    sys.exit(main())
