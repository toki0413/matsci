"""ScienceAgentBench (SAB) HuginnAgent 适配器.

SAB 原生需要 SharePoint 下载完整 benchmark (含 datasets/gold_programs/eval_programs).
Windows + 无 SharePoint 访问时, 本适配器用 HF 上的 annotation CSV:
- 加载 ScienceAgentBench.csv (102 tasks, 4 domains)
- 给 agent task_inst + dataset_preview + domain_knowledge
- agent 写一个完整 Python 程序到 pred_<gold_program_name>
- 评分: LLM judge 评估代码质量 (无法执行因为缺真实数据集)

注: 这不是 SAB 原生的 success_rate 评测, 而是 code-quality 评测.
     SharePoint 数据下载后可切换到执行模式.

用法:
  python sab_huginn.py --instance 1 --score
  python sab_huginn.py --domain "Computational Chemistry" --n 3 --score
  python sab_huginn.py --instance 1 --score --timeout 1200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

SAB_ROOT = Path(__file__).parent / "ScienceAgentBench"
sys.path.insert(0, str(SAB_ROOT))

SAB_CACHE = Path(__file__).parent / ".cache" / "sab"
SAB_CSV = SAB_CACHE / "ScienceAgentBench.csv"

os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None  # type: ignore
except ImportError:
    pass

# reasoner (R1) judge 输出含 <think>...</think>, str2dict 找不到 JSON 边界.
# 剥离 think 块, 只留 </think> 后的 JSON.
try:
    import structai.llm_api as _sai
    _orig_str2dict = _sai.str2dict
    _sai.str2dict = lambda s: _orig_str2dict(
        s.split("</think>", 1)[-1] if "</think>" in s else s
    )
except ImportError:
    pass

# ponytail: 不含 file_write_tool/file_edit_tool — 它们在 Windows 上路径解析有 bug
# (agent 传 /workspace/xxx 虚拟路径, 实际写到别处). 用 code_tool 的 open() 写文件更稳.
SAB_TOOL_FILTER = [
    "code_tool",
    "bash_tool",
    "file_read_tool",
    "glob",
    "grep",
    "web_search_tool",
    "subagent_tool",
    "plot_tool",       # 画图 (Arial 20pt+ 加粗)
    # P1-B2: 恢复数学工具 — SAB 有物理/化学数值题, 量纲检查 + 符号推导直接拿分
    "symbolic_math_tool",
    "unit_tool",
    "validate_tool",
]


def load_sab_tasks() -> list[dict]:
    """加载 SAB annotation CSV, 返回 task dict 列表."""
    if not SAB_CSV.exists():
        raise FileNotFoundError(
            f"SAB CSV not found at {SAB_CSV}. Run `python _sab_download.py` first."
        )
    tasks = []
    with open(SAB_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 类型转换
            row["instance_id"] = int(row["instance_id"])
            tasks.append(row)
    return tasks


def setup_workspace(task: dict, workspace: Path) -> dict:
    """建 workspace, 写 task 信息 + dataset_preview."""
    workspace.mkdir(parents=True, exist_ok=True)

    instance_id = task["instance_id"]
    meta = {
        "instance_id": instance_id,
        "domain": task.get("domain", ""),
        "subtask_categories": task.get("subtask_categories", ""),
        "github_name": task.get("github_name", ""),
        "gold_program_name": task.get("gold_program_name", f"task_{instance_id}.py"),
        "output_fname": task.get("output_fname", ""),
        "task_inst": task.get("task_inst", ""),
    }

    # task.md: 任务说明
    (workspace / "task.md").write_text(
        f"# SAB Task #{instance_id}\n\n"
        f"## Domain\n{meta['domain']}\n\n"
        f"## Subtask Categories\n{meta['subtask_categories']}\n\n"
        f"## Source Repo\n{meta['github_name']}\n\n"
        f"## Task Instruction\n{meta['task_inst']}\n\n"
        f"## Expected Output\nSave to: `{meta['output_fname']}`\n\n"
        f"## Gold Program Name\n{meta['gold_program_name']}\n",
        encoding="utf-8",
    )

    # dataset_preview.md: 数据预览
    preview = task.get("dataset_preview", "")
    (workspace / "dataset_preview.md").write_text(
        f"# Dataset Preview (Task #{instance_id})\n\n```\n{preview}\n```\n",
        encoding="utf-8",
    )

    # dataset_folder_tree.md
    tree = task.get("dataset_folder_tree", "")
    (workspace / "dataset_folder_tree.md").write_text(
        f"# Dataset Folder Tree\n\n```\n{tree}\n```\n",
        encoding="utf-8",
    )

    # domain_knowledge.md (可选)
    knowledge = task.get("domain_knowledge", "")
    if knowledge and str(knowledge).strip() != "nan":
        (workspace / "domain_knowledge.md").write_text(
            f"# Domain Knowledge\n\n{knowledge}\n",
            encoding="utf-8",
        )
        meta["has_domain_knowledge"] = True

    return meta


def build_system_prompt(workspace: Path, meta: dict) -> str:
    """SAB 指令 + Phased Protocol + NOISE AS FEATURE."""
    ws_abs = str(workspace.resolve())
    task_inst = meta["task_inst"]
    output_fname = meta["output_fname"]
    gold_name = meta["gold_program_name"]

    return (
        f"You are an expert Python programming assistant for scientific research.\n"
        f"Workspace: {ws_abs}\n\n"
        f"## Task\n"
        f"{task_inst}\n\n"
        f"## Available Context Files (read with file_read_tool)\n"
        f"- task.md — full task description\n"
        f"- dataset_preview.md — preview of the dataset (column names + sample rows)\n"
        f"- dataset_folder_tree.md — directory structure of the dataset\n"
        f"- domain_knowledge.md — expert knowledge for this task (if present)\n\n"
        f"## Deliverable\n"
        f"Write a COMPLETE, self-contained Python program to: pred_{gold_name}\n"
        f"The program must save its output to: {output_fname}\n"
        f"Use RELATIVE paths in the program (e.g. 'data/input.csv' not '/home/data/...')\n\n"
        f"## CRITICAL: How to Write Files\n"
        f"Use code_tool with Python's open() to write files. Example:\n"
        f"  code_tool: code = '...' \\n with open('pred_{gold_name}', 'w') as f: f.write(code)\n"
        f"Do NOT use file_write_tool — it has path resolution bugs on Windows.\n\n"
        f"## Available Tools\n"
        f"- code_tool: execute Python (pandas, numpy, sklearn, rdkit, etc.) — USE THIS TO WRITE FILES\n"
        f"- bash_tool: pip install missing packages (install first, don't check repeatedly)\n"
        f"- file_read_tool: read text files\n"
        f"- glob / grep\n"
        f"- web_search_tool: search for package docs, API references\n\n"
        f"## PHASED PROTOCOL\n"
        f"Phase 1 (calls 1-5): Read task.md, dataset_preview.md, domain_knowledge.md. "
        f"Understand inputs, expected output, and metric.\n"
        f"Phase 2 (calls 6-15): Write a working program skeleton to pred_{gold_name}. "
        f"Use dataset_preview columns. Mock minimal input data for testing if needed.\n"
        f"Phase 3 (calls 16+): Refine — add error handling, edge cases, "
        f"ensure output format matches expected. Test with code_tool.\n"
        f"Phase 4 (final): Verify pred_{gold_name} is complete and runnable.\n\n"
        f"## Rules\n"
        f"- PATH DISCIPLINE: relative paths only.\n"
        f"- The real dataset is NOT available (only preview). Write code that WOULD work "
        f"if the dataset were present at the paths shown in dataset_folder_tree.md. "
        f"You may create small mock data from the preview for testing.\n"
        f"- A simple correct program beats a broken complex one.\n"
        f"- On error: fix and continue. NEVER stop on a single failed tool call.\n"
        f"- Every response must include a tool call until pred_{gold_name} is complete.\n\n"
        f"## NOISE AS FEATURE (scientific epistemology)\n"
        f"Boundary conditions, edge cases, and noise are NOT bugs — they are intrinsic "
        f"features of how nature runs.\n"
        f"- Missing values (NaNs) often carry signal: structural missingness vs random. "
        f"Encode it, don't just drop it.\n"
        f"- Distinguish: observation noise (filter) vs intrinsic stochasticity (model).\n"
        f"- If the task involves stochastic methods (MCMC, diffusion, sampling), preserve "
        f"the noise structure faithfully. Itô/Stratonovich calculus applies.\n"
        f"- Add a brief comment in the program explaining which noise sources you handle "
        f"and how. This is scientific hygiene, not optional.\n"
    )


async def run_agent(workspace: Path, meta: dict, timeout: int, max_tool_calls: int) -> str:
    """启动 HuginnAgent 跑 SAB 任务."""
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

    system_prompt = build_system_prompt(workspace, meta)

    # ── 主线认知基础设施 (Task 2.4) ──────────────────────────────
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
        max_tool_calls_per_tool=50,
        auto_approve=True,
        tool_filter=SAB_TOOL_FILTER,
        workspace=str(workspace.resolve()),
    )
    register_all_tools()
    agent.register_tools_from_registry()

    instance_id = meta["instance_id"]
    prompt = (
        f"Solve SAB task #{instance_id}.\n\n"
        f"Read task.md and dataset_preview.md first.\n"
        f"Write a complete Python program to pred_{meta['gold_program_name']}.\n"
        f"Start by understanding the task and dataset, then code, then test.\n"
    )

    final = ""
    # 通用 Orchestrator: while 循环 + 三档分流
    from huginn.bench.orchestrator import BenchmarkOrchestrator, SAB_DELIVERABLES
    orch = BenchmarkOrchestrator(
        agent=agent,
        workspace=workspace,
        deliverable_spec=SAB_DELIVERABLES,
        max_total_calls=max_tool_calls,
        timeout=timeout,
        tag="SAB",
    )
    final = await orch.run(prompt)
    return final


def score_submission(workspace: Path, meta: dict) -> dict:
    """LLM judge 评估 agent 生成的代码质量."""
    from judge_helper import judge_score

    pred_name = f"pred_{meta['gold_program_name']}"
    pred_path = workspace / pred_name
    if not pred_path.exists():
        return {"error": f"No {pred_name} found", "score": 0.0}

    code = pred_path.read_text(encoding="utf-8", errors="replace")

    # P0-1: 不截断. 之前 code[:8000] 让 judge 只见 59.5% 文件,
    # 评语"truncated/无main/未保存"全是测量伪影.
    judge_input_chars = len(code)
    prompt = (
        f"## Task\n{meta['task_inst']}\n\n"
        f"## Expected Output\n{meta['output_fname']}\n\n"
        f"## Submitted Program ({pred_name}, {judge_input_chars} chars)\n```python\n{code}\n```\n\n"
        f"## Grading Criteria\n"
        f"1. Does the program implement the task correctly? (0-40)\n"
        f"2. Does it save output to the correct path in the correct format? (0-20)\n"
        f"3. Code quality: error handling, readability, no hardcoded paths? (0-20)\n"
        f"4. Scientific soundness: handles noise/missing values/edge cases? (0-20)\n\n"
        f"Return JSON: {{\"reasoning\": \"...\", \"score\": 0-100, "
        f"\"breakdown\": {{\"correctness\": 0-40, \"output\": 0-20, "
        f"\"quality\": 0-20, \"science\": 0-20}}}}"
    )

    result = judge_score(
        prompt,
        system_prompt=(
            "You are grading a scientific Python program. Score it on: "
            "(1) correctness of approach, (2) output format adherence, "
            "(3) code quality (error handling, readability), "
            "(4) scientific soundness (noise handling, edge cases). "
            "Be strict. The program cannot be executed (no real dataset), "
            "so judge by code inspection."
        ),
        max_tokens=4096,
        max_try=3,
    )

    if isinstance(result, dict):
        score = max(0, min(100, int(result.get("score", 0))))
        reasoning = str(result.get("reasoning", ""))
        breakdown = result.get("breakdown", {})
    else:
        score = 0
        reasoning = "judge returned None"
        breakdown = {}

    return {
        "instance_id": meta["instance_id"],
        "domain": meta["domain"],
        "gold_program_name": meta["gold_program_name"],
        "pred_lines": len(code.splitlines()),
        "judge_input_chars": judge_input_chars,
        "judge_model": os.environ.get("JUDGE_MODEL_NAME", "deepseek-v4-flash"),
        "score": score,
        "breakdown": breakdown,
        # P0-1: 保留 2000+ 字符, judge max_tokens=4096 足以产出长评语
        "reasoning": reasoning[:4000],
    }


def main():
    parser = argparse.ArgumentParser(description="Run HuginnAgent on ScienceAgentBench task")
    parser.add_argument("--instance", type=int, default=None, help="SAB instance id (1-102)")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain")
    parser.add_argument("--n", type=int, default=1, help="Number of tasks to run (with --domain)")
    parser.add_argument("--workspace", default=None, help="Workspace dir")
    parser.add_argument("--score", action="store_true", help="Score after run")
    parser.add_argument("--timeout", type=int, default=1200)
    # P0-3: 40→60 对齐 1200s 超时额; 之前 40 + pred_*.txt 强制烧满
    parser.add_argument("--max-tool-calls", type=int, default=60)
    args = parser.parse_args()

    tasks = load_sab_tasks()
    print(f"[SAB] Loaded {len(tasks)} tasks from {SAB_CSV}")

    if args.instance:
        selected = [t for t in tasks if t["instance_id"] == args.instance]
        if not selected:
            print(f"[SAB] Instance {args.instance} not found")
            return 1
    elif args.domain:
        selected = [t for t in tasks if args.domain.lower() in t.get("domain", "").lower()]
        selected = selected[:args.n]
    else:
        print("[SAB] Must specify --instance or --domain")
        return 1

    all_results = []
    for task in selected:
        instance_id = task["instance_id"]
        workspace = Path(args.workspace) if args.workspace else (
            Path.cwd() / "workspaces" / "sab" / f"task_{instance_id}"
        )
        workspace = workspace.resolve()

        print(f"\n[SAB] Task #{instance_id}: {task.get('domain', '?')}")
        print(f"[SAB] Workspace: {workspace}")

        meta = setup_workspace(task, workspace)
        print(f"[SAB] Task: {meta['task_inst'][:120]}")

        os.chdir(workspace)

        start = time.time()
        print(f"[SAB] Starting agent (timeout={args.timeout}s, max_tool_calls={args.max_tool_calls})")
        final = asyncio.run(run_agent(workspace, meta, args.timeout, args.max_tool_calls))
        elapsed = round(time.time() - start)

        pred_name = f"pred_{meta['gold_program_name']}"
        pred_path = workspace / pred_name
        print(f"[SAB] Done in {elapsed}s. {pred_name} exists: {pred_path.exists()}")

        if args.score:
            print("[SAB] Scoring...")
            try:
                result = score_submission(workspace, meta)
                (workspace / "_score.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )
                if "error" in result:
                    print(f"[SAB] Score error: {result['error']}")
                else:
                    print(f"[SAB] Score: {result['score']}/100")
                    if result.get("breakdown"):
                        bd = result["breakdown"]
                        print(f"  correctness={bd.get('correctness', 0)}/40 "
                              f"output={bd.get('output', 0)}/20 "
                              f"quality={bd.get('quality', 0)}/20 "
                              f"science={bd.get('science', 0)}/20")
                all_results.append(result)
            except Exception as exc:
                print(f"[SAB] Scoring failed: {exc}")

        from huginn.config import config_fingerprint
        meta_out = {
            "instance_id": instance_id,
            "domain": meta["domain"],
            "duration_seconds": elapsed,
            "pred_exists": pred_path.exists(),
            "final_output_preview": final[:500] if final else "",
            # P0-6: 模型配置显性化
            "agent_model": os.environ.get("HUGINN_MODEL", "unknown"),
            "agent_provider": os.environ.get("HUGINN_PROVIDER", "default"),
            "judge_model": os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat"),
            "config_hash": config_fingerprint(),
        }
        (workspace / "_huginn_meta.json").write_text(
            json.dumps(meta_out, indent=2, ensure_ascii=False)
        )

    if args.score and all_results:
        # 多任务汇总
        avg = sum(r.get("score", 0) for r in all_results) / len(all_results)
        print(f"\n[SAB] Average score over {len(all_results)} tasks: {avg:.2f}/100")
        by_domain: dict[str, list[float]] = {}
        for r in all_results:
            d = r.get("domain", "unknown")
            by_domain.setdefault(d, []).append(r.get("score", 0))
        for d, scores in by_domain.items():
            print(f"  {d}: avg={sum(scores)/len(scores):.2f} (n={len(scores)})")

    return 0 if all_results else (0 if selected else 1)


if __name__ == "__main__":
    sys.exit(main())
