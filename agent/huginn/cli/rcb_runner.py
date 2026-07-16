"""RCB harness 入口: 读 workspace/INSTRUCTIONS.md, 跑 huginn agent, 输出到 stdout.

ResearchClawBench 的 TaskRunner 通过 subprocess 跑 agent_cmd, 捕获 stdout.
本脚本作为 huginn 的 RCB adapter:
  python huginn/cli/rcb_runner.py --workspace <workspace>

agent 在 workspace 里工作 (cwd=workspace), 用 code_tool/bash_tool 读写文件,
最终产出 report/report.md. RCB 的 INSTRUCTIONS.md 模板已经很详细, system
prompt 只需简短研究导向.

ponytail: 不重复 RCB prompt 已有的内容, 不加交互式渲染, 纯文本输出.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 在 import huginn 之前关掉秒级限流 — RCB 任务的 prompt 长 + 工具多,
# 默认 5000 tokens/s 会在第一轮就超限. RCB 是离线评测, 不需要限流.
os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
# 允许本地沙箱执行 code_tool/bash_tool — RCB subprocess 没有 docker, 用本地 python
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
# HUGINN_CACHE_DIR 可能被设成空串, 导致 LongTermMemory 用相对路径 memory.db,
# 在 RCB workspace cwd 下 sqlite WAL 创建失败. 强制用绝对路径.
if not os.environ.get("HUGINN_CACHE_DIR"):
    os.environ["HUGINN_CACHE_DIR"] = str(Path.home() / ".huginn")
# RCB 场景 skip CSM — 无人工 subprocess, CSM 是 noise (singularity σ₃)
os.environ.setdefault("HUGINN_SKIP_CSM", "1")
# RCB 场景 compaction 保留前 2 条 root (task + Step 1 checklist) — 修同伦断裂 (σ₂)
os.environ.setdefault("HUGINN_KEEP_ROOT_N", "2")
# RCB 场景跳过 Rust sandbox — 它在 RDKit+sklearn GPR 场景静默崩溃返回空 stderr
os.environ.setdefault("HUGINN_NO_RUST_SANDBOX", "1")


# === 认知原语: adversarial_critique ===
# ponytail: 独立 LLM 调用做 skeptical reviewer, 消除 confirmation bias.
# 治 Gap 3 (M_002 MAE=0.032eV 造假) — agent 自己写的 report, 自己 review 会护短.
# 这是 RCB/autoloop/forest 三种循环都能复用的认知原语 (治 F1 起步).
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl > 0 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def adversarial_critique(
    model: Any,
    report: str,
    checklist: str,
) -> dict[str, Any]:
    """独立 LLM 调用做 skeptical reviewer — 消除 confirmation bias.

    不让 agent 自检, 因为 agent 写的 report 它自己不会判造假.
    用独立 LLM 调用 (新 system prompt, 无对话历史) 做 adversarial review.
    返回结构化 JSON, format_critique_for_agent() 把它转成 agent 可读的修复指令.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "You are a SKEPTICAL SCIENTIFIC REVIEWER who wants to score this report LOW. "
        "Your job is to find FLAWS. Be adversarial. "
        "Output ONLY a valid JSON object. No markdown fences, no preamble."
    ))
    human = HumanMessage(content=(
        f"## Methodology Checklist (from Step 1)\n{checklist}\n\n"
        f"## Report to Critique\n{report}\n\n"
        "## Your Task (output JSON only)\n"
        "1. \"implausible_metrics\": metrics where report value is BETTER than paper baseline. "
        "Format: [{\"metric\": name, \"paper\": value, \"yours\": value, \"red_flag\": why}]. "
        "Empty list if none.\n"
        "2. \"silent_substitutions\": [EXACT] components silently replaced with simpler alternatives. "
        "Format: [{\"component\": name, \"expected\": what, \"actual\": what}]. Empty list if none.\n"
        "3. \"missing_components\": checklist items absent from report. Empty list if none.\n"
        "4. \"overall_verdict\": \"pass\" | \"fix_needed\" | \"fail\"\n\n"
        "Output ONLY the JSON object."
    ))
    try:
        resp = await model.ainvoke([system, human])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = _strip_code_fences(text)
        result = json.loads(text)
        result.setdefault("implausible_metrics", [])
        result.setdefault("silent_substitutions", [])
        result.setdefault("missing_components", [])
        result.setdefault("overall_verdict", "fix_needed")
        logger.info("adversarial_critique: verdict=%s", result["overall_verdict"])
        return result
    except Exception as e:
        logger.warning("adversarial_critique failed: %s", e)
        return {
            "implausible_metrics": [],
            "silent_substitutions": [],
            "missing_components": [],
            "overall_verdict": "fix_needed",
            "error": str(e),
        }


def format_critique_for_agent(critique: dict[str, Any]) -> str:
    """把 critique 结果格式化成 agent 可读的修复指令."""
    lines = ["ADVERSARIAL CRITIQUE RESULTS (from independent reviewer):\n"]
    verdict = critique.get("overall_verdict", "fix_needed")
    lines.append(f"Overall verdict: {verdict.upper()}\n")

    implausible = critique.get("implausible_metrics", [])
    if implausible:
        lines.append("## RED FLAG — Implausible Metrics (better than paper)")
        for m in implausible:
            lines.append(
                f"  - {m.get('metric', '?')}: paper={m.get('paper', '?')}, "
                f"yours={m.get('yours', '?')} — {m.get('red_flag', 'investigate')}"
            )
        lines.append("")

    subs = critique.get("silent_substitutions", [])
    if subs:
        lines.append("## RED FLAG — Silent Methodology Substitutions")
        for s in subs:
            lines.append(
                f"  - {s.get('component', '?')}: expected={s.get('expected', '?')}, "
                f"actual={s.get('actual', '?')}"
            )
        lines.append("")

    missing = critique.get("missing_components", [])
    if missing:
        lines.append("## Missing Components")
        for c in missing:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append("## Fix Instructions:")
    lines.append("- RED FLAG metric: investigate cause (data leakage? wrong split? bug?). "
                 "Fix the bug or document honestly.")
    lines.append("- SILENT SUBSTITUTION: implement [EXACT] component as-specified, "
                 "try >=2 approaches before giving up.")
    lines.append("- MISSING COMPONENT: implement now, or add to Limitations with error evidence.")
    lines.append("- OVERWRITE report/report.md with fixes using file_write_tool.")
    return "\n".join(lines)


async def run(workspace: str) -> int:
    ws = Path(workspace).resolve()
    instructions = ws / "INSTRUCTIONS.md"
    if not instructions.exists():
        print(f"ERROR: {instructions} not found", file=sys.stderr)
        return 1

    # RCB subprocess 跑时主 memory.db 可能被 IDE/桌面端锁定 (sqlite WAL),
    # 改用 workspace 下的独立缓存目录. RCB 是无状态离线评测, 不需要跨任务记忆.
    rcb_cache = ws / ".huginn_cache"
    rcb_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HUGINN_CACHE_DIR"] = str(rcb_cache)

    prompt = instructions.read_text(encoding="utf-8")

    from huginn.agent import HuginnAgent
    from huginn.config import HuginnConfig
    from huginn.models.registry import ModelRegistry
    from huginn.tools import register_all_tools

    # snapshot 默认用 ~/.huginn/snapshots, RCB subprocess 跑时该目录可能被
    # IDE/桌面端锁定 (PermissionError). 重定向到 workspace 下的独立目录.
    from huginn.snapshot import file_snapshot as _fs
    _fs._SNAPSHOT_ROOT = rcb_cache / "snapshots"

    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        print("ERROR: no model configured", file=sys.stderr)
        return 1

    # RCB harness 从 stdout 检测 model 名 (run_task._detect_model)
    model_name = getattr(model, "name", None) or getattr(model, "model_id", None) or str(model)
    print(f'model: {model_name}', flush=True)

    # system prompt: workspace 路径 + 工具操作事实. 让 INSTRUCTIONS.md 做 task gradient.
    # ponytail: 删 CRITICAL override 层 (σ₄) — control loop 不替 LLM 决策"什么重要".
    system_prompt = (
        f"You are an autonomous scientific research agent. "
        f"Your workspace is: {ws}\n"
        f"Current working directory IS the workspace. All relative paths "
        f"(data/, related_work/, code/, outputs/, report/) resolve from here.\n"
        f"Follow INSTRUCTIONS.md as your primary guide — it defines the task.\n"
        f"Prefer real implementations over shortcuts; document failures honestly.\n\n"
        "## Tool facts (sandbox constraints, not priorities)\n"
        "- code_tool: run Python. Sandbox BLOCKS open() and os — CANNOT write files via code_tool.\n"
        "- bash_tool: pip install, run scripts.\n"
        "- file_write_tool: CREATE or OVERWRITE text files (report.md, code/*.py). "
        "Pass FULL content each time.\n"
        "- matplotlib.savefig() WORKS (library code, not AST-scanned) — use it for figures.\n"
        "- code_tool security scanner may false-positive on eval() in torch/numpy — "
        "if so, write script via file_write_tool and run with bash_tool.\n"
        "- file_read_tool/glob/grep: explore data/ and related_work/.\n"
        "- web_search_tool: verify constants, methods, or edge cases.\n\n"
        "## Operating rules\n"
        "- Every response before task completion MUST include a tool call. "
        "Text-only response = task termination.\n"
        "- Push through errors: debug, install missing packages, try alternatives.\n"
        "- Write report/report.md EARLY, then OVERWRITE as you add results.\n"
    )

    # 先注册工具到 ToolRegistry, 再让 agent 从 registry 拉取
    register_all_tools()

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=None,
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=cfg.context_budget_tokens,
        # RCB 完整论文复现: EDA + 建模 + 反向设计 + 报告 + 验证, 80 步不够.
        # ponytail: 提高 code_tool 预算治 "budget exhausted 写不了 report" 短板.
        max_tool_calls=150,
        max_tool_calls_per_tool=50,
        # file_write_tool 写文本文件 (report.md, code/*.py);
        # code_tool 的 sandbox 禁 open(), 只能跑分析/画图 (savefig 库代码不受限).
        # 不给 file_edit_tool: 它要求文件已存在, agent 误用 edit 写新文件会失败.
        tool_filter=[
            "code_tool", "bash_tool",
            "file_read_tool", "file_write_tool",
            "glob", "grep", "web_search_tool",
        ],
        # RCB 是无人工 subprocess, 所有工具自动 approve
        auto_approve=True,
    )
    agent.register_tools_from_registry()

    # 3 步认知循环: 论文方法论提取 → 执行 → 自验证
    # ponytail: 不走 autoloop 7 阶段 (太重), 用 3 步循环治 3 个短板:
    #   Step 1 治 "不读论文就动手" — 强制先提取方法核心组件 + baseline 指标
    #   Step 2 治 "方法降级" — checklist 注入, agent 对照方法约束执行
    #   Step 3 治 "不自验证" — 对照 checklist 检查 report 覆盖度, 补缺
    # 用同一 thread_id 保持上下文连续, Step 2 能看到 Step 1 的 checklist.
    from langchain_core.messages import AIMessage

    thread_id = f"rcb_{ws.name}"

    async def _stream_chat(msg: str, step_label: str) -> str:
        """跑一轮 agent.chat, 流式打印 AIMessage, 返回最后的 AI 文本."""
        ai_text = ""
        try:
            async for chunk in agent.chat(msg, thread_id=thread_id):
                msgs = chunk.get("messages", [])
                if not msgs:
                    continue
                last = msgs[-1]
                if not isinstance(last, AIMessage):
                    continue
                content = getattr(last, "content", "")
                if content:
                    print(content, flush=True)
                    ai_text = content
        except Exception as e:
            print(f"ERROR [{step_label}]: {e}", file=sys.stderr)
        return ai_text

    # Step 1: 论文方法论提取
    # agent 读 INSTRUCTIONS.md + related_work/, 输出方法核心组件 + baseline 指标 checklist
    print("\n=== Step 1: Methodology Extraction ===\n", flush=True)
    step1_prompt = (
        f"Read the task instructions below AND explore related_work/ directory for reference papers.\n"
        f"Extract a METHODOLOGY CHECKLIST from the paper:\n"
        f"1. Core method components (model architecture, training protocol, key algorithms).\n"
        f"   For EACH component, label it [EXACT] (must reproduce as-specified) or [VARIANT]\n"
        f"   (justified deviation with reason). Default to [EXACT]. The label forces honesty\n"
        f"   about substitutions — Step 3 will audit them.\n"
        f"2. Key quantitative metrics with the paper's BASELINE VALUES (e.g. 'R²=0.79, MAE=48K').\n"
        f"   These are the targets your results will be compared against in Step 3.\n"
        f"3. Critical implementation details that must be reproduced\n\n"
        f"Output the checklist as a numbered list. Be SPECIFIC (e.g. 'CGCNNConv with gating "
        f"and residual connections', not just 'GNN'). This checklist will guide your implementation "
        f"and will be used in Step 3's substitution audit and sanity check.\n\n"
        f"Task instructions:\n{prompt}"
    )
    checklist = await _stream_chat(step1_prompt, "step1")
    print(f"\n[checklist extracted: {len(checklist)} chars]\n", flush=True)

    # Step 2: 执行任务
    # checklist 已在 thread_id 的对话历史里, agent 能看到. 不需要显式注入.
    print("\n=== Step 2: Execution ===\n", flush=True)
    step2_prompt = (
        "Now execute the task following your methodology checklist. "
        "Implement each [EXACT] component as-specified in the paper. "
        "If a component fails, debug and push through — do NOT silently substitute a simpler model. "
        "Write report/report.md with your results, referencing the checklist items you covered. "
        "Use file_write_tool for report.md, code_tool for analysis/plotting, bash_tool for running scripts."
    )
    await _stream_chat(step2_prompt, "step2")

    # Step 3: 对抗式自检 — 不是软自验证, 是 skeptical reviewer 视角
    # ponytail: 治 3 个系统性短板 (跨 4 题评分发现的共性 gap):
    #   A. sanity check — 治 "不可信结果不自检" (M_002 MAE=0.032eV 造假)
    #   B. substitution audit — 治 "沉默方法降级" (4 题全中)
    #   C. hard push — 治 "硬组件轻易放弃" (M_001 无 BO, M_003 无 graph-VAE)
    # 双层 critique: (1) 独立 LLM 调用做外部 reviewer (无 confirmation bias);
    #                (2) agent 自己再走一遍 self-critique, 对外部 critique 反应+修复.
    print("\n=== Step 3: Adversarial Self-Critique ===\n", flush=True)

    # 外部 critique: 独立 LLM 调用读 report.md + checklist, 输出结构化 red flags.
    # 失败/无 report 时降级为纯 self-critique, 不阻塞 Step 3.
    external_critique_block = ""
    report_path = ws / "report" / "report.md"
    if report_path.exists() and checklist:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            print(f"[adversarial_critique: reading {len(report_text)} chars of report.md]", flush=True)
            critique = await adversarial_critique(model, report_text, checklist)
            external_critique_block = format_critique_for_agent(critique)
            print(f"[adversarial_critique: verdict={critique.get('overall_verdict', '?')}]", flush=True)
        except Exception as e:
            print(f"[adversarial_critique: skipped due to error: {e}]", flush=True)
    else:
        print("[adversarial_critique: skipped — report.md or checklist missing]", flush=True)

    step3_prompt = (
        "ADVERSARIAL SELF-CRITIQUE. You are now a SKEPTICAL REVIEWER who wants to score this report LOW. "
        "Do NOT be lenient with yourself.\n\n"
        "## A. Sanity Check (do this FIRST — catches fabricated/impossible results)\n"
        "Read your report/report.md. Extract EVERY quantitative claim (MAE, R², accuracy, loss, etc.).\n"
        "Compare each to the paper's baseline value from your Step 1 checklist.\n"
        "Build a table: | Metric | Paper Value | Your Value | Better? |\n"
        "If ANY of your metrics is BETTER than the paper's — that is a RED FLAG.\n"
        "Investigate why: data leakage? wrong train/test split? simplified geometry? fabricated?\n"
        "Fix the bug, or honestly document the discrepancy. "
        "Implausibly good results get ZERO from reviewers.\n\n"
        "## B. Substitution Audit (catches silent methodology downgrade)\n"
        "List every [EXACT] component from your Step 1 checklist.\n"
        "For each, answer honestly: did I implement it AS-SPECIFIED, or did I substitute a simpler alternative?\n"
        "  - Substituted WITHOUT trying the real implementation → FAILURE. Implement it now.\n"
        "  - Substituted AFTER ≥2 genuine failed attempts → document the attempts with error messages.\n"
        "  'I used Random Forest instead of VAE because VAE is hard' is NOT acceptable.\n"
        "  'I used GCNConv instead of CGCNNConv because it was easier' is NOT acceptable.\n\n"
        "## C. Coverage Check\n"
        "List checklist items COVERED (with evidence from report) vs MISSING/WEAK.\n\n"
        "## D. Fix & Rewrite\n"
        "For each gap found in A/B/C:\n"
        "  - Missing metric → compute it now (run code_tool)\n"
        "  - Missing [EXACT] component → implement it (push through, try ≥2 approaches before giving up)\n"
        "  - Implausible result → fix the bug or document honestly why it's off\n"
        "OVERWRITE report/report.md with: improved results + baseline comparison table + "
        "honest Limitations section (only for items where you tried ≥2 approaches and genuinely failed).\n"
        "Use file_write_tool for the rewrite."
    )
    if external_critique_block:
        step3_prompt = external_critique_block + "\n\n## Now act on the critique above:\n" + step3_prompt
    await _stream_chat(step3_prompt, "step3")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Huginn RCB runner")
    parser.add_argument("--workspace", required=True, help="RCB workspace path")
    args = parser.parse_args()

    rc = asyncio.run(run(args.workspace))
    sys.exit(rc)


if __name__ == "__main__":
    main()
