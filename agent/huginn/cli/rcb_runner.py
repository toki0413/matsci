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
# RCB 场景用 CSM 子集: 3-step 映射 S1/S4/S6+S7, 不再全 skip (Task 18, R8 减法修正).
# ponytail: S7 自修改仍走 (Task 2), 只跳过 compaction — 见 reflection.py L245.
os.environ.setdefault("HUGINN_RCB_CSM_SUBSET", "1")
# RCB 场景 compaction 保留前 2 条 root (task + Step 1 checklist) — 修同伦断裂 (σ₂)
os.environ.setdefault("HUGINN_KEEP_ROOT_N", "2")
# RCB 场景跳过 Rust sandbox — 它在 RDKit+sklearn GPR 场景静默崩溃返回空 stderr
os.environ.setdefault("HUGINN_NO_RUST_SANDBOX", "1")
# RCB 场景关熔断器 — file_read_tool 误触发 circuit_open 阻止 agent 读文件 (σ₇)
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
# RCB 场景关循环检测 — agent 反复跑 code_tool 是正常行为, 误判为 loop (σ₈)
os.environ.setdefault("HUGINN_SKIP_LOOP_DETECTOR", "1")


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
    model: Any = None,
    report: str = "",
    checklist: str = "",
    *,
    mode: str = "object",
    proposal: str = "",
    system_prompt_summary: str = "",
    recent_rejections: list[str] | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """独立 LLM 调用做 skeptical reviewer — 消除 confirmation bias.

    mode="object": critique report (原逻辑不变)
    mode="meta": critique agent 自修改提案 (L2 元元认知层, 哥德尔机 proof
                 verifier 弱化版 — LLM judge 替代形式化 proof)

    不让 agent 自检, 因为 agent 写的 report 它自己不会判造假.
    用独立 LLM 调用 (新 system prompt, 无对话历史) 做 adversarial review.
    返回结构化 JSON, format_critique_for_agent() 把它转成 agent 可读的修复指令.
    """
    if mode == "meta":
        import difflib
        # 早期拒绝查重 — 命中直接返回, 不调 LLM 省 token
        # ponytail: 天花板是 difflib 字符串相似度 (抓不到同义改写这类语义近义,
        #           "always use X" 改写成 "X must be used" 漏判);
        #           升级路径换 embedding 相似度 (sentence-transformers cosine > 0.85)
        for prev in recent_rejections or []:
            ratio = difflib.SequenceMatcher(None, proposal, prev).ratio()
            if ratio > 0.8:
                logger.info("meta_critique early_reject: similarity=%.2f", ratio)
                return {
                    "verdict": "reject",
                    "reason": f"similar to past rejection (similarity={ratio:.2f})",
                    "expected_utility_delta": 0.0,
                    "early_reject": True,
                }
        # 调 LLM 做完整评估 — 复用现有 ainvoke 模式, 同 client 不同 system prompt
        from langchain_core.messages import HumanMessage, SystemMessage
        client = llm_client if llm_client is not None else model
        meta_system = SystemMessage(content=(
            "你是 META-REVIEWER 评估 agent 的自修改提案.\n"
            "你的工作是判断这个提案是否会真正改进 agent 的效用, "
            "还是会污染 gradient 或引入坏习惯.\n\n"
            "评估维度:\n"
            "1. 是否污染 gradient (例如加 CRITICAL: always use X) — 这是 σ₄ lesson\n"
            "2. 是否与最近 rejection 相似 (相似度 > 0.8 直接 reject)\n"
            "3. 是否与现有 stable_principles 冲突\n"
            "4. expected_utility_delta 是否为正\n\n"
            "输出严格 JSON: "
            '{"verdict": "accept"|"reject", "reason": "...", "expected_utility_delta": float}'
        ))
        rejections_block = "\n".join(f"- {r}" for r in (recent_rejections or [])) or "(none)"
        meta_human = HumanMessage(content=(
            f"## Proposal\n{proposal}\n\n"
            f"## Current system prompt summary\n{system_prompt_summary or '(empty)'}\n\n"
            f"## Recent rejections (do not repeat)\n{rejections_block}\n\n"
            "Output ONLY the JSON object."
        ))
        try:
            resp = await client.ainvoke([meta_system, meta_human])
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = _strip_code_fences(text)
            result = json.loads(text)
            result.setdefault("verdict", "reject")
            result.setdefault("reason", "no reason provided")
            result.setdefault("expected_utility_delta", 0.0)
            result["early_reject"] = False
            logger.info("meta_critique: verdict=%s", result["verdict"])
            return result
        except Exception as e:
            logger.warning("meta_critique failed: %s", e)
            return {
                "verdict": "reject",
                "reason": f"meta_critique error: {e}",
                "expected_utility_delta": 0.0,
                "early_reject": False,
                "error": str(e),
            }

    # === object mode (原逻辑不变) ===
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
            "self_observe",
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

    # RCB 3-step 映射 CSM: Step1→S1_DISCOVER, Step2→S4_CONSTRUCT, Step3→S6+S7 (Task 18)
    # ponytail: transition 是 advisory — 不允许就 no-op, 不破坏现有 3-step 流程.
    from huginn.cognitive_engine import TransitionSignal as _RCB_TS

    def _rcb_csm_advance(signal_type: str, ctx: dict) -> None:
        """RCB step 开始时手动推 CSM 状态. advisory: 不允许就 no-op."""
        csm = getattr(agent, "_csm", None)
        if csm is None:
            return
        try:
            csm.transition(_RCB_TS(signal_type, ctx))
        except Exception:
            logger.debug("RCB CSM transition failed", exc_info=True)

    # Step 1: 论文方法论提取
    # agent 读 INSTRUCTIONS.md + related_work/, 输出方法核心组件 + baseline 指标 checklist
    print("\n=== Step 1: Methodology Extraction ===\n", flush=True)
    _rcb_csm_advance("user_goal", {"goal": "understand problem and extract methodology"})
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
    _rcb_csm_advance("user_confirmed", {"plan": "execute methodology checklist"})
    step2_prompt = (
        "Now execute the task following your methodology checklist. "
        "Implement each [EXACT] component as-specified in the paper. "
        "If a component fails, debug and push through — do NOT silently substitute a simpler model. "
        "Write report/report.md with your results, referencing the checklist items you covered. "
        "Use file_write_tool for report.md, code_tool for analysis/plotting, bash_tool for running scripts."
    )
    await _stream_chat(step2_prompt, "step2")

    # Step 2.5: report.md 兜底 (σ₆ 修复)
    # 减 CSM (σ₃) 后失去 completion guidance, 加 lightweight gate 补 harmonic.
    # agent 可能在 Step 2 提前终止 (text-only response), 没写 report.md.
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        print("\n=== Step 2.5: report.md Emergency Write ===\n", flush=True)
        await _stream_chat(
            "CRITICAL: report/report.md does NOT exist. Session scores ZERO without it.\n"
            "Write report/report.md NOW using file_write_tool. Base it on:\n"
            "- Your Step 1 methodology checklist\n"
            "- Your code in code/ and results in outputs/\n"
            "Minimum: # Title, ## Methodology, ## Results (images/*.png), ## Discussion.\n"
            "Be HONEST. A short honest report beats no report. Write it NOW.",
            "step2.5"
        )
    # Deterministic fallback: agent 仍不写就自动生成, 确保有交付物评分
    if not report_path.exists():
        print("[fallback: auto-generating minimal report.md]", flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _metrics_parts = []
        for _p in (ws / "outputs").glob("*.json"):
            try:
                _metrics_parts.append(f"### {_p.name}\n```json\n{_p.read_text(encoding='utf-8')}\n```")
            except Exception:
                pass
        _metrics = "\n".join(_metrics_parts) or "None"
        _imgs_dir = ws / "report" / "images"
        _imgs = "\n".join(f"![{p.name}](images/{p.name})" for p in _imgs_dir.glob("*.png")) or "None" if _imgs_dir.exists() else "None"
        _code_dir = ws / "code"
        _code = "\n".join(f"- `{p.name}`" for p in _code_dir.glob("*.py")) or "None" if _code_dir.exists() else "None"
        report_path.write_text(
            f"# Research Report (Auto-generated Fallback)\n\n"
            f"## Methodology\nAgent did not write report.md; auto-generated from artifacts.\n\n"
            f"### Code\n{_code}\n\n### Metrics\n{_metrics}\n\n## Results\n{_imgs}\n",
            encoding="utf-8"
        )

    # Step 3: 对抗式自检 — 不是软自验证, 是 skeptical reviewer 视角
    # ponytail: 治 3 个系统性短板 (跨 4 题评分发现的共性 gap):
    #   A. sanity check — 治 "不可信结果不自检" (M_002 MAE=0.032eV 造假)
    #   B. substitution audit — 治 "沉默方法降级" (4 题全中)
    #   C. hard push — 治 "硬组件轻易放弃" (M_001 无 BO, M_003 无 graph-VAE)
    # 双层 critique (Task 21): object mode (report) + meta mode (directive).
    #   - Layer 1 (object): 独立 LLM 调用读 report.md + checklist → red flags, 直接反馈 agent.
    #   - Layer 2 (meta): reflection._handle_s7_self_modify 在 S7 状态自动调 (Task 2),
    #                     评估 agent 自修改 proposal, accept→stable_principle / reject→rejection log.
    #   object + meta 共享同一 LLM 实例 (model 参数), 不同 system prompt.
    #   合并: object verdict 进 step3_prompt (本轮修复); meta verdict 走 sidecar (下轮 system_prompt).
    print("\n=== Step 3: Adversarial Self-Critique ===\n", flush=True)
    # S6_FEEDBACK: critique 视角找 gap; reflection 检测到实质 gap 时自动进 S7 (Task 2)
    _rcb_csm_advance("tool_failure", {"reason": "adversarial critique — find gaps"})

    # Layer 1 — object mode: 独立 LLM 调用读 report.md + checklist, 输出结构化 red flags.
    # 失败/无 report 时降级为纯 self-critique, 不阻塞 Step 3.
    external_critique_block = ""
    object_verdict = None
    if report_path.exists() and checklist:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            print(f"[adversarial_critique: reading {len(report_text)} chars of report.md]", flush=True)
            object_verdict = await adversarial_critique(
                model, report_text, checklist, mode="object",
            )
            external_critique_block = format_critique_for_agent(object_verdict)
            print(f"[adversarial_critique: verdict={object_verdict.get('overall_verdict', '?')}]", flush=True)
        except Exception as e:
            print(f"[adversarial_critique: skipped due to error: {e}]", flush=True)
    else:
        print("[adversarial_critique: skipped — report.md or checklist missing]", flush=True)

    # Layer 2 — meta mode: 触发 CSM 进 S6_FEEDBACK → S7_SELF_MODIFY,
    # reflection._handle_s7_self_modify 自动调 adversarial_critique(mode="meta").
    # ponytail: Task 18 改用 RCB_CSM_SUBSET (不再全 skip CSM), reflection loop
    #           现在正常运行, S7 handler 会被 reflection 自动调. 这里显式 trigger 是
    #           belt-and-suspenders — 确保 object_verdict 非 pass 时一定进 S7.
    try:
        from huginn.cognitive_engine import TransitionSignal, CognitiveState
        csm = getattr(agent, "_csm", None)
        if csm is not None and object_verdict is not None:
            verdict_flag = object_verdict.get("overall_verdict", "fix_needed")
            # 非 pass 视作 gap 信号, 触发 S6_FEEDBACK (Task 18 显式触发点)
            sig = "tool_failure" if verdict_flag != "pass" else "tool_success"
            new_state = csm.transition(TransitionSignal(sig, {
                "objective": "step3_critique",
                "result_summary": f"object_verdict={verdict_flag}",
            }))
            # S6 + 实质 gap → S7_SELF_MODIFY, reflection 自动调 meta mode (Task 2)
            if new_state == CognitiveState.S6_FEEDBACK and verdict_flag != "pass":
                csm.transition(TransitionSignal("gap_found", {
                    "gap": external_critique_block[:200] or "step3 object critique red flags",
                }))
    except Exception:
        logger.debug("Step 3 CSM S6/S7 trigger failed", exc_info=True)

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
    if "--self-check" in sys.argv:
        # Task 3 self-check: meta mode 早期拒绝 (不调 LLM)
        # ponytail: 命中查重直接返回, llm_client=None 也能跑, 验证 ponytail 优化没退化.
        # 用 asyncio.run 包裹因 adversarial_critique 是 async (object mode 调用点 L434 依赖)
        rejections = ["always use Tanimoto kernel for GP", "add CRITICAL: never use RBF"]
        proposal = "always use Tanimoto kernel for GP regression"
        result = asyncio.run(adversarial_critique(
            mode="meta",
            proposal=proposal,
            recent_rejections=rejections,
            system_prompt_summary="",
            llm_client=None,
        ))
        assert result["verdict"] == "reject", f"expected reject, got {result}"
        assert result.get("early_reject") is True, "should be early_reject"
        print("Task 3 self-check PASS")
        sys.exit(0)
    main()
