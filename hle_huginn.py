"""HLE (Humanity's Last Exam) HuginnAgent 适配器.

HLE 是 CAIS/Scale AI 发布的 3000+ 道专家级超难题, 测试 AI 跨学科深度推理.
本脚本用 HuginnAgent 走完整工具循环解题, 支持:
- 多选题: exact match (A/B/C/D/E)
- 简答题: LLM-as-judge 评分
- 含图题: agent 内视觉路由自动处理

数据集: https://huggingface.co/datasets/cais/hle
用法:
  python hle_huginn.py --n 20                    # 跑前 20 题
  python hle_huginn.py --domain biology          # 只跑生物域
  python hle_huginn.py --n 50 --judge            # 跑 50 题并自动评分
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

# 同 rcb_huginn.py 的环境配置
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None
except ImportError:
    pass

# ponytail: 去掉 file_write_tool — Windows 路径 bug, 答题用不到写文件.
# 含图题: agent 内置 vision 路由 (huginn.vision) 会自动处理消息里的 image,
# 不需要在 tool_filter 显式列; 但需要 image_index/visualize_tool 做后处理.
HLE_TOOL_FILTER = [
    "code_tool",       # 计算、推导验证
    "bash_tool",       # pip install
    "web_search_tool", # 查常数/定义
    "file_read_tool",
    "subagent_tool",
]


def load_hle_dataset(n: int | None = None, domain: str | None = None):
    """从 HuggingFace 加载 HLE 数据集.

    Windows 上 HF cache 用 symlink 会触发 WinError 14007, 直接 HTTP 拉 parquet
    再 pandas.read_parquet 更稳. 走 hf-mirror.com 回退.
    """
    import urllib.request
    import pandas as pd

    # HLE 数据集是单 parquet 文件
    # ponytail: 假设 test split 一个 parquet, HLE 只有 test split 所以可行.
    candidates = [
        "https://huggingface.co/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet",
        "https://hf-mirror.com/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet",
    ]
    last_err = None
    for url in candidates:
        try:
            print(f"[HLE] Fetching parquet from {url}...")
            # P0-3: cache 移出 workspace. agent workspace=Path.cwd(),
            # 之前 .cache/hle_test.parquet 在 workspace 内, agent 可读 parquet
            # 获取所有 answer. 放 ~/.huginn/cache/ 隔离.
            cache = Path.home() / ".huginn" / "cache" / "hle_test.parquet"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if not cache.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "huginn-hle/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp, open(cache, "wb") as f:
                    f.write(resp.read())
            df = pd.read_parquet(cache)
            print(f"[HLE] Loaded {len(df)} rows, columns: {list(df.columns)[:8]}")
            items = df.to_dict(orient="records")
            if domain:
                items = [x for x in items if str(x.get("domain", "")).lower() == domain.lower()]
            if n:
                items = items[:n]
            return items
        except Exception as exc:
            print(f"[HLE] fetch failed: {exc}")
            last_err = exc
            continue
    raise RuntimeError(f"Failed to load HLE dataset from all endpoints: {last_err}")


def build_question_prompt(item: dict) -> str:
    """把 HLE 题目转成 agent prompt."""
    question = item.get("question", "")
    choices = item.get("choices", None)
    answer_type = item.get("answer_type", "")
    image = item.get("image", None)

    prompt = f"# Question (Domain: {item.get('domain', '?')}, Subdomain: {item.get('subdomain', '?')})\n\n{question}\n"

    if choices and isinstance(choices, list) and len(choices) > 0:
        prompt += "\n## Choices\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D, ...
            prompt += f"{letter}. {choice}\n"
        prompt += (
            "\n## Instructions\n"
            "Think step by step. Show your reasoning. Then give your final answer as:\n"
            "ANSWER: X\n"
            "where X is the letter (A/B/C/D/...) of the correct choice.\n"
            "Use code_tool for calculations. Use web_search_tool for constants.\n"
        )
    else:
        # 简答题
        prompt += (
            "\n## Instructions\n"
            "Think step by step. Show your reasoning. Then give your final answer as:\n"
            "ANSWER: <your concise answer>\n"
            "Use code_tool for calculations. Use web_search_tool for constants.\n"
        )

    if image:
        prompt += "\n(Note: This question includes an image. If you can see it, use it. If not, reason from the text.)\n"

    return prompt


def extract_answer(response: str) -> str:
    """从 agent 响应中提取 ANSWER."""
    # 匹配 "ANSWER: X" 或 "ANSWER: X." 等
    m = re.search(r'ANSWER:\s*([A-Ea-e]|[^\n]+)', response, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.').strip()
    # 回退: 找最后一个字母
    m2 = re.search(r'\b([A-E])\b\s*$', response.strip())
    if m2:
        return m2.group(1)
    return response.strip()[-200:]  # 简答题: 取最后 200 字符


def score_mc(pred: str, gold: str) -> bool:
    """多选题评分: exact match."""
    pred_clean = pred.strip().upper()[:1]  # 取首字母
    gold_clean = gold.strip().upper()[:1]
    return pred_clean == gold_clean


def score_sa(pred: str, gold: str, judge_agent) -> tuple[bool, str]:
    """简答题评分: LLM-as-judge."""
    prompt = (
        f"Does this predicted answer match the gold answer?\n\n"
        f"Predicted: {pred[:500]}\n"
        f"Gold: {gold[:500]}\n\n"
        f"Respond with JSON: {{\"match\": true/false, \"reason\": \"...\"}}"
    )
    try:
        result = judge_agent(prompt, return_example={"match": False, "reason": "str"}, max_try=2)
        if isinstance(result, dict):
            return bool(result.get("match", False)), str(result.get("reason", ""))
    except Exception:
        pass
    # 回退: 简单字符串匹配
    return pred.strip().lower() == gold.strip().lower(), "string match fallback"


async def solve_one(item: dict, timeout: int, max_tool_calls: int) -> str:
    """用 HuginnAgent 解一道题."""
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

    system_prompt = (
        "You are an expert taking Humanity's Last Exam — a test of expert-level "
        "knowledge across all scientific and academic disciplines. "
        "Think deeply, use tools for calculations and lookups, and give precise answers.\n\n"
        "Use code_tool for mathematical calculations. "
        "Use web_search_tool for constants, definitions, and facts you're unsure about. "
        "Show your reasoning, then end with ANSWER: X."
    )

    # ── 主线认知基础设施 (Task 2.3) ──────────────────────────────
    # 默认 MemoryManager() 用 ~/.huginn/memory (TRAE 沙箱外, 写入失败),
    # 显式指 memory_dir 到 cwd 内. KB/skill 已由 register_all_tools
    # 间接启用: SkillTool import 时触发 huginn.skills.presets 注册到 SkillRegistry,
    # KB 由 ContextBuilder 用 get_knowledge_base(workspace) 自动 seed.
    # hle 的 solve_one 无 workspace 参数, 用 cwd 作 workspace (与原行为一致).
    memory_dir = Path.cwd() / ".hle_memory"
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
    checkpoint_path = Path.cwd() / ".hle_checkpoint.sqlite"

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
        max_tool_calls_per_tool=20,
        auto_approve=True,
        tool_filter=HLE_TOOL_FILTER,
        workspace=str(Path.cwd()),
    )
    register_all_tools()
    agent.register_tools_from_registry()

    prompt = build_question_prompt(item)
    # HLE 无 deliverable 检查, Orchestrator 退化为单次 chat
    from huginn.bench.orchestrator import BenchmarkOrchestrator, HLE_DELIVERABLES
    orch = BenchmarkOrchestrator(
        agent=agent,
        workspace=Path.cwd(),
        deliverable_spec=HLE_DELIVERABLES,
        max_total_calls=max_tool_calls,
        timeout=timeout,
        tag="HLE",
    )
    final = await orch.run(prompt)
    return final


def main():
    parser = argparse.ArgumentParser(description="Run HuginnAgent on HLE")
    parser.add_argument("--n", type=int, default=10, help="Number of questions (default 10)")
    parser.add_argument("--domain", default=None, help="Filter by domain (e.g. biology, physics)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-question timeout (default 300s)")
    parser.add_argument("--max-tool-calls", type=int, default=20, help="Max tool calls per question")
    parser.add_argument("--judge", action="store_true", help="Score short-answer with LLM judge")
    parser.add_argument("--output", default="hle_results.json", help="Output file")
    args = parser.parse_args()

    print(f"[HLE] Loading dataset (n={args.n}, domain={args.domain or 'all'})...")
    items = load_hle_dataset(n=args.n, domain=args.domain)
    print(f"[HLE] Loaded {len(items)} questions")

    # 简答题 judge (可选)
    judge_agent = None
    if args.judge:
        from structai import LLMAgent
        from huginn.config import HuginnConfig
        cfg = HuginnConfig.from_env()
        judge_agent = LLMAgent(
            api_key=cfg.resolved_api_key or os.environ.get("JUDGE_API_KEY", ""),
            api_base=cfg.base_url or os.environ.get("JUDGE_API_BASE", ""),
            model_version=os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat"),
            system_prompt="You are grading exam answers. Determine if the predicted answer is semantically equivalent to the gold answer.",
            temperature=0,
            max_tokens=200,
            time_limit=60,
            max_try=2,
        )

    results = []
    n_correct = 0
    n_mc = 0
    n_mc_correct = 0
    n_sa = 0
    n_sa_correct = 0

    for i, item in enumerate(items):
        qid = item.get("id", f"q{i}")
        question_preview = item.get("question", "")[:100]
        print(f"\n[HLE] Q{i+1}/{len(items)}: {question_preview}...", flush=True)

        start = time.time()
        response = asyncio.run(solve_one(item, args.timeout, args.max_tool_calls))
        elapsed = round(time.time() - start)

        pred = extract_answer(response)
        gold = item.get("answer", "")
        answer_type = item.get("answer_type", "")
        is_mc = bool(item.get("choices"))

        if is_mc:
            n_mc += 1
            correct = score_mc(pred, gold)
            if correct:
                n_correct += 1
                n_mc_correct += 1
        else:
            n_sa += 1
            if args.judge and judge_agent:
                correct, reason = score_sa(pred, gold, judge_agent)
            else:
                correct = pred.strip().lower() == gold.strip().lower()
                reason = "string match"
            if correct:
                n_correct += 1
                n_sa_correct += 1

        status = "OK" if correct else "MISS"
        print(f"[HLE] {status} ({elapsed}s) pred={pred[:50]} gold={gold[:50]}", flush=True)

        results.append({
            "id": qid,
            "domain": item.get("domain", ""),
            "subdomain": item.get("subdomain", ""),
            "answer_type": "MC" if is_mc else "SA",
            "question_preview": question_preview,
            "prediction": pred[:300],
            "gold": gold[:300],
            "correct": correct,
            "elapsed": elapsed,
            "response_preview": response[:500],
        })

    # 汇总
    total = len(items)
    accuracy = n_correct / total if total > 0 else 0
    mc_acc = n_mc_correct / n_mc if n_mc > 0 else 0
    sa_acc = n_sa_correct / n_sa if n_sa > 0 else 0

    summary = {
        "total": total,
        "correct": n_correct,
        "accuracy": round(accuracy, 4),
        "mc_total": n_mc,
        "mc_correct": n_mc_correct,
        "mc_accuracy": round(mc_acc, 4),
        "sa_total": n_sa,
        "sa_correct": n_sa_correct,
        "sa_accuracy": round(sa_acc, 4),
        "domain_filter": args.domain or "all",
    }

    output = {"summary": summary, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n[HLE] === Results ===")
    print(f"Total: {total}, Correct: {n_correct}, Accuracy: {accuracy:.1%}")
    if n_mc:
        print(f"MC: {n_mc_correct}/{n_mc} = {mc_acc:.1%}")
    if n_sa:
        print(f"SA: {n_sa_correct}/{n_sa} = {sa_acc:.1%}")
    print(f"[HLE] Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
