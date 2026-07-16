"""LLM Judge — 对 benchmark 结果做二次评审.

对标 PaperBench SimpleJudge (F1=0.83 vs 人类专家).
当 regex evaluator 返回部分分 (score < 1.0) 或 FAIL 时, 触发 LLM judge.

judge 通过 langchain_openai 调 DeepSeek API (兼容 OpenAI 接口),
judge prompt 让模型扮演严格科研评审员, 按 rubric 打分.

rubric 树结构 (借鉴 PaperBench):
  - correctness (0-1): 结果数值是否正确
  - methodology (0-1): 方法/推导是否合理
  - completeness (0-1): 是否完整回答了问题
  - code_quality (0-1): 代码是否可运行 (代码题适用)
总分 = 加权平均 (correctness 0.5, methodology 0.3, completeness 0.2, code_quality 0.0/0.3)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from .task import TaskResult


@dataclass
class JudgeRubric:
    """单个 rubric 维度."""
    correctness: float = 0.0       # 结果数值正确性
    methodology: float = 0.0       # 方法/推导合理性
    completeness: float = 0.0      # 回答完整性
    code_quality: float = 0.0      # 代码质量 (代码题)
    reason: str = ""

    @property
    def score(self) -> float:
        """加权总分. code_quality 仅当 >0 时计入."""
        weights = {"correctness": 0.5, "methodology": 0.3, "completeness": 0.2}
        if self.code_quality > 0:
            weights = {"correctness": 0.4, "methodology": 0.2, "completeness": 0.2, "code_quality": 0.2}
        total = sum(getattr(self, k) * v for k, v in weights.items())
        return round(total, 3)

    @property
    def passed(self) -> bool:
        return self.score >= 0.6


JUDGE_SYSTEM = """你是一个严格的科研评审员, 负责评估 AI agent 在材料科学/物理/计算任务中的回答质量.

评审规则:
1. 按 4 个维度打分 (0.0-1.0):
   - correctness: 结果数值是否正确 (与参考答案比对, 容差内算 1.0)
   - methodology: 方法/推导是否合理 (公式正确、步骤清晰)
   - completeness: 是否完整回答了问题 (所有子问题都答了)
   - code_quality: 代码题适用 (代码结构、可运行性)
2. 总分 >= 0.6 算 PASS, 否则 FAIL
3. 只输出 JSON, 不要其他内容

输出格式 (严格 JSON):
{"correctness": 0.0-1.0, "methodology": 0.0-1.0, "completeness": 0.0-1.0, "code_quality": 0.0-1.0, "reason": "简要说明"}"""


def _build_judge_prompt(
    task_prompt: str,
    agent_output: str,
    reference: str | None = None,
    is_code_task: bool = False,
) -> str:
    """构造 judge prompt."""
    ref_section = f"\n参考答案: {reference}" if reference else "\n参考答案: (无, 凭你的知识判断)"
    code_hint = "\n注意: 这是代码题, code_quality 维度需评估." if is_code_task else ""
    return f"""请评估以下 AI agent 的回答.

任务题目:
{task_prompt}
{ref_section}{code_hint}

Agent 回答:
{agent_output[:3000]}

请按 rubric 打分, 只输出 JSON."""


def _parse_judge_response(text: str) -> JudgeRubric:
    """从 LLM 响应里提取 JSON, 解析成 JudgeRubric."""
    # 找 JSON 块
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return JudgeRubric(reason="judge 响应解析失败")
    try:
        data = json.loads(m.group(0))
        return JudgeRubric(
            correctness=float(data.get("correctness", 0)),
            methodology=float(data.get("methodology", 0)),
            completeness=float(data.get("completeness", 0)),
            code_quality=float(data.get("code_quality", 0)),
            reason=str(data.get("reason", ""))[:200],
        )
    except (json.JSONDecodeError, ValueError) as e:
        return JudgeRubric(reason=f"judge JSON 解析失败: {e}")


def judge_task(
    task_prompt: str,
    agent_output: str,
    reference: str | None = None,
    is_code_task: bool = False,
    api_key: str | None = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/v1",
) -> JudgeRubric:
    """调 LLM judge 评估单题.

    api_key 默认读 DEEPSEEK_API_KEY 环境变量.
    返回 JudgeRubric, 含 4 维度分数和总分.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("HUGINN_API_KEY")
    if not key:
        return JudgeRubric(reason="无 API key, 跳过 LLM judge")

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=500,
        )
        prompt = _build_judge_prompt(task_prompt, agent_output, reference, is_code_task)
        resp = llm.invoke([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return _parse_judge_response(str(text))
    except Exception as e:
        return JudgeRubric(reason=f"judge 调用失败: {e}")


def judge_with_regex_fallback(
    task_prompt: str,
    agent_output: str,
    regex_result: TaskResult,
    reference: str | None = None,
    is_code_task: bool = False,
    threshold: float = 0.8,
) -> TaskResult:
    """先看 regex evaluator 结果, 高分直接采纳; 低分触发 LLM judge.

    threshold: regex score >= threshold 时直接采纳, 不调 judge.
    """
    # regex 高分直接采纳 (score 为 None 时按 passed=True 的 1.0 处理)
    regex_score = regex_result.score if regex_result.score is not None else (1.0 if regex_result.passed else 0.0)
    if regex_result.passed and regex_score >= threshold:
        return regex_result

    # 低分触发 LLM judge
    rubric = judge_task(task_prompt, agent_output, reference, is_code_task)
    if rubric.reason.startswith(("无 API key", "judge 调用失败", "judge 响应解析失败")):
        # judge 失败, 回退到 regex 结果
        return regex_result

    return TaskResult(
        task_id=regex_result.task_id,
        category=regex_result.category,
        passed=rubric.passed,
        reason=f"[LLM judge] {rubric.reason} (score={rubric.score:.2f})",
        output=regex_result.output,
        score=rubric.score,
        exec_time_seconds=regex_result.exec_time_seconds,
    )
