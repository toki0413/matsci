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
    strict: bool = False           # B1: strict 模式 (0/1 判定, 不算加权分)
    strict_passed: bool | None = None  # B1: strict 模式下 [judge] 字段解析结果

    @property
    def score(self) -> float:
        """加权总分. code_quality 仅当 >0 时计入. strict 模式返回 0/1."""
        if self.strict:
            return 1.0 if self.strict_passed else 0.0
        weights = {"correctness": 0.5, "methodology": 0.3, "completeness": 0.2}
        if self.code_quality > 0:
            weights = {"correctness": 0.4, "methodology": 0.2, "completeness": 0.2, "code_quality": 0.2}
        total = sum(getattr(self, k) * v for k, v in weights.items())
        return round(total, 3)

    @property
    def passed(self) -> bool:
        # B1: strict 模式只看 [judge] 字段, 不算加权总分
        if self.strict:
            return bool(self.strict_passed)
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


# B1: strict 模式 — 8 条铁律, 0/1 判定. 借鉴 PerceptionBench 严格判定风格.
JUDGE_STRICT_SYSTEM = """你是材料科学/物理/计算任务的严格评审员. 按 8 条铁律做 0/1 判定 (True=完全正确, False=有任何瑕疵).

8 条铁律:
1. 结果数值必须在参考答案容差内 (默认相对容差 5%), 否则 False
2. 物理单位必须正确且完整 (eV/Å/GPa/K 等), 缺单位或单位错 → False
3. 方法/公式必须正确, 用错公式或推导错误 → False
4. 答非所问或漏答子问题 → False
5. 编造数据、文献、参数 (无出处) → False
6. 逻辑跳跃, 缺关键推导步骤 → False
7. 代码题: 代码不可运行或核心逻辑错 → False
8. 只有完全满足 1-7 才判 True, 任何一条违反 → False

输出格式 (严格遵循, 不要其他内容):
[reason]简要说明违反了哪条铁律或全部满足[judge]True 或 [judge]False"""


def _parse_strict_response(text: str) -> tuple[str, bool]:
    """B1: 解析 strict 模式响应 [reason]...[judge]True/False.

    返回 (reason, passed). 解析失败时 passed=False.
    ponytail: regex 提取 [judge] 字段, 不上 JSON parser.
    """
    import re
    reason = ""
    judge = False
    reason_m = re.search(r"\[reason\](.*?)(?:\[judge\]|$)", text, re.DOTALL)
    if reason_m:
        reason = reason_m.group(1).strip()[:200]
    judge_m = re.search(r"\[judge\]\s*(True|False)", text, re.IGNORECASE)
    if judge_m:
        judge = judge_m.group(1).lower() == "true"
    else:
        # 解析失败, 默认 False (保守)
        reason = reason or "strict judge 响应解析失败"
    return reason, judge


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
    strict: bool = False,
) -> JudgeRubric:
    """调 LLM judge 评估单题.

    api_key 默认读 DEEPSEEK_API_KEY 环境变量.
    返回 JudgeRubric, 含 4 维度分数和总分.

    B1: strict=True 时用 8 条铁律 prompt + 0/1 判定, 不算加权总分.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("HUGINN_API_KEY")
    if not key:
        return JudgeRubric(reason="无 API key, 跳过 LLM judge", strict=strict)

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=500,
        )
        prompt = _build_judge_prompt(task_prompt, agent_output, reference, is_code_task)
        system_msg = JUDGE_STRICT_SYSTEM if strict else JUDGE_SYSTEM
        resp = llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = str(text)
        if strict:
            reason, passed = _parse_strict_response(text)
            return JudgeRubric(
                reason=reason, strict=True, strict_passed=passed,
                correctness=1.0 if passed else 0.0,
                methodology=1.0 if passed else 0.0,
                completeness=1.0 if passed else 0.0,
            )
        return _parse_judge_response(text)
    except Exception as e:
        return JudgeRubric(reason=f"judge 调用失败: {e}", strict=strict)


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


# ── self-check ─────────────────────────────────────────────────


def _selfcheck() -> None:
    """B1 selfcheck: strict 模式响应解析 + JudgeRubric 0/1 判定.

    ponytail: mock judge 响应文本验解析逻辑, 不调真 LLM.
    ceiling: 没验真 LLM 返回的多样性 (edge case 见 acceptance test).
    """
    import os

    # 1. _parse_strict_response: [judge]True → passed=True
    text_true = "[reason]全部满足铁律1-7, 数值单位均正确[judge]True"
    reason_t, passed_t = _parse_strict_response(text_true)
    assert passed_t is True, f"[judge]True 应解析为 True, got {passed_t}"
    assert "全部满足" in reason_t, f"reason 提取错误: {reason_t}"
    print(f"1. parse [judge]True → passed={passed_t}, reason='{reason_t}' OK")

    # 2. _parse_strict_response: [judge]False → passed=False
    text_false = "[reason]违反铁律2: 缺单位 eV[judge]False"
    reason_f, passed_f = _parse_strict_response(text_false)
    assert passed_f is False, f"[judge]False 应解析为 False, got {passed_f}"
    assert "违反铁律2" in reason_f, f"reason 提取错误: {reason_f}"
    print(f"2. parse [judge]False → passed={passed_f}, reason='{reason_f}' OK")

    # 3. 解析失败 (无 [judge] 字段) → passed=False (保守)
    text_bad = "这个回答不完全正确"
    reason_b, passed_b = _parse_strict_response(text_bad)
    assert passed_b is False, f"无 [judge] 字段应默认 False, got {passed_b}"
    assert "解析失败" in reason_b, f"应提示解析失败: {reason_b}"
    print(f"3. parse no [judge] → passed={passed_b} (保守) OK")

    # 4. JudgeRubric strict=True + strict_passed=True → passed=True, score=1.0
    r_true = JudgeRubric(strict=True, strict_passed=True, reason="全部满足")
    assert r_true.passed is True, "strict_passed=True → passed 应 True"
    assert r_true.score == 1.0, f"strict True score 应 1.0, got {r_true.score}"
    print(f"4. JudgeRubric strict=True, passed=True → score={r_true.score} OK")

    # 5. JudgeRubric strict=True + strict_passed=False → passed=False, score=0.0
    r_false = JudgeRubric(strict=True, strict_passed=False, reason="违反铁律")
    assert r_false.passed is False, "strict_passed=False → passed 应 False"
    assert r_false.score == 0.0, f"strict False score 应 0.0, got {r_false.score}"
    print(f"5. JudgeRubric strict=True, passed=False → score={r_false.score} OK")

    # 6. 非 strict 模式不受 strict_passed 影响 (向后兼容)
    r_normal = JudgeRubric(correctness=0.9, methodology=0.8, completeness=0.7)
    assert r_normal.passed is True, f"非 strict 加权 0.83 应 PASS, got {r_normal.score}"
    assert r_normal.strict is False
    print(f"6. 非 strict 模式加权 score={r_normal.score} → passed={r_normal.passed} OK")

    # 7. judge_task 无 API key 时返回 strict 标记
    orig_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    orig_key2 = os.environ.pop("HUGINN_API_KEY", None)
    try:
        rubric = judge_task("test", "answer", strict=True)
        assert rubric.strict is True, "无 key 时 strict 标记应保留"
        assert "无 API key" in rubric.reason, f"reason 应提示无 key: {rubric.reason}"
        assert rubric.passed is False, "无 key 时 passed 应 False"
        print(f"7. judge_task no key + strict=True → strict={rubric.strict}, passed={rubric.passed} OK")
    finally:
        if orig_key:
            os.environ["DEEPSEEK_API_KEY"] = orig_key
        if orig_key2:
            os.environ["HUGINN_API_KEY"] = orig_key2

    print("llm_judge B1 selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
