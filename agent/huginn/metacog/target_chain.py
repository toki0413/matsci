"""TargetChain — 目标链反推 (G62).

把 checklist 的 Mode A 条目 (定量复现) 反向推导成一条
target → required_results → required_methods → required_data → verification
的需求链. 每步执行后用 update_progress 对照产出, 检测漂移.

设计原则 (ponytail):
- dataclass + 模块级函数, 不引入新组件
- LLM 失败降级到 required_results=[target], verification="目视检查"
- KB 可选 (None 时跳过先验查询, 不报错)
- 匹配用子串包含, 不上 embedding (升级路径接 RAG)

接入点:
- Step 1 生成 checklist 后, Step 1.2 调 build_target_chains
- Step 2+ 每步执行后调 update_progress, 调 detect_drift
- context 注入用 format_target_chain_text
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class TargetChain:
    """单条目标链 — 一个 Mode A checklist 条目反推的需求链."""

    target_id: str  # "target_1", "target_2", ...
    target: str  # checklist 条目的目标
    required_results: list[str]  # 中间结果
    required_methods: list[str]  # 方法
    required_data: list[str]  # 数据
    verification: str  # 验证标准
    progress: float = 0.0  # 0.0-1.0
    dependencies: list[str] = field(default_factory=list)  # 依赖的 target_id
    completed_results: set[str] = field(default_factory=set)  # 已完成的 required_results

    def is_complete(self) -> bool:
        """所有 required_results 完成 → True."""
        if not self.required_results:
            return True
        return all(r in self.completed_results for r in self.required_results)


def build_target_chains(
    checklist: list[dict],
    kb: Any | None,
    model: Any,
    task_context: str = "",
) -> list[TargetChain]:
    """对每个 Mode A 条目调 LLM 反向推导需求链.

    Mode A = 定量复现条目 (checklist 条目 mode='A').
    Mode B 条目跳过 (定性理解, 不需 target chain).
    LLM 失败时降级: required_results=[target], verification="目视检查".
    """
    chains: list[TargetChain] = []
    target_idx = 0
    for item in checklist:
        mode = str(item.get("mode", "")).strip().upper()
        if mode != "A":
            continue
        target_text = str(item.get("item", "")).strip()
        if not target_text:
            continue
        target_idx += 1
        target_id = f"target_{target_idx}"
        chain = _build_single_chain(
            target_id=target_id,
            target=target_text,
            kb=kb,
            model=model,
            task_context=task_context,
        )
        chains.append(chain)
    return chains


def _build_single_chain(
    target_id: str,
    target: str,
    kb: Any | None,
    model: Any,
    task_context: str,
) -> TargetChain:
    """对单条目调 LLM 推导. 失败降级."""
    kb_prior = _query_kb_prior(kb, target)

    sys_msg = (
        "You are a research methodology planner. Given a research target, "
        "decompose it into a chain of required results, methods, and data. "
        "Output ONLY a JSON object, no markdown."
    )
    user_parts = [f"Target: {target}"]
    if kb_prior:
        user_parts.append(f"Paper methodology prior:\n{kb_prior}")
    if task_context:
        user_parts.append(f"Task context:\n{task_context}")
    user_parts.append(
        "Output JSON with keys: required_results, required_methods, "
        "required_data, verification"
    )
    user_msg = "\n\n".join(user_parts)

    try:
        # ponytail: langchain_core 在 metacog 别处没用, 局部 import 避免模块导入时硬依赖.
        # 升级路径: 若 metacog 后续大量用 LLM, 提到模块顶.
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]
        resp = model.invoke(messages)
        text = getattr(resp, "content", str(resp))
        return _parse_llm_response(text, target, target_id)
    except Exception as e:
        _logger.warning("target_chain LLM failed for %s: %s — fallback", target_id, e)
        return _fallback_chain(target_id, target)


def _query_kb_prior(kb: Any | None, target: str) -> str:
    """查 KB 获取"论文用什么方法"的先验. 返回文本, 无则空串."""
    if kb is None:
        return ""
    try:
        results = kb.query(target, top_k=5)
    except Exception as e:
        _logger.debug("kb query failed: %s", e)
        return ""
    if not results:
        return ""
    # ponytail: 拼 content, 不做 score 过滤 (KB 已排序). 升级路径: 加 score 阈值.
    chunks: list[str] = []
    for r in results:
        if isinstance(r, dict):
            c = r.get("content")
            if c:
                chunks.append(str(c).strip())
    return "\n---\n".join(chunks)


def _fallback_chain(target_id: str, target: str) -> TargetChain:
    """LLM 失败时的降级 chain."""
    return TargetChain(
        target_id=target_id,
        target=target,
        required_results=[target],
        required_methods=[],
        required_data=[],
        verification="目视检查",
    )


def _parse_llm_response(text: str, target: str, target_id: str) -> TargetChain:
    """解析 LLM JSON 输出为 TargetChain. 解析失败降级."""
    if not text or not text.strip():
        return _fallback_chain(target_id, target)

    # 提取 JSON 块 — LLM 有时包 markdown fence, 贪婪 .* 抓最外层 {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        _logger.warning("no JSON block in LLM response for %s — fallback", target_id)
        return _fallback_chain(target_id, target)

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        _logger.warning("JSON parse failed for %s: %s — fallback", target_id, e)
        return _fallback_chain(target_id, target)

    def _as_str_list(key: str) -> list[str]:
        v = data.get(key, [])
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    required_results = _as_str_list("required_results")
    required_methods = _as_str_list("required_methods")
    required_data = _as_str_list("required_data")
    verification = str(data.get("verification", "")).strip() or "目视检查"

    # required_results 不能空 — 空则降级
    if not required_results:
        _logger.warning("empty required_results for %s — fallback", target_id)
        return _fallback_chain(target_id, target)

    return TargetChain(
        target_id=target_id,
        target=target,
        required_results=required_results,
        required_methods=required_methods,
        required_data=required_data,
        verification=verification,
    )


def update_progress(target_chain: TargetChain, step_found: str) -> float:
    """根据步骤产出 (step_found 文本) 更新完成度.

    found 匹配某 required_results → 加入 completed_results, 重算 progress.
    返回新 progress.

    ponytail: 子串双向包含匹配, 不上 embedding. 升级路径: 接 RAG 相似度.
    """
    if not target_chain.required_results:
        target_chain.progress = 1.0
        return 1.0
    if not step_found or not step_found.strip():
        return target_chain.progress
    found_lower = step_found.lower()
    for r in target_chain.required_results:
        if r in target_chain.completed_results:
            continue
        r_lower = r.lower()
        # 双向子串 — step 含 result 或 result 含 step
        if r_lower in found_lower or found_lower in r_lower:
            target_chain.completed_results.add(r)
            break
    completed = sum(
        1 for r in target_chain.required_results
        if r in target_chain.completed_results
    )
    target_chain.progress = completed / len(target_chain.required_results)
    return target_chain.progress


def detect_drift(
    evaluations: list,
    window: int = 3,
) -> tuple[bool, str]:
    """连续 window 步 on_track=False → 漂移告警.

    evaluations 兼容两种形态: dict ({'on_track': True/False/'true'/'false'}) 或
    StepEvaluation dataclass (.on_track ∈ {'true','false','unsure'}). rcb_runner
    传 _evals_history (list[StepEvaluation]), 测试 mock 传 list[dict], 两边都能吃.

    返回 (is_drift, message).

    ponytail: 只看最近 window 步, 不做趋势分析. 升级路径: 接时序异常检测.
    """
    if window <= 0 or len(evaluations) < window:
        return (False, "")
    recent = evaluations[-window:]
    if all(_is_off_track(ev) for ev in recent):
        return (
            True,
            f"连续 {window} 步偏离目标链 — 触发重定向: 检查 required_results 是否仍可达, "
            f"或重新 build_target_chains",
        )
    return (False, "")


def _is_off_track(ev: Any) -> bool:
    """统一判定单步是否算 off-track — dict / dataclass 都能吃.

    dict 形式: {'on_track': True/False} → not True 即 off-track (兼容旧 mock).
    对象形式: StepEvaluation.on_track ∈ {'true','false','unsure'} → 仅 'false' 算 off-track,
              'unsure' 不算 (不够证据, 不触发漂移告警, 避免误报).
    """
    if isinstance(ev, dict):
        return not ev.get("on_track", True)
    val = getattr(ev, "on_track", "true")
    return val == "false"


def format_target_chain_text(chains: list[TargetChain], current_step: int) -> str:
    """格式化目标链为 context 注入文本.

    含: 当前步骤、每个 target 的 progress、还差哪些 required_results、verification 标准.
    """
    if not chains:
        return ""
    lines: list[str] = [f"[TargetChain @ step {current_step}]"]
    for c in chains:
        status = "DONE" if c.is_complete() else f"{int(c.progress * 100)}%"
        lines.append(f"- {c.target_id}: {c.target} [{status}]")
        lines.append(f"  verification: {c.verification}")
        if c.required_results:
            done = [r for r in c.required_results if r in c.completed_results]
            missing = [r for r in c.required_results if r not in c.completed_results]
            if done:
                lines.append(f"  done: {done}")
            if missing:
                lines.append(f"  missing: {missing}")
        if c.required_methods:
            lines.append(f"  methods: {c.required_methods}")
        if c.required_data:
            lines.append(f"  data: {c.required_data}")
    return "\n".join(lines)


# ── 自检 ─────────────────────────────────────────────────────────
# ponytail: 非平凡逻辑留 runnable check. LLM 用 mock, 不调真实模型.


class _FakeModel:
    """测试用 mock — 模拟 langchain BaseChatModel.invoke → resp.content."""

    def __init__(self, response_text: str, raise_error: bool = False) -> None:
        self._response = response_text
        self._raise = raise_error

    def invoke(self, messages):
        if self._raise:
            raise RuntimeError("mock LLM failure")

        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self._response)


class _FakeKB:
    """测试用 mock KB — 返回固定 chunks."""

    def __init__(self, results: list[dict] | None = None, raise_error: bool = False) -> None:
        self._results = results or []
        self._raise = raise_error

    def query(self, text: str, top_k: int = 5):
        if self._raise:
            raise RuntimeError("mock KB failure")
        return self._results[:top_k]


def _selfcheck() -> None:
    # 1. TargetChain.is_complete
    c = TargetChain(
        target_id="target_1",
        target="复现 median",
        required_results=["概率分布拟合", "统计量计算"],
        required_methods=["MLE"],
        required_data=["原始观测"],
        verification="median 在 4e-4 ± 20%",
    )
    assert not c.is_complete(), "空 completed 不应 complete"
    c.completed_results.add("概率分布拟合")
    assert not c.is_complete(), "只完成 1/2 不应 complete"
    c.completed_results.add("统计量计算")
    assert c.is_complete(), "全部完成应 complete"
    empty_rr = TargetChain(
        target_id="t0", target="x",
        required_results=[], required_methods=[], required_data=[],
        verification="目视",
    )
    assert empty_rr.is_complete(), "空 required_results 应 trivially complete"

    # 2. update_progress: 匹配 → progress 增加
    c2 = TargetChain(
        target_id="target_2",
        target="复现 mean",
        required_results=["数据加载", "均值计算"],
        required_methods=[], required_data=[],
        verification="mean 在 1.0 ± 0.1",
    )
    assert c2.progress == 0.0
    p1 = update_progress(c2, "完成了数据加载步骤")
    assert p1 > 0.0, f"匹配后 progress 应增加: {p1}"
    assert "数据加载" in c2.completed_results
    # 不匹配的 step 不变
    p_mid = update_progress(c2, "做了一些无关的事")
    assert p_mid == p1, "不匹配的 step 不应改变 progress"
    # 完成所有
    p2 = update_progress(c2, "均值计算完成")
    assert p2 == 1.0
    assert c2.is_complete()
    # 空 step 不影响
    assert update_progress(c2, "") == 1.0

    # 3. detect_drift
    evals_off = [{"on_track": False}, {"on_track": False}, {"on_track": False}]
    is_drift, msg = detect_drift(evals_off, window=3)
    assert is_drift, "连续 3 步 off-track 应漂移"
    assert msg, "漂移应有 message"
    evals_mixed = [{"on_track": False}, {"on_track": True}, {"on_track": False}]
    is_drift2, msg2 = detect_drift(evals_mixed, window=3)
    assert not is_drift2, "中间有 True 不应漂移"
    assert msg2 == ""
    # 不足 window → 无漂移
    assert detect_drift([{"on_track": False}, {"on_track": False}], window=3) == (False, "")
    # 空 evals → 无漂移
    assert detect_drift([], window=3) == (False, "")

    # 3.1 detect_drift 兼容 StepEvaluation 对象 — rcb_runner 传 _evals_history
    from types import SimpleNamespace as _NS
    evals_obj_off = [_NS(on_track="false"), _NS(on_track="false"), _NS(on_track="false")]
    is_drift3, msg3 = detect_drift(evals_obj_off, window=3)
    assert is_drift3, "对象形式连续 3 步 false 应漂移"
    assert "3 步" in msg3
    # 'unsure' 不算 off-track — 避免证据不足也触发漂移告警
    evals_unsure = [_NS(on_track="false"), _NS(on_track="unsure"), _NS(on_track="false")]
    is_drift4, _ = detect_drift(evals_unsure, window=3)
    assert not is_drift4, "unsure 不应算 off-track"
    # 混合 dict + 对象也能吃
    evals_mix = [{"on_track": False}, _NS(on_track="false"), _NS(on_track="false")]
    is_drift5, _ = detect_drift(evals_mix, window=3)
    assert is_drift5, "dict + 对象混合应正常判定"

    # 4. _parse_llm_response: 合法 JSON → TargetChain; 非法 → 降级
    good_json = json.dumps({
        "required_results": ["拟合", "计算"],
        "required_methods": ["MLE"],
        "required_data": ["原始数据"],
        "verification": "median 在 4e-4 ± 20%",
    })
    tc = _parse_llm_response(good_json, target="复现 median", target_id="t1")
    assert tc.target == "复现 median"
    assert tc.target_id == "t1"
    assert tc.required_results == ["拟合", "计算"]
    assert tc.required_methods == ["MLE"]
    assert tc.required_data == ["原始数据"]
    assert tc.verification == "median 在 4e-4 ± 20%"

    # markdown fence 包裹的 JSON
    fenced = f"```json\n{good_json}\n```"
    tc2 = _parse_llm_response(fenced, target="x", target_id="t2")
    assert tc2.required_results == ["拟合", "计算"], "markdown fence 应被正则剥掉"

    # 非法 JSON → 降级
    tc3 = _parse_llm_response("not a json at all", target="复现 median", target_id="t3")
    assert tc3.required_results == ["复现 median"], \
        f"非法 JSON 应降级 required_results=[target], got {tc3.required_results}"
    assert tc3.verification == "目视检查"

    # 空 required_results → 降级
    empty_resp = json.dumps({
        "required_results": [], "required_methods": [],
        "required_data": [], "verification": "",
    })
    tc4 = _parse_llm_response(empty_resp, target="x", target_id="t4")
    assert tc4.required_results == ["x"], "空 required_results 应降级"
    assert tc4.verification == "目视检查"

    # 空文本 → 降级
    tc5 = _parse_llm_response("", target="y", target_id="t5")
    assert tc5.required_results == ["y"]

    # 5. format_target_chain_text: 含 target / progress / verification
    chains = [
        TargetChain(
            target_id="target_1",
            target="复现 median",
            required_results=["拟合", "计算"],
            required_methods=["MLE"],
            required_data=["原始数据"],
            verification="median 在 4e-4 ± 20%",
            progress=0.5,
            completed_results={"拟合"},
        ),
    ]
    text = format_target_chain_text(chains, current_step=5)
    assert "step 5" in text, "应含当前步骤"
    assert "复现 median" in text, "应含 target"
    assert "median 在 4e-4" in text, "应含 verification"
    assert "50%" in text, "应含 progress"
    assert "拟合" in text, "应含已完成的 required_result"
    assert "计算" in text, "应含缺失的 required_result"
    assert "MLE" in text, "应含方法"
    # 空 chains → 空串
    assert format_target_chain_text([], current_step=1) == ""
    # 完成的 chain 显示 DONE
    done_chain = TargetChain(
        target_id="target_2", target="x",
        required_results=["a"], required_methods=[], required_data=[],
        verification="v",
        completed_results={"a"},
    )
    done_chain.progress = 1.0
    assert "DONE" in format_target_chain_text([done_chain], current_step=10)

    # 6. build_target_chains: Mode A 推 chain, Mode B 跳过, LLM 失败降级
    checklist = [
        {"item": "复现 median 4e-4", "mode": "A", "metric": "median"},
        {"item": "理解论文方法", "mode": "B", "metric": ""},
        {"item": "复现 mean 1.0", "mode": "A", "metric": "mean"},
        {"item": "", "mode": "A", "metric": "x"},  # 空 item 跳过
    ]
    fake_model = _FakeModel(response_text=good_json)
    built = build_target_chains(checklist, kb=None, model=fake_model, task_context="背景")
    # Mode B 跳过 + 空 item 跳过 → 2 条
    assert len(built) == 2, f"应建 2 条 chain, got {len(built)}"
    assert built[0].target_id == "target_1"
    assert built[1].target_id == "target_2"
    assert built[0].target == "复现 median 4e-4"
    assert built[0].required_results == ["拟合", "计算"]

    # LLM 抛错 → 降级
    fail_model = _FakeModel(response_text="", raise_error=True)
    built_fail = build_target_chains(checklist, kb=None, model=fail_model)
    assert len(built_fail) == 2
    for c in built_fail:
        assert c.required_results == [c.target], "LLM 失败应降级"
        assert c.verification == "目视检查"

    # 7. KB prior: None / 抛错 / 空结果 / 多 chunk 都不崩
    fake_kb = _FakeKB(results=[{"content": "论文用 MLE 拟合"}])
    assert _query_kb_prior(fake_kb, "median") == "论文用 MLE 拟合"
    assert _query_kb_prior(None, "median") == ""
    assert _query_kb_prior(_FakeKB(raise_error=True), "median") == ""
    assert _query_kb_prior(_FakeKB(results=[]), "median") == ""
    multi_prior = _query_kb_prior(
        _FakeKB(results=[{"content": "chunk1"}, {"content": "chunk2"}]), "x",
    )
    assert "chunk1" in multi_prior and "chunk2" in multi_prior
    assert "---" in multi_prior

    # 8. KB + task_context 注入 build_target_chains 不崩, chain 正常出
    built_with_kb = build_target_chains(
        [{"item": "复现 median", "mode": "A", "metric": "median"}],
        kb=fake_kb, model=fake_model, task_context="任务背景",
    )
    assert len(built_with_kb) == 1
    assert built_with_kb[0].required_results == ["拟合", "计算"]

    print("target_chain selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
