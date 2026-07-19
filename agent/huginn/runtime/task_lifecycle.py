"""G60 + G71: 任务生命周期管理。

状态机: created → running → paused(两个子状态) → resumed → completed/failed
paused 子状态借鉴 Temporal signal-based pause/resume 思路，可持久化数天/数周。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED_WAITING_APPROVAL = "paused_waiting_approval"  # G71: 被动等待审批
    PAUSED_ASKING_DECISION = "paused_asking_decision"    # G71: 主动请求决策
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"


# 合法状态转换表。终态出口为空集。
ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.CREATED: {TaskState.RUNNING, TaskState.FAILED},
    TaskState.RUNNING: {
        TaskState.PAUSED_WAITING_APPROVAL,
        TaskState.PAUSED_ASKING_DECISION,
        TaskState.COMPLETED,
        TaskState.FAILED,
    },
    TaskState.PAUSED_WAITING_APPROVAL: {TaskState.RESUMED, TaskState.FAILED},
    TaskState.PAUSED_ASKING_DECISION: {TaskState.RESUMED, TaskState.FAILED},
    TaskState.RESUMED: {
        TaskState.RUNNING,
        TaskState.PAUSED_WAITING_APPROVAL,
        TaskState.PAUSED_ASKING_DECISION,
        TaskState.COMPLETED,
        TaskState.FAILED,
    },
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
}

_PAUSED_STATES = frozenset({TaskState.PAUSED_WAITING_APPROVAL, TaskState.PAUSED_ASKING_DECISION})
_TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED})


@dataclass
class DecisionRequest:
    """G71: agent 主动请求用户决策的上下文。"""

    step_id: int
    question: str
    options: list[dict]  # [{"id": "A", "label": "...", "pros": "...", "cons": "..."}]
    context_summary: str
    created_at: float = field(default_factory=time.time)
    answered_at: float | None = None
    answer: str | None = None  # 用户选择的 option id


@dataclass
class TaskLifecycle:
    task_id: str
    state: TaskState = TaskState.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    paused_at: float | None = None
    pause_reason: str | None = None
    decision_request: DecisionRequest | None = None  # G71: 当前挂起的决策请求

    def transition(self, new_state: TaskState, reason: str = "") -> None:
        """状态转换，非法转换 raise ValueError。"""
        allowed = ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"非法状态转换: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state
        self.updated_at = time.time()
        # 进入暂停态时记一下时间和原因，方便审计/持久化
        if new_state in _PAUSED_STATES:
            self.paused_at = time.time()
            if reason:
                self.pause_reason = reason

    def pause_for_approval(self, reason: str = "") -> None:
        """被动暂停，等待用户审批。"""
        self.transition(TaskState.PAUSED_WAITING_APPROVAL, reason)

    def pause_for_decision(self, decision: DecisionRequest) -> None:
        """G71: 主动暂停，请求用户决策。"""
        self.transition(TaskState.PAUSED_ASKING_DECISION, decision.question)
        self.decision_request = decision

    def resume(self, answer: str | None = None) -> None:
        """恢复任务。如有 decision_request，answer 是用户选择的 option id。"""
        # 先把答案落到 decision_request 上（调用方若持有引用可读到），再清掉 lifecycle 的挂起引用
        if self.decision_request is not None and answer is not None:
            self.decision_request.answer = answer
            self.decision_request.answered_at = time.time()
        self.transition(TaskState.RESUMED)
        self.decision_request = None

    def complete(self) -> None:
        """任务完成。"""
        self.transition(TaskState.COMPLETED)

    def fail(self, reason: str = "") -> None:
        """任务失败。"""
        # ponytail: 不新增 fail_reason 字段，reason 仅用于 transition 内部审计，外部需要请走事件总线
        self.transition(TaskState.FAILED, reason)

    def to_dict(self) -> dict:
        """序列化。"""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "paused_at": self.paused_at,
            "pause_reason": self.pause_reason,
            "decision_request": asdict(self.decision_request) if self.decision_request else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskLifecycle":
        """反序列化。"""
        dr_data = data.get("decision_request")
        dr = DecisionRequest(**dr_data) if dr_data else None
        return cls(
            task_id=data["task_id"],
            state=TaskState(data["state"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            paused_at=data.get("paused_at"),
            pause_reason=data.get("pause_reason"),
            decision_request=dr,
        )


def save_task_lifecycle(lifecycle: TaskLifecycle, workspace: Path) -> Path:
    """落盘到 .huginn/task_lifecycle.json，原子写（tmp + rename），返回路径。"""
    ws = Path(workspace).resolve()
    huginn_dir = ws / ".huginn"
    huginn_dir.mkdir(parents=True, exist_ok=True)
    target = huginn_dir / "task_lifecycle.json"
    tmp = huginn_dir / "task_lifecycle.json.tmp"
    tmp.write_text(
        json.dumps(lifecycle.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(target)  # 原子替换
    return target


def load_task_lifecycle(task_id: str, workspace: Path) -> TaskLifecycle | None:
    """从 .huginn/task_lifecycle.json 加载。文件不存在或 task_id 不匹配返回 None。"""
    ws = Path(workspace).resolve()
    target = ws / ".huginn" / "task_lifecycle.json"
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("task_id") != task_id:
        return None
    return TaskLifecycle.from_dict(data)


def is_terminal(state: TaskState) -> bool:
    """是否终态。"""
    return state in _TERMINAL_STATES


def is_paused(state: TaskState) -> bool:
    """是否暂停态（含两个子状态）。"""
    return state in _PAUSED_STATES


# === G71: 人机协同 pause 触发判定 ===


def _tc_attr(tc: Any, name: str, default: Any = None) -> Any:
    """TargetChain 兼容取值: dataclass / dict 都能取."""
    if isinstance(tc, dict):
        return tc.get(name, default)
    return getattr(tc, name, default)


def _tc_is_complete(tc: Any) -> bool:
    """required_results 是否全完成. duck typing 兼容 dataclass / dict."""
    required = _tc_attr(tc, "required_results", []) or []
    if not required:
        return False
    completed = _tc_attr(tc, "completed_results", set()) or set()
    return all(r in completed for r in required)


def _intention_attr(it: Any, name: str, default: Any = None) -> Any:
    """ProspectiveIntention 兼容取值: dataclass / dict."""
    if isinstance(it, dict):
        return it.get(name, default)
    return getattr(it, name, default)


def _option(idx: str, label: str, pros: str, cons: str) -> dict:
    """构造标准 option dict."""
    return {"id": idx, "label": label, "pros": pros, "cons": cons}


def _options_from_intention(desc: str) -> list[dict]:
    """从 intention description 提取选项.

    ponytail: 默认三选一 (执行/跳过/修改), 不解析 description 语义.
    升级路径: 按 intention kind 分类生成选项.
    """
    short = desc[:50]
    return [
        _option("A", f"按意图执行: {short}", "遵循原计划", "可能不合当前情境"),
        _option("B", "跳过该意图", "节省时间", "可能遗漏关键决策"),
        _option("C", "修改计划后再执行", "更贴合当前情境", "需要重新规划"),
    ]


# === P0-A: PMK 一致性检查 (Čech H¹ proxy) ===
# 高阶网络视角: persona/memory/knowledge 三路立场是三个局部模型, sheaf gluing
# 失败 = H¹ ≠ 0 = 局部模型无法粘合成全局一致策略. 这里用启发式判定, 不调 LLM.
#
# ponytail: 规则版只看"显式对立词" (反对/不行/错/不要 vs 推荐/应该/用X),
# 抓显式冲突. 升级路径: LLM 判定三路立场语义一致性, 抓隐式冲突.
# 天花板: 规则版抓不到同义改写 ("用 A" vs "A 是唯一选择" 算一致但规则可能误判),
# 但误判代价低 (多 pause 一次让用户选, 不影响正确性).


# 显式推荐词 — 出现表示该路"主张 X"
_RECOMMEND_WORDS = frozenset({
    "推荐", "应该", "建议", "用", "采用", "选择", "倾向", "偏好", "走",
    "recommend", "should", "suggest", "use", "prefer",
})
# 显式反对词 — 出现表示该路"反对 X"
_OPPOSE_WORDS = frozenset({
    "反对", "不行", "不要", "别用", "避免", "错", "不行", "不可", "拒绝",
    "oppose", "avoid", "don't", "wrong", "refuse",
})


def _extract_pmk_stance(text: str) -> tuple[str, str]:
    """从单路立场文本提取 (stance, subject).

    stance ∈ {"recommend", "oppose", "neutral"}: 该路主张或反对什么.
    subject: 立场指向的对象 (简单取首个推荐/反对词后的名词短语, 截断 30 字).

    缺失文本 / 无推荐反对词 → ("neutral", "").
    ponytail: 不做完整 NLP, 只抓显式信号. 升级路径: LLM 抽立场.
    """
    if not text or not text.strip():
        return ("neutral", "")
    text_lower = text.lower()
    has_recommend = any(w in text_lower for w in _RECOMMEND_WORDS)
    has_oppose = any(w in text_lower for w in _OPPOSE_WORDS)
    if has_oppose:
        stance = "oppose"
    elif has_recommend:
        stance = "recommend"
    else:
        return ("neutral", "")
    # 抓立场对象: 推荐/反对词后 30 字内的中文/英文短语
    import re as _re
    for w in list(_OPPOSE_WORDS) + list(_RECOMMEND_WORDS):
        idx = text_lower.find(w)
        if idx >= 0:
            tail = text[idx + len(w):].strip()[:30]
            # 去标点
            tail = _re.sub(r"[，。、,.;:!?()\s]+", " ", tail).strip()
            if tail:
                return (stance, tail)
    return (stance, "")


def _pmk_subject_tokens(text: str) -> set[str]:
    """从 subject 文本抽关键 token — 用于相似度判定.

    抽取规则: 按非字母数字分割, 保留:
    - 长度 >= 2 的 token (中英文都行)
    - 单个大写字母 (X/Y/Z/A/B 等数学变量风格占位符)
    过滤掉常见动词 (用/采用/选择/历史/失败/过 等修饰词) + 英文冠词.

    ponytail: 不上 embedding, 用 token 重合度. 升级路径: embedding cosine.
    天花板: 同义不同形 ("GBR" vs "gradient boosting") 抓不到, 但显式冲突
    一般用同一名词, 实际够用.
    """
    if not text:
        return set()
    import re as _re
    _STOP = {"用", "采用", "选择", "走", "倾向", "偏好", "历史", "上", "失败",
             "过", "的", "是", "在", "方法", "该", "此", "this", "the", "method",
             "a", "an", "is", "to", "for", "of"}
    tokens: set[str] = set()
    for m in _re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fa5]{2,}", text):
        tok = m.group(0).lower()
        if tok in _STOP:
            continue
        # 长度 1 的 token 只保留原始大写形式 (X/Y/Z 风格), 小写单字母过滤
        if len(tok) == 1:
            if m.group(0).isupper():
                tokens.add(tok)  # lower 化后存, 比较时也 lower
            continue
        tokens.add(tok)
    return tokens


def _pmk_subjects_similar(sub1: str, sub2: str) -> bool:
    """两个 subject 是否相似 — token 重合度 >= 1 即相似.

    ponytail: 阈值 1 (任一共同 token 即相似). 保守低阈值, 宁可多 pause 让
    用户选, 不漏掉真实冲突. 升级路径: Jaccard >= 0.5 或 embedding cosine.
    """
    t1 = _pmk_subject_tokens(sub1)
    t2 = _pmk_subject_tokens(sub2)
    if not t1 or not t2:
        return False
    return bool(t1 & t2)


def _check_pmk_consistency(pmk_state: dict) -> tuple[bool, str]:
    """PMK 三路立场一致性检查 — Čech H¹ proxy.

    pmk_state: {"persona": str, "memory": str, "kb": str, "deviation": str}
    返回 (is_inconsistent, reason). 不一致 = 至少两路对同一 subject 显式对立.

    判定规则 (规则版, 不调 LLM):
    1. 提取三路 stance + subject
    2. 若两路 stance 为 oppose 且 subject 相似 (token 重合) → 冲突
    3. 若一路 recommend X, 另一路 oppose X 且 subject 相似 → 冲突
    4. neutral 不参与冲突
    5. 三路全 neutral → 一致

    ponytail: subject 相似用 token 重合度, 不上 embedding. 升级路径: embedding cosine.
    """
    if not pmk_state:
        return (False, "")

    stances = {
        source: _extract_pmk_stance(text)
        for source, text in pmk_state.items()
        if source in ("persona", "memory", "kb")
    }
    # 过滤掉 neutral
    active = {src: (st, sub) for src, (st, sub) in stances.items() if st != "neutral"}
    if len(active) < 2:
        return (False, "")  # 少于两路有立场, 不可能冲突

    # 两两检查: 同 subject 对立 → 冲突
    sources = list(active.keys())
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, (st1, sub1) = sources[i], active[sources[i]]
            s2, (st2, sub2) = sources[j], active[sources[j]]
            if not sub1 or not sub2:
                continue
            if not _pmk_subjects_similar(sub1, sub2):
                continue
            # 显式对立: 一个 recommend 一个 oppose, 或两个都 oppose
            if (st1 == "oppose" and st2 == "recommend") or \
               (st1 == "recommend" and st2 == "oppose") or \
               (st1 == "oppose" and st2 == "oppose"):
                return (
                    True,
                    f"{s1}({st1} '{sub1}') vs {s2}({st2} '{sub2}') — "
                    f"对同一 subject 立场冲突"
                )

    return (False, "")


def should_pause_for_decision(
    step_evaluations: list,
    target_chains: list,
    kb_recall_empty: bool = False,
    fired_intentions: list | None = None,
    pmk_state: dict | None = None,
    grill_state: dict | None = None,
) -> tuple[bool, str, list[dict]]:
    """G71: 检查是否该暂停请求用户决策.

    触发条件 (按优先级):
    0. grill 触发 (grill-me 风格开工前澄清): 歧义高 / tier=A/B + scene 歧义 /
       plan 为空 → 进入 grill 模式, 一次一问, 摊开假设
    1. 连续 3 步 on_track=="false" → pause, options=[换方法, 补数据, 重定向目标链]
    2. TargetChain 验证失败 (required_results 全完成但结构检查 failed) → pause
    3. 连续 3 步 evidence_quality=="low" 且 kb_recall_empty → pause
    4. fired intentions description 含 "用户决策" / "user decision" → pause
    5. PMK 一致性障碍 (Čech H¹ proxy, G71+P0-A): persona/memory/knowledge 三路
       立场冲突且无法粘合成全局一致策略 → pause

    grill_state 格式: {"has_grilled": bool, "ambiguity_score": float,
    "tier": str, "scene_tag": str, "plan_is_empty": bool}.
    详见 pre_plan_grill.should_start_grill. Engine 在 plan_check 之前
    调本函数检查条件 0; 触发后 LLM system prompt 切到 GRILL_SYSTEM_PROMPT,
    LLM 自己负责 "一次一题", 用户答完累计后说 "shared understanding reached"
    退出 grill 模式.

    pmk_state 格式: {"persona": str, "memory": str, "kb": str, "deviation": str}
    三路立场文本. 缺失字段视为 "无立场" (不参与冲突判定).

    返回 (should_pause, reason, options). 每个 option 是
    {"id": "A", "label": "...", "pros": "...", "cons": "..."} 格式.
    不触发 → (False, "", []).

    ponytail: 规则启发式, 不调 LLM 判断该不该 pause. 升级路径: LLM 评估决策价值,
    避免过度暂停. fired_intentions 当前只做文本匹配, 升级路径是 intention 分类.
    """
    # 条件 0: grill-me 风格开工前澄清 — 优先级最高, 先于其他检查
    # 详见 pre_plan_grill.should_start_grill. 触发后 Engine 切 LLM system prompt.
    if grill_state:
        from huginn.runtime.pre_plan_grill import (
            grill_pause_options, should_start_grill,
        )
        should_grill, grill_reason = should_start_grill(**grill_state)
        if should_grill:
            return (
                True,
                f"GRILL 模式建议启动: {grill_reason}",
                grill_pause_options(),
            )

    # 条件 5: PMK 一致性障碍 — 三路立场冲突 → 局部模型无法粘合成全局
    # 高阶网络 H¹ proxy: sheaf gluing 失败的启发式判定, 非严格 Čech cohomology.
    if pmk_state:
        inconsistent, pmk_reason = _check_pmk_consistency(pmk_state)
        if inconsistent:
            opts = [
                _option("A", "遵从 Persona", "保持方法品味一致性", "可能忽视经验/知识"),
                _option("B", "遵从 Memory", "复用历史成功经验", "可能不适合当前情境"),
                _option("C", "遵从 Knowledge", "用领域规则接地", "可能过于保守"),
            ]
            return (True, f"PMK 一致性障碍: {pmk_reason}", opts)

    # 条件 4 不依赖 step_evaluations, 先单独检查 fired intentions
    if fired_intentions:
        for it in fired_intentions:
            desc = str(_intention_attr(it, "description", "") or "")
            if "用户决策" in desc or "user decision" in desc.lower():
                return (
                    True,
                    f"Prospective Memory 触发需用户决策: {desc}",
                    _options_from_intention(desc),
                )

    if not step_evaluations:
        return (False, "", [])

    recent = step_evaluations[-3:]

    # 条件 1: 连续 3 步 on_track=="false" → 重定向
    if len(recent) >= 3 and all(e.on_track == "false" for e in recent):
        opts = [
            _option("A", "换方法", "可能突破当前瓶颈", "前期工作可能浪费"),
            _option("B", "补数据", "更充分的输入可能改善结果", "耗时"),
            _option("C", "重定向目标链", "调整目标更贴合实际", "需要重新规划"),
        ]
        return (True, "连续 3 步脱轨, 需决策换路线", opts)

    # 条件 2: TargetChain 验证失败 (required_results 全完成但结构检查 failed)
    for tc in target_chains or []:
        if not _tc_is_complete(tc):
            continue
        tid = _tc_attr(tc, "target_id", None)
        # 最近一步匹配该 target 且结构硬失败 → 验证不达标
        for ev in recent:
            if ev.target_chain_ref == tid and ev.structure_check == "failed":
                opts = [
                    _option("A", f"重做 {tid}", "重新执行可能修复", "耗时"),
                    _option("B", "换方法", "换路线绕过问题", "前期工作可能浪费"),
                    _option("C", "接受部分结果", "保住已有进展", "结果不完整"),
                ]
                return (
                    True,
                    f"TargetChain {tid} 验证失败 (结构检查 failed)",
                    opts,
                )

    # 条件 3: 连续 3 步 evidence_quality=="low" 且 kb_recall_empty
    if (
        len(recent) >= 3
        and kb_recall_empty
        and all(e.evidence_quality == "low" for e in recent)
    ):
        opts = [
            _option("A", "补文献", "增强理论依据", "耗时"),
            _option("B", "换数据源", "更高质数据可能改善证据", "需要重新采集"),
            _option("C", "接受低证据", "推进任务", "结论可信度低"),
        ]
        return (True, "证据质量持续低且 KB 无召回, 需决策补强", opts)

    return (False, "", [])


if __name__ == "__main__":
    import tempfile

    # 1. 合法转换：CREATED → RUNNING → PAUSED_ASKING_DECISION → RESUMED → RUNNING → COMPLETED
    lc = TaskLifecycle(task_id="t1")
    lc.transition(TaskState.RUNNING)
    lc.pause_for_decision(DecisionRequest(
        step_id=1,
        question="继续吗?",
        options=[{"id": "A", "label": "继续", "pros": "推进", "cons": "耗资源"}],
        context_summary="跑到第 1 步",
    ))
    assert lc.state == TaskState.PAUSED_ASKING_DECISION
    lc.resume(answer="A")
    assert lc.state == TaskState.RESUMED
    lc.transition(TaskState.RUNNING)
    lc.complete()
    assert lc.state == TaskState.COMPLETED
    assert is_terminal(lc.state)

    # 2. 非法转换 raise ValueError：COMPLETED → RUNNING
    try:
        lc.transition(TaskState.RUNNING)
        raise AssertionError("COMPLETED → RUNNING 应该 raise ValueError")
    except ValueError:
        pass

    # 3. pause_for_decision 设置 decision_request，resume 时清空
    lc2 = TaskLifecycle(task_id="t2")
    lc2.transition(TaskState.RUNNING)
    dr = DecisionRequest(
        step_id=2,
        question="选 A 还是 B?",
        options=[{"id": "A"}, {"id": "B"}],
        context_summary="ctx",
    )
    lc2.pause_for_decision(dr)
    assert lc2.decision_request is not None
    assert lc2.decision_request is dr
    assert lc2.pause_reason == "选 A 还是 B?"
    lc2.resume(answer="A")
    assert lc2.decision_request is None
    # 答案落到了原始 dr 对象上（调用方可读）
    assert dr.answer == "A"
    assert dr.answered_at is not None

    # 4. is_paused 对两个 paused 子状态都返回 True
    lc3 = TaskLifecycle(task_id="t3")
    lc3.transition(TaskState.RUNNING)
    lc3.pause_for_approval("等审批")
    assert is_paused(lc3.state)
    assert lc3.state == TaskState.PAUSED_WAITING_APPROVAL
    lc3.resume()
    lc3.transition(TaskState.RUNNING)
    lc3.pause_for_decision(DecisionRequest(step_id=1, question="q", options=[], context_summary="c"))
    assert is_paused(lc3.state)
    assert lc3.state == TaskState.PAUSED_ASKING_DECISION
    # 非暂停态返回 False
    lc3.resume()
    assert not is_paused(lc3.state)

    # 5. to_dict / from_dict 往返一致
    lc4 = TaskLifecycle(task_id="t4")
    lc4.transition(TaskState.RUNNING)
    lc4.pause_for_decision(DecisionRequest(
        step_id=5,
        question="决策问题?",
        options=[{"id": "A", "label": "选项A", "pros": "好处", "cons": "坏处"}],
        context_summary="上下文",
    ))
    d = lc4.to_dict()
    assert d["state"] == "paused_asking_decision"
    assert d["decision_request"]["question"] == "决策问题?"
    lc4_restored = TaskLifecycle.from_dict(d)
    assert lc4_restored.task_id == lc4.task_id
    assert lc4_restored.state == lc4.state
    assert lc4_restored.created_at == lc4.created_at
    assert lc4_restored.updated_at == lc4.updated_at
    assert lc4_restored.paused_at == lc4.paused_at
    assert lc4_restored.pause_reason == lc4.pause_reason
    assert lc4_restored.decision_request is not None
    assert lc4_restored.decision_request.step_id == 5
    assert lc4_restored.decision_request.options == [{"id": "A", "label": "选项A", "pros": "好处", "cons": "坏处"}]
    # decision_request=None 的情况也要能往返
    lc4_empty = TaskLifecycle(task_id="t4_empty")
    d_empty = lc4_empty.to_dict()
    assert d_empty["decision_request"] is None
    assert TaskLifecycle.from_dict(d_empty).decision_request is None

    # 6. save / load 往返一致
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        save_task_lifecycle(lc4, ws)
        loaded = load_task_lifecycle("t4", ws)
        assert loaded is not None
        assert loaded.task_id == "t4"
        assert loaded.state == TaskState.PAUSED_ASKING_DECISION
        assert loaded.decision_request is not None
        assert loaded.decision_request.question == "决策问题?"
        assert loaded.decision_request.options == lc4.decision_request.options
        assert loaded.paused_at == lc4.paused_at
        # task_id 不匹配返回 None
        assert load_task_lifecycle("other_id", ws) is None
        # 不存在的 workspace 返回 None
        assert load_task_lifecycle("t4", ws / "no_such_dir") is None
        # 原子写：目标文件存在，tmp 文件已被 replace 掉
        assert (ws / ".huginn" / "task_lifecycle.json").exists()
        assert not (ws / ".huginn" / "task_lifecycle.json.tmp").exists()

    # 7. should_pause_for_decision 四种触发条件
    @dataclass
    class _MockEval:
        on_track: str = "true"
        evidence_quality: str = "medium"
        target_chain_ref: str | None = None
        structure_check: str = "not_applicable"

    @dataclass
    class _MockTC:
        target_id: str = "T1"
        required_results: list = field(default_factory=lambda: ["formation energy"])
        completed_results: set = field(default_factory=set)

    # 条件 1: 连续 3 步 on_track=="false" → pause
    evals_false = [_MockEval(on_track="false") for _ in range(3)]
    pause, reason, opts = should_pause_for_decision(evals_false, [])
    assert pause, f"3x false → pause, got {pause}"
    assert "脱轨" in reason, f"reason: {reason}"
    assert len(opts) == 3, f"3 options, got {len(opts)}"
    assert opts[0]["id"] == "A" and "label" in opts[0] and "pros" in opts[0] and "cons" in opts[0]

    # 不触发: 只有 2 步 false (窗口不够)
    pause, _, _ = should_pause_for_decision(
        [_MockEval(on_track="false") for _ in range(2)], [])
    assert not pause, "2x false → no pause"

    # 条件 2: TargetChain 验证失败 (is_complete + structure_check failed)
    tc_done = _MockTC(completed_results={"formation energy"})
    ev_failed = _MockEval(
        on_track="true", target_chain_ref="T1", structure_check="failed")
    pause, reason, opts = should_pause_for_decision([ev_failed], [tc_done])
    assert pause, f"verification fail → pause, got {pause}"
    assert "T1" in reason, f"reason should mention T1: {reason}"
    assert len(opts) == 3

    # 不触发: target_chain 未完成
    tc_incomplete = _MockTC(completed_results=set())
    pause, _, _ = should_pause_for_decision([ev_failed], [tc_incomplete])
    assert not pause, "incomplete tc → no pause"

    # 条件 3: 连续 3 步 evidence_quality=="low" + kb_recall_empty
    evals_low = [_MockEval(evidence_quality="low") for _ in range(3)]
    pause, reason, opts = should_pause_for_decision(
        evals_low, [], kb_recall_empty=True)
    assert pause, f"3x low + empty kb → pause, got {pause}"
    assert "证据" in reason or "低" in reason, f"reason: {reason}"

    # 不触发: 3x low 但 kb_recall_empty=False
    pause, _, _ = should_pause_for_decision(evals_low, [], kb_recall_empty=False)
    assert not pause, "3x low + kb not empty → no pause"

    # 条件 4: fired intention description 含 "用户决策"
    @dataclass
    class _MockIntention:
        description: str = "需要用户决策: 是否继续"

    pause, reason, opts = should_pause_for_decision(
        [_MockEval(on_track="true")], [], fired_intentions=[_MockIntention()])
    assert pause, f"用户决策 intention → pause, got {pause}"
    assert "用户决策" in reason or "user decision" in reason.lower(), \
        f"reason: {reason}"
    assert len(opts) == 3

    # 不触发: fired intention 不含 "用户决策"
    pause, _, _ = should_pause_for_decision(
        [_MockEval(on_track="true")], [],
        fired_intentions=[_MockIntention(description="just a reminder")])
    assert not pause, "non-decision intention → no pause"

    # 空 evaluations + 无 fired intentions → 不触发
    pause, _, _ = should_pause_for_decision([], [])
    assert not pause, "empty → no pause"

    # dict 形式 target_chain 也兼容
    tc_dict = {"target_id": "T2", "required_results": ["band gap"],
               "completed_results": {"band gap"}}
    ev_dict = _MockEval(target_chain_ref="T2", structure_check="failed")
    pause, reason, _ = should_pause_for_decision([ev_dict], [tc_dict])
    assert pause and "T2" in reason, f"dict tc → pause, got {pause}/{reason}"

    # 条件 0: grill 触发 (优先级最高, 先于 PMK / fired intentions / 等)
    # 高歧义 → grill
    pause, reason, opts = should_pause_for_decision(
        [], [], grill_state={"ambiguity_score": 0.8})
    assert pause, "grill_state ambiguity=0.8 → pause"
    assert "GRILL" in reason, f"reason 应含 GRILL: {reason}"
    assert len(opts) == 3 and opts[0]["id"] == "A"

    # tier A + 歧义 scene → grill
    pause, reason, _ = should_pause_for_decision(
        [], [], grill_state={"tier": "A", "scene_tag": "ambiguous_req"})
    assert pause and "GRILL" in reason

    # plan 为空 → grill
    pause, reason, _ = should_pause_for_decision(
        [], [], grill_state={"plan_is_empty": True})
    assert pause and "plan 为空" in reason

    # has_grilled=True 不重复触发
    pause, _, _ = should_pause_for_decision(
        [], [], grill_state={"ambiguity_score": 0.9, "has_grilled": True})
    assert not pause, "已 grill 过不应重复触发"

    # grill_state=None 兼容旧调用
    pause, _, _ = should_pause_for_decision([], [])
    assert not pause, "grill_state=None 兼容旧调用"

    # grill 优先级高于 PMK: 即使 PMK 冲突也先 grill (grill 期间不应有 PMK)
    pause, reason, _ = should_pause_for_decision(
        [], [],
        pmk_state={"persona": "推荐用 GNN", "memory": "反对 GNN",
                    "kb": "", "deviation": ""},
        grill_state={"ambiguity_score": 0.7},
    )
    assert pause and "GRILL" in reason, "grill 优先级高于 PMK"

    print("All self-checks passed.")


async def check_pmk_consistency_llm(
    pmk_state: dict, llm_chat_fn,
) -> tuple[bool, str]:
    """v8: LLM 判定 PMK 三路立场语义一致性. 规则版抓显式冲突, LLM 版抓隐式冲突.

    规则版 _check_pmk_consistency 用 token 重合度判 subject 相似, 漏同义改写
    (e.g. "use GNN" vs "graph neural network approach"). LLM 版做语义判断.

    调用时机: 规则版判 (False, "") 时, 若三路都有非空文本, 调 LLM 做二次确认.
    规则版判 True 时直接 pause, 不需要 LLM (显式冲突已够).

    llm_chat_fn: async callable, 接 prompt 返回 str. 失败返回 (False, "").

    ponytail: 不每步调 (成本高), 只在规则版无冲突但三路都有立场时调.
    升级路径: 缓存 LLM 判断, 同一 pmk_state 不重复调.
    """
    if not pmk_state:
        return (False, "")
    texts = {k: v for k, v in pmk_state.items()
             if k in ("persona", "memory", "kb") and v and str(v).strip()}
    if len(texts) < 2:
        return (False, "")
    prompt = """Judge if these three stances (Persona/Memory/Knowledge) are semantically consistent.

Persona stance: {persona}
Memory stance: {memory}
Knowledge stance: {kb}

Two stances CONFLICT if they take opposing positions on the SAME subject (even with different wording, e.g. "use GNN" vs "avoid graph neural networks" = conflict on GNN usage).

Respond JSON only:
{{"consistent": true/false, "conflict_pair": "persona-vs-memory|persona-vs-kb|memory-vs-kb|none", "reason": "1 sentence why"}}""".format(
        persona=texts.get("persona", "NONE")[:300],
        memory=texts.get("memory", "NONE")[:300],
        kb=texts.get("kb", "NONE")[:300],
    )
    try:
        resp = await llm_chat_fn(prompt)
        if not resp:
            return (False, "")
        resp = resp.strip()
        # 提取 JSON
        import json as _json
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end <= start:
            return (False, "")
        data = _json.loads(resp[start:end + 1])
        is_consistent = bool(data.get("consistent", True))
        if not is_consistent:
            pair = data.get("conflict_pair", "unknown")
            reason = data.get("reason", "LLM detected semantic conflict")
            return (True, f"LLM: {pair} conflict — {reason}")
        return (False, "")
    except Exception:
        return (False, "")
