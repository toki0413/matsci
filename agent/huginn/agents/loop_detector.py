"""工具调用循环检测 —— 抓 LLM 反复调同一个工具的死循环.

跟 ToolCallBudget 互补: budget 只看次数, 不看调用模式. 这个 detector
专门看 "同工具同参数连调" 这种典型循环 —— LLM 拿到错误结果后, 没换
思路, 又把一样的参数塞回去, 几轮下来纯烧 token.

判定规则:
  - 同工具 + 同参数 (hash) 连续调用 ≥ 3 次 → 循环
  - 同工具连续调用 ≥ 5 次 (即使参数不同) → 疑似循环
两条规则任一命中就算 loop, agent 拿到 reason 喂回 LLM, 让它换思路.

设计要点:
  - 只看最近 N 次 (默认 10), 太老的不算, 避免 LLM 多步计算被误伤
  - 同工具不同参数是正常的多步计算 (比如先 relax 再 scf), 不算循环
  - 线程安全, 跟 budget / router 一样按单轮 chat 生命周期管理
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from typing import Any

from huginn.utils.common import hash_text


# 同工具同参数连续命中多少次算循环
_SAME_PARAMS_THRESHOLD = 3

# 同工具连续调用多少次 (参数可不同) 算疑似循环
_SAME_TOOL_THRESHOLD = 5

# 滑动窗口大小, 只看最近 N 次调用, 避免长任务被误判
_DEFAULT_WINDOW = 10


def _hash_tool_input(tool_input: Any) -> str:
    """把工具参数 hash 一下, 用来判断 "同参数".

    用 json + sha256, 对 dict 顺序不敏感 (sort_keys=True). 任意类型
    都尽量序列化, 序列化失败的退回到 str() 兜底, 别让异常打断检测.
    """
    try:
        payload = json.dumps(tool_input, sort_keys=True, default=str)
    except Exception:
        payload = str(tool_input)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LoopDetector:
    """单轮 agent chat 内的工具调用循环检测器.

    用法::

        detector = LoopDetector()
        is_loop = detector.record("vasp_tool", {"action": "relax", ...})
        if is_loop:
            stop, reason = detector.should_break()
            if stop:
                return {"error": reason}
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        same_params_threshold: int = _SAME_PARAMS_THRESHOLD,
        same_tool_threshold: int = _SAME_TOOL_THRESHOLD,
    ) -> None:
        self.window_size = max(window_size, same_tool_threshold + 1)
        self.same_params_threshold = same_params_threshold
        self.same_tool_threshold = same_tool_threshold
        self._lock = threading.RLock()
        # 滑动窗口: 最近 N 次 (tool_name, param_hash)
        self._history: deque[tuple[str, str]] = deque(maxlen=self.window_size)
        # 最近一次命中的循环描述, 给 should_break / get_pattern 用
        self._last_loop: dict[str, Any] | None = None

    # ------------------------------------------------------------------ API

    def record(self, tool_name: str, tool_input: Any) -> bool:
        """记一笔调用, 返回是否检测到循环.

        不管返回啥都会把这次调用塞进历史窗口. 调用方拿到 True 就该
        走 should_break() 拿 reason 喂回 LLM.
        """
        param_hash = _hash_tool_input(tool_input)
        with self._lock:
            self._history.append((tool_name, param_hash))
            return self._detect(tool_name, param_hash)

    def should_break(self) -> tuple[bool, str]:
        """返回 (是否该 break, 原因).

        原因里会带上 LLM 能看懂的提示: 哪个工具连调了几次, 该怎么破.
        没检测到循环就返回 (False, "").
        """
        with self._lock:
            if not self._last_loop:
                return False, ""
            loop = self._last_loop
            tool = loop["tool_name"]
            kind = loop["kind"]
            count = loop["count"]

            if kind == "same_params":
                reason = (
                    f"Detected tool call loop: {tool} called {count} times "
                    f"with same params. Break the loop by trying a different "
                    f"approach."
                )
            else:  # same_tool
                reason = (
                    f"Detected suspected tool call loop: {tool} called "
                    f"{count} times consecutively with varying params. "
                    f"Consider whether the previous results are sufficient "
                    f"to proceed, or switch to a different tool."
                )
            return True, reason

    def get_pattern(self) -> dict[str, Any]:
        """返回当前循环模式, 用于反馈 LLM 或写入 telemetry.

        没有循环时返回空 dict. 有循环时包含:
          - tool_name: 触发循环的工具名
          - kind: "same_params" / "same_tool"
          - count: 连续命中次数
          - recent_calls: 最近几次调用的 (tool, hash) 列表, 方便 debug
        """
        with self._lock:
            if not self._last_loop:
                return {}
            pattern = dict(self._last_loop)
            pattern["recent_calls"] = list(self._history)
            return pattern

    def reset(self) -> None:
        """清空历史 (下一轮 agent chat 开始时用)."""
        with self._lock:
            self._history.clear()
            self._last_loop = None

    def status(self) -> dict[str, Any]:
        """返回当前检测器状态, 方便 debug / telemetry."""
        with self._lock:
            return {
                "window_size": self.window_size,
                "same_params_threshold": self.same_params_threshold,
                "same_tool_threshold": self.same_tool_threshold,
                "history_size": len(self._history),
                "last_loop": dict(self._last_loop) if self._last_loop else None,
            }

    # ------------------------------------------------------------------ internal

    def _detect(self, tool_name: str, param_hash: str) -> bool:
        """从窗口尾部往回看, 判断是否命中循环规则.

        必须在持锁状态下调用. 命中就把 _last_loop 填上, 方便
        should_break / get_pattern 拿.
        """
        if len(self._history) < self.same_params_threshold:
            return False

        # 从最近一次往回数连续相同的 (tool, hash) 段
        same_params_run = self._count_trailing_same(tool_name, param_hash)
        if same_params_run >= self.same_params_threshold:
            self._last_loop = {
                "tool_name": tool_name,
                "kind": "same_params",
                "count": same_params_run,
                "param_hash": param_hash,
            }
            return True

        # 同工具连续调用 (参数可不同) — 从尾部往回数同工具的连续段
        same_tool_run = self._count_trailing_same_tool(tool_name)
        if same_tool_run >= self.same_tool_threshold:
            self._last_loop = {
                "tool_name": tool_name,
                "kind": "same_tool",
                "count": same_tool_run,
                "param_hash": param_hash,
            }
            return True

        return False

    def _count_trailing_same(self, tool_name: str, param_hash: str) -> int:
        """从窗口尾部往回数, 连续 (tool_name, param_hash) 都相同的次数."""
        count = 0
        for t, h in reversed(self._history):
            if t == tool_name and h == param_hash:
                count += 1
            else:
                break
        return count

    def _count_trailing_same_tool(self, tool_name: str) -> int:
        """从窗口尾部往回数, 连续 tool_name 相同 (参数可不同) 的次数."""
        count = 0
        for t, _ in reversed(self._history):
            if t == tool_name:
                count += 1
            else:
                break
        return count

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"LoopDetector(history={len(self._history)}, "
                f"last_loop={self._last_loop})"
            )


# ── 思考循环检测 ────────────────────────────────────────────────


def _similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using Jaccard on word sets.

    Returns 0.0-1.0 where 1.0 means identical word sets.
    Fast and language-agnostic, suitable for loop detection.
    """
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


class ThoughtLoopDetector:
    """检测 LLM 思考/输出陷入死循环.

    跟 LoopDetector 互补: LoopDetector 看工具调用模式, 这个看
    LLM 的输出文本模式. 场景:
      - LLM 反复生成几乎相同的回答 (word similarity > 0.85)
      - LLM 在 plan/clarification 之间无限来回
      - LLM 反复说 "let me try again" 但没有实质进展

    判定规则:
      - 连续 ≥3 次输出相似度 > 0.85 → 死循环
      - 连续 ≥5 次输出相似度 > 0.60 → 疑似循环
      - 同一文本完全重复 ≥2 次 → 立即判定

    检测到循环后, agent 应:
      1. 向 LLM 发送 "break" 指令, 强制换思路
      2. 如果连续 2 次 break 后仍未脱离, 终止执行
    """

    def __init__(
        self,
        window_size: int = 10,
        high_sim_threshold: float = 0.85,
        low_sim_threshold: float = 0.60,
        high_sim_count: int = 3,
        low_sim_count: int = 5,
        exact_dup_count: int = 2,
    ) -> None:
        self.window_size = window_size
        self.high_sim_threshold = high_sim_threshold
        self.low_sim_threshold = low_sim_threshold
        self.high_sim_count = high_sim_count
        self.low_sim_count = low_sim_count
        self.exact_dup_count = exact_dup_count
        self._lock = threading.RLock()
        self._outputs: deque[str] = deque(maxlen=window_size)
        self._break_count = 0  # 已发送 break 指令的次数
        self._last_loop: dict[str, Any] | None = None

    def record_output(self, text: str) -> bool:
        """记录一次 LLM 输出, 返回是否检测到循环."""
        with self._lock:
            self._outputs.append(text)
            return self._detect(text)

    def should_break(self) -> tuple[bool, str, bool]:
        """返回 (是否该 break, 原因, 是否该终止).

        终止标志为 True 时, agent 应立即停止执行, 不再重试.
        """
        with self._lock:
            if not self._last_loop:
                return False, "", False

            loop = self._last_loop
            kind = loop["kind"]
            count = loop["count"]

            if kind == "exact_dup":
                reason = (
                    f"LLM output repeated exactly {count} times. "
                    f"The model is stuck. Breaking to try a different approach."
                )
            elif kind == "high_sim":
                reason = (
                    f"LLM output has been very similar {count} times "
                    f"(similarity > {self.high_sim_threshold}). "
                    f"The model may be in a loop. Try a different strategy."
                )
            else:
                reason = (
                    f"LLM output has been similar {count} times "
                    f"(similarity > {self.low_sim_threshold}). "
                    f"Consider whether the current approach is working."
                )

            self._break_count += 1
            # 连续 break 2 次仍未脱离 → 终止
            should_terminate = self._break_count >= 2
            if should_terminate:
                reason = (
                    f"Thought loop persists after {self._break_count} breaks. "
                    f"Terminating to prevent infinite loop."
                )

            return True, reason, should_terminate

    def reset(self) -> None:
        """清空历史 (下一轮 agent chat 开始时用)."""
        with self._lock:
            self._outputs.clear()
            self._break_count = 0
            self._last_loop = None

    def status(self) -> dict[str, Any]:
        """返回当前检测器状态."""
        with self._lock:
            return {
                "outputs_tracked": len(self._outputs),
                "break_count": self._break_count,
                "last_loop": self._last_loop,
                "high_sim_threshold": self.high_sim_threshold,
                "low_sim_threshold": self.low_sim_threshold,
            }

    def _detect(self, text: str) -> bool:
        """检测当前输出是否形成循环."""
        outputs = list(self._outputs)

        # Rule 1: Exact duplicate
        text_hash = hash_text(text)
        exact_count = sum(1 for o in outputs if hash_text(o) == text_hash)
        if exact_count >= self.exact_dup_count:
            self._last_loop = {
                "kind": "exact_dup",
                "count": exact_count,
                "text_preview": text[:100],
            }
            return True

        # Rule 2: High similarity (≥ 0.85) consecutive
        if len(outputs) >= self.high_sim_count:
            recent = outputs[-(self.high_sim_count):]
            all_high = all(
                _similarity(text, o) >= self.high_sim_threshold
                for o in recent
                if o != text
            )
            if all_high and len(recent) >= self.high_sim_count - 1:
                self._last_loop = {
                    "kind": "high_sim",
                    "count": self.high_sim_count,
                    "similarity": self.high_sim_threshold,
                    "text_preview": text[:100],
                }
                return True

        # Rule 3: Low similarity (≥ 0.60) consecutive
        if len(outputs) >= self.low_sim_count:
            recent = outputs[-(self.low_sim_count):]
            all_low = all(
                _similarity(text, o) >= self.low_sim_threshold
                for o in recent
                if o != text
            )
            if all_low and len(recent) >= self.low_sim_count - 1:
                self._last_loop = {
                    "kind": "low_sim",
                    "count": self.low_sim_count,
                    "similarity": self.low_sim_threshold,
                    "text_preview": text[:100],
                }
                return True

        return False

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"ThoughtLoopDetector(outputs={len(self._outputs)}, "
                f"breaks={self._break_count}, "
                f"last_loop={self._last_loop})"
            )
