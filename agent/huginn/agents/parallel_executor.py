"""工具并行执行器 —— 一次性把多个独立工具调用并发跑掉.

LLM 经常会一次返回多个 tool_call (查 SiO2 同时查 TiO2, 或者读 POSCAR
同时算 lattice), langgraph 自带的 ToolNode 已经能把同一 AIMessage 里
的多个 tool_call 并发 dispatch. 但 ToolNode 不区分依赖关系 —— 如果
tool B 的输入引用了 tool A 的输出, B 必须等 A 跑完拿到结果才能调,
这种依赖场景硬并行会拿到空输入.

这个 executor 做两件事:
  1. are_independent(calls) —— 启发式判断调用之间有没有依赖关系,
     看后一个 call 的 tool_input 里有没有显式引用前一个 call 的输出
     (比如 {"_depends_on": 0} 或者参数里出现 "{0.result}" 这种占位符)
  2. execute_parallel(calls) —— 用 asyncio.gather 并发执行, 每个
     调用独立计时, 单个失败不影响其它, 返回带 error/dt 的结果列表

调用方拿到结果自己决定要不要把依赖串行跑第二遍.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# 依赖标记: tool_input 里出现这些键/模式, 就认为该 call 依赖前一个 call.
# 1. 显式键: {"_depends_on": <index>} 或 {"_after": <index>}
# 2. 占位符: "{0.result}", "{1.output}", "${0}" 等
_DEPENDS_KEY = re.compile(r"^\s*(?:_depends_on|_after|_depends_index)\s*$")
_PLACEHOLDER = re.compile(r"\{(?P<idx>\d+)\.[^}]+\}|\$\{(?P<idx2>\d+)\}")


class ParallelToolExecutor:
    """并发执行多个独立工具调用, 自动隔离错误.

    用法::

        executor = ParallelToolExecutor(invoke_fn=lambda name, inp: ...)
        results = await executor.execute_parallel([
            {"tool_name": "materials_database_tool", "tool_input": {...}},
            {"tool_name": "structure_tool", "tool_input": {...}},
        ])
        # results: [{"tool_name": ..., "result": ..., "error": ..., "dt": ...}, ...]
    """

    def __init__(
        self,
        invoke_fn: Any | None = None,
        max_concurrency: int = 8,
    ) -> None:
        """初始化.

        Args:
            invoke_fn: 异步 callable, 签名 (tool_name, tool_input) -> result.
                默认 None, 调用方可以传 self._invoke_tool 之类的. 如果不
                传, execute_parallel 会抛 ValueError, 强制调用方提供.
            max_concurrency: 最大并发数, 用 asyncio.Semaphore 限流, 防一次
                扔几十个 IO 把外部 API 打爆. 默认 8, 对大多数 materials
                数据库够用.
        """
        if invoke_fn is None:
            raise ValueError(
                "ParallelToolExecutor 需要 invoke_fn (异步 callable) 才能执行"
            )
        self._invoke = invoke_fn
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------ API

    async def execute_parallel(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """并发执行多个工具调用.

        Args:
            calls: [{"tool_name": str, "tool_input": dict}, ...]

        Returns:
            跟 calls 等长的列表, 每项形如:
              {"tool_name": str, "result": Any | None, "error": str | None, "dt": float}
            单个调用失败不会中断其它调用, 错误塞进 error 字段.
        """
        if not calls:
            return []

        # 单个调用直接跑, 不绕一圈 gather
        if len(calls) == 1:
            return [await self._run_one(calls[0])]

        # 多个调用: 用 semaphore 限并发, gather 并行
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks = [self._run_one(c) for c in calls]
        return await asyncio.gather(*tasks)

    def are_independent(self, calls: list[dict[str, Any]]) -> bool:
        """启发式判断调用之间有没有依赖关系.

        判定依据:
          - 任意一个 call 的 tool_input 里出现 _depends_on / _after /
            _depends_index 键 → 有依赖
          - 任意一个 call 的 tool_input (递归扫字符串值) 里出现
            "{0.result}" / "${1}" 这种占位符 → 有依赖

        注意: 这是保守判断, 误报 "有依赖" 比误报 "独立" 安全得多 ——
        误报独立会导致并行跑拿到空输入. 所以宁可串行也别错并.
        """
        if len(calls) < 2:
            return True

        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_input = call.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            if self._has_dependency_marker(tool_input):
                return False
        return True

    # ------------------------------------------------------------------ helpers

    async def _run_one(self, call: dict[str, Any]) -> dict[str, Any]:
        """跑单个工具调用, 捕获异常, 计时.

        单个调用挂掉不影响 gather 里其它任务, 错误塞进 error 字段返回.
        """
        tool_name = call.get("tool_name", "unknown")
        tool_input = call.get("tool_input", {}) or {}
        start = time.time()
        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    result = await self._invoke(tool_name, tool_input)
            else:
                result = await self._invoke(tool_name, tool_input)
            dt = time.time() - start
            return {
                "tool_name": tool_name,
                "result": result,
                "error": None,
                "dt": dt,
            }
        except Exception as exc:
            dt = time.time() - start
            logger.warning(
                "parallel tool call %s failed: %s", tool_name, exc, exc_info=True
            )
            return {
                "tool_name": tool_name,
                "result": None,
                "error": str(exc),
                "dt": dt,
            }

    @classmethod
    def _has_dependency_marker(cls, obj: Any) -> bool:
        """递归扫 tool_input, 看有没有依赖标记.

        dict 里出现 _depends_on / _after / _depends_index 键, 或任意字符串
        值里出现 "{0.xxx}" / "${0}" 占位符, 就算有依赖.
        """
        if isinstance(obj, dict):
            for key in obj.keys():
                if isinstance(key, str) and _DEPENDS_KEY.match(key):
                    return True
            for value in obj.values():
                if cls._has_dependency_marker(value):
                    return True
            return False
        if isinstance(obj, list):
            return any(cls._has_dependency_marker(v) for v in obj)
        if isinstance(obj, str):
            return bool(_PLACEHOLDER.search(obj))
        return False

    def status(self) -> dict[str, Any]:
        """返回当前执行器配置, 方便 debug."""
        return {
            "max_concurrency": self._max_concurrency,
            "has_invoke_fn": self._invoke is not None,
        }


def split_by_dependency(
    calls: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """把一串 calls 切成若干 "可以并行" 的批次.

    每个批次内部都是独立调用, 可以丢给 execute_parallel 并发跑;
    批次之间串行, 前一批跑完才能跑下一批. 用 are_independent 的
    同款规则判断依赖.

    Returns:
        [{batch 1 calls}, {batch 2 calls}, ...]. 单元素批次表示该 call
        必须单独串行跑.
    """
    if not calls:
        return []
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def _flush():
        if current:
            batches.append(list(current))
            current.clear()

    for call in calls:
        if current and not ParallelToolExecutor._has_dependency_marker(
            call.get("tool_input", {})
        ):
            # 当前 call 跟 previous 独立, 加进当前 batch
            current.append(call)
        else:
            # 要么 current 是空的 (这是第一个 call), 要么该 call 有依赖
            # 标记 —— 都得先把当前 batch 切掉, 然后该 call 单独成 batch
            if current and ParallelToolExecutor._has_dependency_marker(
                call.get("tool_input", {})
            ):
                _flush()
                batches.append([call])
            else:
                current.append(call)
    _flush()
    return batches
