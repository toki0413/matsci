"""钩子系统: 工具调用前后 / 会话生命周期 / 上下文压缩 等事件可注入自定义逻辑.

事件清单 (对齐 Claude Code):
- pre_tool_use:           工具调用前, 可拦截/改入参/阻断
- post_tool_use:          工具调用后, 观察结果和耗时
- post_tool_use_failure:  工具调用失败后, 专门处理异常路径
- session_start:          会话开始
- session_end:            会话结束
- stop:                   agent 完成一轮回复
- subagent_stop:          子 agent 完成时
- pre_compact:            上下文压缩前
- user_prompt_submit:     用户提交输入后

钩子是 async 的, 按 FIFO 顺序执行. 任意一个 pre 钩子置 blocked=True 即阻止.
单个钩子抛异常不会中断主流程, 只记日志.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 事件名常量, register() 时用
PRE_TOOL_USE = "pre_tool_use"
POST_TOOL_USE = "post_tool_use"
# 新增事件
SESSION_START = "session_start"            # 会话开始时触发
SESSION_END = "session_end"                # 会话结束时触发
STOP = "stop"                              # agent 完成一轮回复后触发
SUBAGENT_STOP = "subagent_stop"            # 子 agent 完成时触发
PRE_COMPACT = "pre_compact"                # 上下文压缩前触发
USER_PROMPT_SUBMIT = "user_prompt_submit"  # 用户提交输入后触发
POST_TOOL_USE_FAILURE = "post_tool_use_failure"  # 工具调用失败后触发

# 全部事件, 给 register() 校验和 __init__ 初始化用
ALL_EVENTS: tuple[str, ...] = (
    PRE_TOOL_USE,
    POST_TOOL_USE,
    SESSION_START,
    SESSION_END,
    STOP,
    SUBAGENT_STOP,
    PRE_COMPACT,
    USER_PROMPT_SUBMIT,
    POST_TOOL_USE_FAILURE,
)

HookCallback = Callable[["HookContext"], Awaitable["HookContext | None"]]
"""钩子回调签名: 接收 ctx, 返回修改后的 ctx 或 None.

- 返回 None: 放行, ctx 不变
- 返回 ctx: 用返回的 ctx 覆盖(可改 args / 置 blocked)
"""


@dataclass
class HookContext:
    """单次工具调用的上下文, 贯穿 pre/post 两个阶段.

    对于非工具事件 (session_start / stop / pre_compact 等), tool_name
    填一个语义化的占位串 (如 'agent_turn' / 'context_compact'), args/result
    留空, 真正的数据放 metadata 里.
    """

    tool_name: str
    args: Any = None
    # post 阶段才填充
    result: Any = None
    error: BaseException | None = None
    duration_ms: float = 0.0
    # pre 钩子置 True 直接阻断调用
    blocked: bool = False
    # 自由扩展位, 钩子之间传递数据用
    metadata: dict[str, Any] = field(default_factory=dict)


class HookManager:
    """管理各类钩子的注册和触发.

    不是线程安全的 —— agent 单线程异步跑, 没加锁. 多线程环境外层自己加锁.
    """

    def __init__(self) -> None:
        # event -> 回调列表, 保持注册顺序. 全部事件都预初始化, 避免 KeyError.
        self._callbacks: dict[str, list[HookCallback]] = {
            ev: [] for ev in ALL_EVENTS
        }

    def register(self, event: str, callback: HookCallback) -> None:
        """注册钩子. event 必须是 ALL_EVENTS 里的一个."""
        if event not in self._callbacks:
            raise ValueError(
                f"Unknown hook event: {event!r}. "
                f"Supported: {', '.join(ALL_EVENTS)}"
            )
        self._callbacks[event].append(callback)

    def clear(self, event: str | None = None) -> None:
        """清空钩子. event=None 清所有事件."""
        if event is None:
            for key in self._callbacks:
                self._callbacks[key].clear()
        elif event in self._callbacks:
            self._callbacks[event].clear()

    def has_hooks(self, event: str) -> bool:
        """有没有注册过某事件的钩子."""
        return bool(self._callbacks.get(event))

    async def trigger(self, event: str, ctx: HookContext) -> HookContext:
        """通用事件触发器.

        对没有专门 run_* 方法的 event (session_start / stop / pre_compact
        / user_prompt_submit / subagent_stop / post_tool_use_failure 等)
        统一走这里. 回调返回的 ctx 会覆盖入参, 异常被吞掉只记日志, 不影响
        后续回调.
        """
        callbacks = self._callbacks.get(event, [])
        for cb in callbacks:
            try:
                ret = await cb(ctx)
                if ret is not None:
                    ctx = ret
            except Exception:
                # 单个钩子出错不能把整个事件链搞挂
                logger.warning(
                    "%s hook raised", event, exc_info=True
                )
        return ctx

    async def run_pre(
        self,
        tool_name: str,
        args: Any,
        thread_id: str | None = None,
    ) -> tuple[bool, Any, "HookContext"]:
        """依次跑 pre_tool_use 钩子.

        Returns:
            (allowed, args, ctx) — allowed=False 表示被阻断,
            args 可能被钩子改写过, ctx.metadata 里可能有 block_reason
            等额外信息给调用方使用.
        """
        callbacks = self._callbacks[PRE_TOOL_USE]
        ctx = HookContext(tool_name=tool_name, args=args)
        # thread_id 塞进 metadata, pre 钩子(如 DesignPlanGate)要靠它按会话隔离
        if thread_id:
            ctx.metadata["thread_id"] = thread_id
        if not callbacks:
            return True, args, ctx
        for cb in callbacks:
            try:
                ret = await cb(ctx)
                if ret is not None:
                    # 钩子返回了修改后的 ctx, 采用它
                    ctx = ret
                if ctx.blocked:
                    logger.info("pre_tool_use hook blocked %s", tool_name)
                    return False, ctx.args, ctx
            except Exception:
                # 单个钩子出错不阻断主流程
                logger.warning(
                    "pre_tool_use hook for %s raised", tool_name, exc_info=True
                )
        return True, ctx.args, ctx

    async def run_post(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        error: BaseException | None,
        duration_ms: float,
        thread_id: str | None = None,
        user_message: str | None = None,
    ) -> None:
        """依次跑 post_tool_use 钩子. 纯观察性质, 返回值忽略.

        user_message 是当前回合的用户提问, 可选. PRT Level 1 的 LLM
        判定钩子靠它识别"用户给的值"和"工具返回的值"是否冲突, 没有就
        只看工具自身的 args/result.
        """
        callbacks = self._callbacks[POST_TOOL_USE]
        if not callbacks:
            return

        ctx = HookContext(
            tool_name=tool_name,
            args=args,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        if thread_id:
            ctx.metadata["thread_id"] = thread_id
        if user_message:
            ctx.metadata["user_message"] = user_message
        for cb in callbacks:
            try:
                await cb(ctx)
            except Exception:
                logger.warning(
                    "post_tool_use hook for %s raised", tool_name, exc_info=True
                )

        # 工具失败时再补一个 POST_TOOL_USE_FAILURE 事件, 让专门处理异常的
        # 钩子也能拿到上下文. 普通观察钩子不需要关心失败, 所以单独走一个事件.
        if error is not None and self._callbacks[POST_TOOL_USE_FAILURE]:
            fail_ctx = HookContext(
                tool_name=tool_name,
                args=args,
                result=result,
                error=error,
                duration_ms=duration_ms,
            )
            if thread_id:
                fail_ctx.metadata["thread_id"] = thread_id
            if user_message:
                fail_ctx.metadata["user_message"] = user_message
            await self.trigger(POST_TOOL_USE_FAILURE, fail_ctx)


# ---------------------------------------------------------------------------
# PRT Level 0 — 异常登记钩子
# ---------------------------------------------------------------------------

# 常见材料晶格常数标准值 (Å). 只内置几个最常碰到的, 别往里塞一堆.
# key 是约化化学式(人类习惯写法), value 里 a/b/c 缺省表示该材料没用那个参数.
_STANDARD_LATTICE: dict[str, dict[str, float]] = {
    "Si":   {"a": 5.43},
    "GaAs": {"a": 5.65},
    "Cu":   {"a": 3.61},
    "Fe":   {"a": 2.87},
    "TiO2": {"a": 4.59, "c": 2.96},
    "ZnO":  {"a": 3.25, "c": 5.21},
}


def _parse_formula(formula: str) -> dict[str, int]:
    """解析化学式 -> {元素: 原子数}. 不区分大小写错误, 只按元素符号切."""
    import re
    counts: dict[str, int] = {}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not elem:
            continue
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def _canonical_key(counts: dict[str, int]) -> tuple:
    """把 {元素: 数} 约化到最简整数比, 再按元素名排序成 tuple.
    两边(标准值 / 输入)都过一遍这个函数, 顺序就不重要了.
    """
    import math
    vals = list(counts.values())
    if not vals:
        return ()
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    return tuple(sorted((e, c // g) for e, c in counts.items()))


# 标准值按 canonical key 建一张反查表, 匹配时直接 dict 查
_STANDARD_LATTICE_LOOKUP: dict[tuple, str] = {
    _canonical_key(_parse_formula(k)): k for k in _STANDARD_LATTICE
}

# 材料中文名, 给 description 用
_MATERIAL_CN = {
    "Si": "硅", "GaAs": "砷化镓", "Cu": "铜", "Fe": "铁",
    "TiO2": "二氧化钛", "ZnO": "氧化锌",
}

# 工具返回里出现这些词, 认定结果"不合格"
_VALIDATE_FAIL_KEYWORDS = ("不合规", "不合格", "未通过", "不合理")
# 冲突类关键词
_CONFLICT_KEYWORDS = ("冲突", "矛盾", "不一致")

# 偏差阈值: 15% 已经很显著 (1% 就能改变电子结构)
_DEVIATION_THRESHOLD = 0.15

# 钩子要拦截的工具
_WATCHED_TOOLS = {"validate_tool", "numerical_tool", "structure_tool"}


class AnomalyDetectionHook:
    """PostToolUse 钩子: 扫工具输出, 把异常信号登记进 AnomalyLogStore.

    只登记, 不回顾, 不打断主流程. 一条工具调用可能命中多个信号,
    每个 signal 各登记一条. 同一次调用里同 category 的只记第一条, 避免刷屏.
    """

    def __init__(self, store: "AnomalyLogStore") -> None:
        # 延后导入, 避免 hooks 模块硬依赖 anomaly_log (循环导入风险)
        self._store = store

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            if ctx.tool_name not in _WATCHED_TOOLS:
                return None
            for anomaly in self._detect(ctx):
                self._store.log(anomaly)
        except Exception:
            # 登记本身不能把 agent 搞挂
            logger.warning("AnomalyDetectionHook raised", exc_info=True)
        return None

    # ---- 检测逻辑 ----

    def _detect(self, ctx: HookContext) -> list:
        from datetime import datetime

        from huginn.anomaly_log import Anomaly

        results: list[Anomaly] = []
        seen_categories: set[str] = set()

        tool = ctx.tool_name
        args = ctx.args if isinstance(ctx.args, dict) else {}
        result = ctx.result if isinstance(ctx.result, dict) else {}
        # serialize 出来的结构: 成功 {"result": {...}}, 失败 {"error": "..."}
        result_data = result.get("result") if isinstance(result.get("result"), dict) else {}
        result_text = self._stringify(result)

        def add(anomaly: Anomaly) -> None:
            if anomaly.category in seen_categories:
                return
            seen_categories.add(anomaly.category)
            results.append(anomaly)

        # 1) 工具抛异常 或 返回 error 字段 -> TOOL_FAILURE
        if ctx.error is not None or result.get("error"):
            err_msg = result.get("error") or str(ctx.error)
            add(Anomaly(
                id="",
                ts=datetime.now(),
                category="TOOL_FAILURE",
                severity="MEDIUM",
                description=f"{tool} 调用失败: {self._truncate(err_msg, 160)}",
                detection_method="tool_call_failed",
                source="tool_output",
                context_snapshot={
                    "tool": tool,
                    "args": self._shrink(args),
                    "error": self._truncate(err_msg, 400),
                    "duration_ms": ctx.duration_ms,
                },
                unresolved_dimensions=[f"{tool} 为什么失败？需要重试还是换参数？"],
            ))
            # 工具都挂了, 后面的数据校验意义不大, 直接返回
            return results

        # 2) validate_tool 返回不合格 -> COMPUTATION_RESULT
        if tool == "validate_tool":
            all_passed = result_data.get("all_passed")
            checks = result_data.get("checks") or []
            failed_msgs = [
                c.get("message") or c.get("name")
                for c in checks
                if not c.get("passed", True)
            ]
            keyword_hit = any(kw in result_text for kw in _VALIDATE_FAIL_KEYWORDS)
            if all_passed is False or keyword_hit or failed_msgs:
                desc = "validate_tool 判定结果不合格"
                if failed_msgs:
                    desc += f": {self._truncate('; '.join(str(m) for m in failed_msgs[:3]), 160)}"
                add(Anomaly(
                    id="",
                    ts=datetime.now(),
                    category="COMPUTATION_RESULT",
                    severity="HIGH",
                    description=desc,
                    detection_method="validate_tool_error",
                    source="tool_output",
                    context_snapshot={
                        "tool": tool,
                        "args": self._shrink(args),
                        "all_passed": all_passed,
                        "failed_checks": failed_msgs[:5],
                        "result_type": args.get("result_type"),
                    },
                    unresolved_dimensions=[
                        "计算结果物理上为什么不合理？参数/模型/输入哪个环节出问题？"
                    ],
                ))

        # 3) 晶格常数与标准值偏差 > 20% -> INPUT_DATA
        if tool == "structure_tool":
            for anomaly in self._check_lattice(args, result_data):
                add(anomaly)

        # 4) 输出含冲突类关键词 -> DATA_CONFLICT
        if any(kw in result_text for kw in _CONFLICT_KEYWORDS):
            add(Anomaly(
                id="",
                ts=datetime.now(),
                category="DATA_CONFLICT",
                severity="MEDIUM",
                description=f"{tool} 输出疑似存在数据冲突",
                detection_method="keyword_match",
                source="tool_output",
                context_snapshot={
                    "tool": tool,
                    "args": self._shrink(args),
                    "snippet": self._truncate(result_text, 400),
                },
                unresolved_dimensions=["多源数据哪里对不上？以哪个为准？"],
            ))

        return results

    def _check_lattice(self, args: dict, result_data: dict) -> list:
        """对比 structure_tool 输出的晶格常数和标准值."""
        from datetime import datetime

        from huginn.anomaly_log import Anomaly

        out: list[Anomaly] = []
        formula = str(result_data.get("formula") or args.get("formula") or "")
        lattice = result_data.get("lattice_params") or {}
        if not formula or not lattice:
            return out

        material = self._match_material(formula)
        if material is None:
            return out
        standard = _STANDARD_LATTICE[material]
        cn = _MATERIAL_CN.get(material, material)

        for param in ("a", "b", "c"):
            val = lattice.get(param)
            std = standard.get(param)
            if val is None or std is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            if std <= 0:
                continue
            dev = abs(val_f - std) / std
            if dev > _DEVIATION_THRESHOLD:
                out.append(Anomaly(
                    id="",
                    ts=datetime.now(),
                    category="INPUT_DATA",
                    severity="HIGH",
                    description=(
                        f"{cn}({material}) 晶格常数 {param}={val_f:.3f}Å "
                        f"与标准 {std:.2f}Å 偏差 {dev*100:.1f}%"
                    ),
                    detection_method="standard_value_compare",
                    source="tool_output",
                    context_snapshot={
                        "tool": "structure_tool",
                        "formula": formula,
                        "material": material,
                        "param": param,
                        "value": val_f,
                        "standard": std,
                        "deviation": dev,
                        "lattice_params": lattice,
                    },
                    unresolved_dimensions=[
                        f"为什么 {param}={val_f:.3f}Å 而非标准 {std:.2f}Å？"
                        "单位混淆(Å vs nm)? 文件读错? 结构本身就有问题?"
                        " 或者是新颖结构/相——需独立计算验证"
                    ],
                ))
        return out

    # ---- 工具方法 ----

    @staticmethod
    def _match_material(formula: str) -> str | None:
        """把 formula (如 'Si2', 'Ga1 As1', 'Ti1 O2') 约化后匹配标准材料."""
        counts = _parse_formula(formula)
        if not counts:
            return None
        return _STANDARD_LATTICE_LOOKUP.get(_canonical_key(counts))

    @staticmethod
    def _stringify(obj: Any) -> str:
        try:
            import json
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    @staticmethod
    def _truncate(text: str, n: int) -> str:
        if not text:
            return ""
        return text if len(text) <= n else text[:n] + "…"

    @staticmethod
    def _shrink(args: dict) -> dict:
        """args 快照, 砍掉超长字段避免 context_snapshot 爆炸."""
        out: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 200:
                out[k] = v[:200] + "…"
            else:
                out[k] = v
        return out
