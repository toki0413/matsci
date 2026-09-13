"""action_fusion —— Huginn 版 SoL-Pi「Action Fusion」（落地①号）。

缺口(见对比清单): Huginn 没有"一个变更动作 + 其紧跟的验证命令"的**契约注册表**,
每轮"改完→单独发一条验证工具"造成重复模型往返。SoL-Pi 把它融合进同一个调用。

本模块补的是机制①的关键接线点: **tool → verifier 契约映射 + 同调用融合执行**。
  - ``register_verifier(tool_name, contract)``: 为某变更工具注册"跟随验证"契约
    (支持精确名或前缀, 与 ``register_tool_keep_keys`` 同套路, opt-in 不污全局)。
  - ``verifier_for(tool_name)``: 查契约。
  - ``fused_verify(tool_name, mutation_result, ctx)``: 有契约时, 在**同一调用**
    里执行 verifier 工具并把结果折叠进变更结果(无契约返回 None, 不改变原结果)。

设计约束: 纯本地注册 + 复用 ToolRegistry 分派执行, 不新增框架; verifier 执行
失败返回 verified=False 而非 raise(不吞变更结果, 只是标记验证失败)。任何契约
缺失/执行异常 → 静默 None, 完全不放宽/不阻塞原工具。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerifierContract:
    """一份"变更后跟随验证"契约.

    verifier_tool: 执行的验证工具名(必须在 ToolRegistry 里注册)。
    args: 传给验证工具的静态参数, 支持 ``$result.<path>`` 占位符——从本次
          变更结果里取值(如把结果里的文件路径/参数带给验证工具)。
    """

    verifier_tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def build_args(self, mutation_result: Any) -> dict[str, Any]:
        """把 args 里的 ``$result.<dotpath>`` 占位符替换成本次变更结果里的值."""
        if not mutation_result:
            return dict(self.args)
        out: dict[str, Any] = {}
        for k, v in self.args.items():
            if isinstance(v, str) and v.startswith("$result."):
                out[k] = _lookup_dotpath(mutation_result, v[len("$result."):])
            else:
                out[k] = v
        return out


def _lookup_dotpath(obj: Any, dotpath: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotpath.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, default)
        if cur is None:
            return default
    return cur


class _VerifierRegistry:
    """tool_name(精确或前缀) → 优先级排序的 Verifier 列表."""

    def __init__(self) -> None:
        self._items: dict[str, list[tuple[int, VerifierContract]]] = {}

    def register(
        self, tool_name: str, contract: VerifierContract, priority: int = 0
    ) -> None:
        self._items.setdefault(tool_name, []).append((priority, contract))
        self._items[tool_name].sort(key=lambda t: -t[0])  # 高优先级在前

    def resolve(self, tool_name: str) -> VerifierContract | None:
        if tool_name in self._items:
            for _p, c in self._items[tool_name]:
                return c
        # 前缀匹配: "vasp." 覆盖 vasp_run / vasp_parse (去尾部点号后按名字前缀匹配)
        for prefix, entries in self._items.items():
            if prefix.endswith(".") and tool_name.startswith(prefix[:-1]):
                for _p, c in entries:
                    return c
        return None


_VERIFIERS = _VerifierRegistry()


def register_verifier(
    tool_name: str,
    verifier_tool: str,
    args: dict[str, Any] | None = None,
    priority: int = 0,
) -> None:
    """为变更工具注册"跟随验证"契约(精确名或前缀). 用法::

        register_verifier("file_edit_tool", "bash_tool",
                          {"command": "python -m py_compile $result.path"})
    """
    _VERIFIERS.register(
        tool_name,
        VerifierContract(verifier_tool=verifier_tool, args=args or {}),
        priority=priority,
    )


def verifier_for(tool_name: str) -> VerifierContract | None:
    return _VERIFIERS.resolve(tool_name)


async def fused_verify(
    tool_name: str,
    mutation_result: Any,
    ctx: Any,
) -> dict[str, Any] | None:
    """有契约时, 在**同一调用**里执行跟随验证并折叠结果.

    返回 dict 或 None:
      - None  = 无契约 / 契约工具缺失 / 执行异常 (静默, 不变更原结果)
      - dict  = {"verifier_tool", "verified": bool, "detail": <验证结果>, "fused": True}

    不 raise: 验证失败不算致命, 交给上层决定是否保留变更。
    """
    contract = _VERIFIERS.resolve(tool_name)
    if contract is None:
        return None
    from huginn.tools.registry import ToolRegistry

    vtool = ToolRegistry.get(contract.verifier_tool)
    if vtool is None:
        logger.debug("action_fusion: verifier tool %r absent", contract.verifier_tool)
        return None
    try:
        vargs = contract.build_args(mutation_result)
        if not vargs:
            vargs = {}
        result = await vtool.call(vargs, ctx)
    except Exception as exc:
        logger.debug("action_fusion verifier execution failed: %s", exc)
        return None
    success = bool(getattr(result, "success", False))
    return {
        "verifier_tool": contract.verifier_tool,
        "verified": success,
        "detail": getattr(result, "data", None) or getattr(result, "error", None),
        "fused": True,
    }


def _selfcheck() -> None:
    import asyncio
    import types

    from huginn.core_types import ToolContext, ToolResult
    from huginn.tools.registry import ToolRegistry

    async def call(self, args, ctx):
        return ToolResult(success=True, data={"validated": args})

    fake_validate = types.SimpleNamespace()
    fake_validate.name = "fake_validate"
    fake_validate.call = types.MethodType(call, fake_validate)

    snap = ToolRegistry.snapshot()
    try:
        ToolRegistry.register(fake_validate)

        # 1. 精确名契约 + $result.path 替换
        register_verifier("file_edit_tool", "fake_validate", {"path": "$result.path"})
        ctx = ToolContext(session_id="selftest", workspace="/tmp", config=None)
        r = asyncio.run(fused_verify("file_edit_tool", {"path": "/tmp/m.py"}, ctx))
        assert r is not None and r["verified"] is True, r
        assert r["detail"]["validated"]["path"] == "/tmp/m.py", r

        # 2. 前缀匹配: "vasp." 覆盖 vasp_run / vasp_parse
        register_verifier("vasp.", "fake_validate", {"x": 1})
        assert verifier_for("vasp_run") is not None
        assert verifier_for("vasp_parse") is not None

        # 3. 未注册 → None
        assert verifier_for("ghost_tool") is None
        assert asyncio.run(fused_verify("ghost_tool", {}, ctx)) is None

        # 4. 契约工具缺失 → None 不 raise
        register_verifier("only_reg", "missing_verifier_tool", {})
        assert asyncio.run(fused_verify("only_reg", {}, ctx)) is None
        print("OK action_fusion self-check passed (tool->verifier contract + fused verify)")
    finally:
        ToolRegistry.restore(snap)
        _VERIFIERS._items.clear()


if __name__ == "__main__":
    _selfcheck()