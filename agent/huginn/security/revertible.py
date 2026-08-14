"""Revertible effects for the Huginn sandbox (Cordis paper: spatiotemporal composability).

时间可组合的核心主张: 每个副作用都应携带**逆** (disposer), 运行时跟踪逆,
复合效应的逆由复合自动派生 (LIFO / twisted composition), 组件移除时完整恢复
——正确性从"开发者纪律"变成"结构保证".

对照 Cordis 论文 (A Programming Paradigm for Spatiotemporal Composability):
- ``RevertibleContext`` 是上下文类型 ``Γ∞`` 的运行时类比 — 持有累积器
  (disposer 栈), 所有对共享环境的变更都经它, 卸载即恢复 (revert_all).
- ``effect()`` 是 ``ctx.effect`` 的同步版: 执行一个可逆操作并把其逆累积进栈.
- 具体可逆沙箱效应 (set_env / create_file / create_dir / remove_file /
  spawn / register) 每个都返回逆, 交给 ``RevertibleContext`` 自动累积.
- ``transaction()`` 是事务边界: ``with`` 块内注册的效应在异常时自动全量回滚,
  正常退出则保留 (disposers 交给外围 scope) — 对应论文的 withha/commit 语义.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 一个逆: 撤销单个效应. 无返回. 每个原子效应必须提供一个.
Disposer = Callable[[], None]


class RevertibleContext:
    """跟踪逆的运行时上下文 (Γ∞ 类比).

    持有一个 LIFO disposer 栈 (累积器 φ). 每个可逆操作把逆压栈; ``revert_all``
    按后进先出顺序逐个执行, 把上下文恢复到本 context 建立时的状态.
    """

    def __init__(self) -> None:
        self._disposers: list[Disposer] = []

    # ── 核心 ─────────────────────────────────────────────────────
    def track(self, dispose: Disposer | None) -> None:
        """手动注册一个逆. ``None`` 表示无副作用, 忽略."""
        if dispose is not None:
            self._disposers.append(dispose)

    @property
    def depth(self) -> int:
        """当前累积的逆的数量 (调试/测试用)."""
        return len(self._disposers)

    def effect(self, op: Callable[[RevertibleContext], tuple[Any, Disposer | None]]) -> Any:
        """执行一个可逆操作并累计其逆.

        ``op(ctx)`` 返回 ``(value, dispose)``; ``dispose`` 为操作产生的逆或多图空.
        """
        value, dispose = op(self)
        self.track(dispose)
        return value

    def revert_all(self) -> None:
        """后进先出地执行所有已累积的逆, 恢复到本 context 建立时的状态.

        单个逆失败不中断其余逆 (best-effort, 逐个 try), 避免一个坏逆让
        后续资源泄漏.
        """
        while self._disposers:
            dispose = self._disposers.pop()
            try:
                dispose()
            except Exception:
                logger.warning("revertible dispose failed", exc_info=True)

    # ── 事务边界 ─────────────────────────────────────────────────
    @contextmanager
    def transaction(self) -> Iterator[RevertibleContext]:
        """事务边界.

        - ``with`` 块正常退出: 块内累积的逆保留, 交给外围 scope.
        - ``with`` 块抛异常: 块内累积的逆全量回滚 (LIFO), 然后重新抛出.
        """
        start = self.depth
        try:
            yield self
        except BaseException:
            # 回滚本 scope 内新增的逆, 不碰外围 (外围由更外层 scope 负责).
            while self.depth > start:
                dispose = self._disposers.pop()
                try:
                    dispose()
                except Exception:
                    logger.warning("revertible rollback failed", exc_info=True)
            raise

    # ── 具体可逆效应 (每个返回逆, 由本 context 自动累积) ───────────
    def set_env(self, key: str, value: str, env: dict[str, str]) -> str | None:
        """设置环境变量, 返回旧值. 逆: 恢复旧值 (不存在则删除)."""
        prev = env.get(key)
        env[key] = value

        def dispose() -> None:
            if prev is None:
                env.pop(key, None)
            else:
                env[key] = prev

        self.track(dispose)
        return prev

    def create_file(
        self,
        path: str | Path,
        content: bytes | str | None = None,
        *,
        encoding: str | None = None,
    ) -> Path:
        """创建/覆盖文件. 逆: 若文件原不存在则删除; 若原存在则恢复原内容."""
        path = Path(path)
        existed = path.exists()
        prev = path.read_bytes() if existed else None
        if content is None:
            path.touch()
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding=encoding or "utf-8")

        def dispose() -> None:
            if existed:
                path.write_bytes(prev or b"")
            else:
                path.unlink(missing_ok=True)

        self.track(dispose)
        return path

    def create_dir(self, path: str | Path) -> Path:
        """创建目录 (含父目录). 逆: 若目录原本不存在则尝试删除 (仅当为空时)."""
        path = Path(path)
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)

        def dispose() -> None:
            if not existed:
                try:
                    path.rmdir()
                except OSError:
                    # 目录非空 (被本效应之外的内容占用) — 不删, 记录即可.
                    logger.debug("revertible: dir not empty, skip rmdir %s", path)

        self.track(dispose)
        return path

    def remove_file(self, path: str | Path) -> Path:
        """移除一个文件 (用于把文件移出工作区). 逆: 恢复原文件内容."""
        path = Path(path)
        data = path.read_bytes() if path.exists() else None
        path.unlink(missing_ok=True)

        def dispose() -> None:
            if data is not None:
                path.write_bytes(data)

        self.track(dispose)
        return path

    def spawn(self, proc: Any) -> Any:
        """登记一个后台进程. 逆: 若仍在运行则终止."""
        def dispose() -> None:
            if proc.poll() is None:
                proc.kill()

        self.track(dispose)
        return proc

    def register(
        self,
        key: str,
        value: Any,
        store: dict[str, Any],
    ) -> Any:
        """co-effect 类比: 在 ``store`` 里绑定 ``key -> value``. 逆: 恢复旧值.

        对应论文的 ``set(k, v)`` (reactive coeffects): 依赖注册是可逆效应,
        卸载组件时自动撤销注册.
        """
        prev = store.get(key)
        store[key] = value

        def dispose() -> None:
            if prev is None:
                store.pop(key, None)
            else:
                store[key] = prev

        self.track(dispose)
        return value

    # ── 复合逆 (扭结算子 / twisted composition) ───────────────────
    @staticmethod
    def composite(*disposers: Disposer | None) -> Disposer:
        """把多个逆复合为一个逆, 按**相反顺序**执行 (扭结算子).

        对应论文的扭结复合 ``(f1,g1)∘(f2,g2) = (f1∘f2, g2∘g1)`` — 逆向按
        应用顺序的逆序累积, 保证 LIFO 恢复.
        """

        def dispose() -> None:
            for d in reversed(disposers):
                if d is not None:
                    try:
                        d()
                    except Exception:
                        logger.warning("revertible composite dispose failed", exc_info=True)

        return dispose
