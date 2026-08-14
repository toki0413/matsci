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

import base64
import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 一个逆: 撤销单个效应. 无返回. 每个原子效应必须提供一个.
Disposer = Callable[[], None]


# ── 数据驱动逆 (journal 能力的基础) ───────────────────────────────
# 每个可逆操作被记录为可序列化的 op 描述 (而非仅闭包), 这样逆可落盘,
# 进程崩溃后能按 journal 重放 (把时间可组合性从运行时提升到跨崩溃).
OP_SET_ENV = "set_env"
OP_CREATE_FILE = "create_file"
OP_CREATE_DIR = "create_dir"
OP_REMOVE_FILE = "remove_file"
OP_SPAWN = "spawn"
OP_REGISTER = "register"
OP_COMPENSATE = "compensate"  # 出站写操作的补偿逆 (t1): 携带可序列化的补偿字典


class RevertibleContext:
    """跟踪逆的运行时上下文 (Γ∞ 类比).

    持有一个 LIFO disposer 栈 (累积器 φ). 每个可逆操作把逆压栈; ``revert_all``
    按后进先出顺序逐个执行, 把上下文恢复到本 context 建立时的状态.
    """

    def __init__(self, journal_path: str | Path | None = None) -> None:
        self._disposers: list[Disposer] = []
        self._ops: list[dict[str, Any]] = []
        self._journal_path = (
            Path(journal_path) if journal_path is not None else None
        )

    # ── 核心 ─────────────────────────────────────────────────────
    def track(self, dispose: Disposer | None) -> None:
        """手动注册一个逆. ``None`` 表示无副作用, 忽略.

        注意: 闭包逆无法落盘 journal (不可序列化). 需要跨崩溃恢复的
        副作用请用 :meth:`track_op` 的数据驱动形式.
        """
        if dispose is not None:
            self._disposers.append(dispose)

    def track_op(
        self,
        op: dict[str, Any] | None,
        *,
        env: dict[str, str] | None = None,
        store: dict[str, Any] | None = None,
    ) -> None:
        """注册一个**数据驱动**逆 (可序列化, 落盘 journal).

        同时生成一个绑定 ``env``/``store`` 的可执行闭包压栈 — 进程内
        ``revert_all`` 用它精确恢复; ``_ops`` 里的数据描述供崩溃后重放.
        """
        if op is None:
            return
        self._ops.append(op)
        self._disposers.append(
            lambda: _apply_inverse(op, env=env, store=store)
        )
        if self._journal_path is not None:
            self._flush_journal()

    def effects(self) -> list[dict[str, Any]]:
        """已累积的数据驱动逆 (journal 帧)."""
        return list(self._ops)

    def _flush_journal(self) -> None:
        """把当前 ops 序列化到 journal 文件 (追加式重写, 供崩溃后重放)."""
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_path.write_text(
                json.dumps(self._ops, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        except Exception:
            logger.warning(
                "revertible: journal flush failed to %s", self._journal_path,
                exc_info=True,
            )

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
        self.track_op(
            {"type": OP_SET_ENV, "key": key, "prev": prev, "absent": prev is None},
            env=env,
        )
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

        self.track_op(
            {
                "type": OP_CREATE_FILE,
                "path": str(path),
                "existed": existed,
                "prev_b64": base64.b64encode(prev).decode() if prev is not None else None,
            },
        )
        return path

    def create_dir(self, path: str | Path) -> Path:
        """创建目录 (含父目录). 逆: 若目录原本不存在则尝试删除 (仅当为空时)."""
        path = Path(path)
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)

        self.track_op(
            {"type": OP_CREATE_DIR, "path": str(path), "existed": existed},
        )
        return path

    def remove_file(self, path: str | Path) -> Path:
        """移除一个文件 (用于把文件移出工作区). 逆: 恢复原文件内容."""
        path = Path(path)
        data = path.read_bytes() if path.exists() else None
        path.unlink(missing_ok=True)

        self.track_op(
            {
                "type": OP_REMOVE_FILE,
                "path": str(path),
                "data_b64": base64.b64encode(data).decode() if data is not None else None,
            },
        )
        return path

    def spawn(self, proc: Any) -> Any:
        """登记一个后台进程. 逆: 若仍在运行则终止."""
        self.track_op({"type": OP_SPAWN, "pid": proc.pid})
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
        absent = key not in store
        store[key] = value
        if _json_safe(prev):
            self.track_op(
                {
                    "type": OP_REGISTER,
                    "key": key,
                    "prev": prev,
                    "absent": absent,
                },
                store=store,
            )
        else:
            # prev 不可序列化 — 无法 journal, 退化为闭包逆 (进程内仍精确恢复).
            self.track(
                lambda: (
                    store.pop(key, None)
                    if absent
                    else store.update({key: prev})
                )
            )
        return value

    def compensate(self, kind: str, payload: dict[str, Any]) -> None:
        """注册一个**出站写操作的补偿逆** (t1).

        ``kind`` 是补偿类型 (如 ``git_commit`` / ``git_checkout``), ``payload``
        携带撤销所需的可序列化状态 (如原 commit hash / 原分支). 进程内按
        :meth:`revertible.api` 的注册补偿器执行, 崩溃后经 journal 重放
        (跨崩溃恢复出站副作用).
        """
        self.track_op({"type": OP_COMPENSATE, "kind": kind, "payload": payload})

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


# ── 模块级辅助: journal 序列化 + 逆执行器 ─────────────────────────

def _json_safe(value: Any) -> bool:
    """能否被 JSON 序列化 (决定该逆能否落盘 journal)."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _json_default(value: Any) -> Any:
    """JSON 兜底: 类型名作为占位 (审计可见, 但不保留值)."""
    return {"__repr__": repr(value)}


def _apply_inverse(
    op: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    store: dict[str, Any] | None = None,
) -> None:
    """执行一个数据驱动逆. 进程内由闭包调用 (带 env/store 引用);
    崩溃后经 :func:`recover_from` 重放 (env/store 缺省, 只恢复可持久副作用).
    """
    op_type = op.get("type")
    try:
        if op_type == OP_SET_ENV:
            if env is None:
                return  # 进程已消失, env 无法重建 — 跳过
            key = op["key"]
            if op.get("absent"):
                env.pop(key, None)
            else:
                env[key] = op["prev"]
        elif op_type == OP_CREATE_FILE:
            p = Path(op["path"])
            if op.get("existed"):
                prev = base64.b64decode(op["prev_b64"]) if op.get("prev_b64") else b""
                p.write_bytes(prev)
            else:
                p.unlink(missing_ok=True)
        elif op_type == OP_CREATE_DIR:
            p = Path(op["path"])
            if not op.get("existed"):
                try:
                    p.rmdir()
                except OSError:
                    logger.debug("revertible: dir not empty, skip rmdir %s", p)
        elif op_type == OP_REMOVE_FILE:
            p = Path(op["path"])
            data = base64.b64decode(op["data_b64"]) if op.get("data_b64") else None
            if data is not None:
                p.write_bytes(data)
        elif op_type == OP_SPAWN:
            _kill_pid(op["pid"])
        elif op_type == OP_REGISTER:
            if store is None:
                return  # 进程内 store 已消失 — 跳过
            if op.get("absent"):
                store.pop(op["key"], None)
            else:
                store[op["key"]] = op["prev"]
        elif op_type == OP_COMPENSATE:
            _run_compensation(op.get("kind"), op.get("payload") or {})
        else:
            logger.warning("revertible: unknown inverse op type %r", op_type)
    except Exception:
        logger.warning("revertible: inverse failed for op %s", op_type, exc_info=True)


def _kill_pid(pid: int) -> None:
    """终止进程 (尽力而为, 跨崩溃重放用)."""
    try:
        os.kill(pid, 9)  # SIGKILL
    except (ProcessLookupError, PermissionError, OSError):
        pass  # 进程已退出或无权 — 视为已完成


# ── 补偿器注册 (出站写操作的逆) ───────────────────────────────────
# 每种出站写操作注册一个 kind → 补偿函数, 供进程内 revert_all 与崩溃后
# journal 重放共用. 补偿函数签名: ``fn(payload: dict) -> None``.
_COMPENSATORS: dict[str, Callable[[dict[str, Any]], None]] = {}


def register_compensator(kind: str, fn: Callable[[dict[str, Any]], None]) -> None:
    """注册一个出站写操作的补偿器 (t1). 重复 kind 覆盖."""
    _COMPENSATORS[kind] = fn


def _run_compensation(kind: str | None, payload: dict[str, Any]) -> None:
    if not kind:
        return
    fn = _COMPENSATORS.get(kind)
    if fn is None:
        logger.warning("revertible: no compensator registered for kind %r", kind)
        return
    try:
        fn(payload)
    except Exception:
        logger.warning("revertible: compensation %r failed", kind, exc_info=True)


# ── journal 读取 / 崩溃后重放 ─────────────────────────────────────
def recover_from(journal_path: str | Path) -> int:
    """从 journal 文件重放所有逆 (崩溃后调用).

    LIFO 顺序恢复 (与运行时一致). 只恢复可持久副作用 (文件系统/spawn/
    已注册的出站补偿); env/register 等进程内状态随崩溃天然消失, 跳过.
    返回成功执行的逆数量.
    """
    path = Path(journal_path)
    if not path.exists():
        return 0
    try:
        ops = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("revertible: journal corrupted, cannot recover %s", path)
        return 0
    applied = 0
    for op in reversed(ops):
        _apply_inverse(op)
        applied += 1
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    return applied
