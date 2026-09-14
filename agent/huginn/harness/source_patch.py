"""A3: Direct self-source modification — 运行时函数级自改源码（v0, 默认关）.

前两层 (M-R1 improv 模板 / A1 strategist) 只在 "prompt 模板" 空间演化, 从未触碰真实
代码. 本模块补 RSI 第三道质变缺口: 让 agent 能**改自己的实现**, 而不只改提示.

安全策略 (重要):
- v0 只做**运行时函数级自改**: apply 把候选新函数体 monkeypatch 进目标模块的
  ``__dict__`` (运行时代码对象替换), 绝不对磁盘 .py 落盘. 全程可逆 (RevertibleContext
  的 disposer 栈 / 注册补偿器), 可验证 (先 ``compile`` + 隔离 exec 再应用), 不可能
  破坏仓库 (不写盘, 重启即弃 -> 会话级).
- 所有自改经 toggle ``harness_source_patch`` 全局门 (默认 off), 关闭时零行为变更.

验证三步 (verify_source_patch):
  ① compile(new_code) 合法, 否则 SyntaxError 拒;
  ② 目标模块可 import 且含待换符号 (anchor 存在), 否则拒;
  ③ 隔离 ns 里 exec 定义无异常 (ImportError/NameError/def-time error) 且产出该符号.

应用 (apply_source_patch):
  - 先 verify, 失败绝不 apply;
  - 记原对象到模块级 ``_ORIGINALS`` (供进程内/补偿器可逆回复);
  - ``exec(new_code, module.__dict__)`` 原位换入新实现;
  - ctx(track) 或 ctx.compensate + 注册补偿器 均可逆回滚.

honest boundary: 会话级、内存级; 跨 session 不保留 (好事, 不污染仓库).
磁盘文件级自改写留 v1.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

# 每成功生成多少 source 候选触发一次 maybe_propose_source (由 note_generation 驱动).
_SOURCE_EVERY_N_GENERATIONS = 7
# LRU 上限 — store 抗无限增长.
_SOURCE_STORE_MAX = 20


@dataclass
class SourcePatch:
    """一个运行时函数级源码补丁.

    ``module``: 目标模块全名 (如 "huginn.harness.prompt_patch").
    ``symbol``: 待替换符号名 (函数名或顶层赋值名).
    ``new_code``: 新代码文本, 必须是一个 ``def <symbol>(...): ...`` 或
                 ``<symbol> = ...``, 能被 ``compile(..., "exec")`` 接受.
    """
    id: str
    module: str
    symbol: str
    new_code: str
    op: str = "replace_symbol"
    active: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourcePatch:
        return cls(
            id=d["id"],
            module=d.get("module", ""),
            symbol=d.get("symbol", ""),
            new_code=d.get("new_code", ""),
            op=d.get("op", "replace_symbol"),
            active=bool(d.get("active", False)),
            created_at=float(d.get("created_at", time.time())),
        )


class SourcePatchStore:
    """持久化 source patch store. 单例, 存 .huginn/source_patches/<id>.json."""

    _instance: SourcePatchStore | None = None

    def __init__(self) -> None:
        cache_dir = get_runtime_home()
        self._store_dir = cache_dir / "source_patches"
        with contextlib.suppress(Exception):
            self._store_dir.mkdir(parents=True, exist_ok=True)
        self._patches: dict[str, SourcePatch] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> SourcePatchStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        with contextlib.suppress(Exception):
            for f in self._store_dir.glob("*.json"):
                with contextlib.suppress(Exception):
                    p = SourcePatch.from_dict(json.loads(f.read_text(encoding="utf-8")))
                    self._patches[p.id] = p

    def _save(self, patch: SourcePatch) -> None:
        with contextlib.suppress(Exception):
            self._store_dir.mkdir(parents=True, exist_ok=True)
            (self._store_dir / f"{patch.id}.json").write_text(
                json.dumps(patch.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if len(self._patches) > _SOURCE_STORE_MAX:  # LRU 抗无限增长
            for k in list(self._patches)[: len(self._patches) - _SOURCE_STORE_MAX]:
                self._patches.pop(k, None)
                with contextlib.suppress(OSError):
                    (self._store_dir / f"{k}.json").unlink()

    def add_patch(self, patch: SourcePatch) -> None:
        self._patches[patch.id] = patch
        self._save(patch)

    def get(self, patch_id: str) -> SourcePatch | None:
        return self._patches.get(patch_id)

    def list_patches(self, module: str | None = None, symbol: str | None = None) -> list[SourcePatch]:
        out = list(self._patches.values())
        if module:
            out = [p for p in out if p.module == module]
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        return out

    def clear_active(self, module: str, symbol: str) -> None:
        """停用同 (module, symbol) 的其它 active 补丁 (同 slot 只一个生效)."""
        for p in self._patches.values():
            if p.module == module and p.symbol == symbol and p.active:
                p.active = False
                self._save(p)


# 模块级缓存: 被换前的原实现, 供进程内 revert / 补偿器回复 (会话级).
_ORIGINALS: dict[str, dict[str, Any]] = {}


def _slot(module: str, symbol: str) -> str:
    return f"{module}.{symbol}"


def verify_source_patch(patch: SourcePatch) -> dict[str, Any]:
    """应用前验证. 返回 {passed, error, issues}. 任一步失败 passed=False."""
    issues: list[str] = []
    # ① 语法
    try:
        compile(patch.new_code, f"<source_patch:{patch.symbol}>", "exec")
    except SyntaxError as exc:
        return {"passed": False, "error": f"syntax: {exc}", "issues": ["syntax"]}
    # ② 目标模块可 import + anchor 存在
    try:
        mod = importlib.import_module(patch.module)
    except Exception as exc:
        return {"passed": False, "error": f"module_missing: {exc}",
                "issues": ["module_missing"]}
    if not hasattr(mod, patch.symbol):
        return {"passed": False, "error": f"anchor_missing: {patch.module}.{patch.symbol}",
                "issues": ["anchor_missing"]}
    # ③ 隔离 ns 里 exec 定义无异常且产出该符号
    ns = {k: v for k, v in vars(mod).items() if not k.startswith("__")}
    try:
        exec(patch.new_code, ns)  # noqa: S102 有意执行候选代码以验证可定义
    except Exception as exc:
        return {"passed": False, "error": f"runtime_def: {type(exc).__name__}: {exc}",
                "issues": ["runtime_def"]}
    if patch.symbol not in ns:
        return {"passed": False, "error": f"not_defined: {patch.symbol}",
                "issues": ["not_defined"]}
    return {"passed": True, "error": None, "issues": []}


def revert_source_patch(payload: dict[str, Any]) -> None:
    """补偿器: 恢复 ``module.__dict__[symbol]`` 为原实现 (会话级)."""
    module = payload.get("module")
    symbol = payload.get("symbol")
    orig = _ORIGINALS.get(_slot(module, symbol))
    if orig is None:
        return
    with contextlib.suppress(Exception):
        m = importlib.import_module(module)
        m.__dict__[symbol] = orig


def _register_source_compensator() -> None:
    from huginn.security.revertible import register_compensator

    register_compensator("source_patch_apply", revert_source_patch)


def apply_source_patch(patch: SourcePatch, ctx: Any | None = None) -> bool:
    """验证通过后才应用: monkeypatch ``module.__dict__[symbol]`` (运行时, 可逆).

    - ``ctx`` 非 None → 用 ctx.track(disposer) 会话级回滚; 同时 compensate 注册补偿器.
    - 成功后清同 slot 其它 active, 标记本 patch active 并落盘.
    """
    v = verify_source_patch(patch)
    if not v["passed"]:
        return False
    with contextlib.suppress(Exception):
        mod = importlib.import_module(patch.module)
        orig = mod.__dict__.get(patch.symbol)
        _ORIGINALS[_slot(patch.module, patch.symbol)] = orig

        def _dispose() -> None:
            with contextlib.suppress(Exception):
                ob = _ORIGINALS.get(_slot(patch.module, patch.symbol))
                m = importlib.import_module(patch.module)
                if ob is None:
                    m.__dict__.pop(patch.symbol, None)
                else:
                    m.__dict__[patch.symbol] = ob

        exec(patch.new_code, mod.__dict__)  # noqa: S102 运行时代码对象替换
        if ctx is not None:
            with contextlib.suppress(Exception):
                ctx.track(_dispose)
                ctx.compensate("source_patch_apply", {
                    "module": patch.module, "symbol": patch.symbol,
                })
        store = SourcePatchStore.get_instance()
        store.clear_active(patch.module, patch.symbol)
        patch.active = True
        store.add_patch(patch)
        return True
    return False


def _selfcheck() -> None:
    """A3 selfcheck: 验证门(语法/缺anchor) + 应用 + 可逆回复."""
    import shutil
    import tempfile

    import huginn.harness.source_patch as sp

    tmp = tempfile.mkdtemp()
    import os as _os
    _os.environ["HUGINN_CACHE_DIR"] = tmp
    sp.SourcePatchStore._instance = None

    # 1. 非法语法 → 拒
    bad = sp.SourcePatch(id="b1", module="huginn.harness.prompt_patch",
                         symbol="DEFAULT_IMPROV_TEMPLATE", new_code="def x(")
    assert sp.verify_source_patch(bad)["passed"] is False
    print("1. syntax reject OK")

    # 2. 缺 anchor → 拒
    no_anchor = sp.SourcePatch(id="b2", module="huginn.harness.prompt_patch",
                               symbol="NO_SUCH_SYMBOL_XYZ", new_code="NO_SUCH_SYMBOL_XYZ = 1")
    assert sp.verify_source_patch(no_anchor)["passed"] is False
    print("2. anchor-missing reject OK")

    # 3. 应用 (换一个函数/常量实现) + 变行为
    import huginn.security.revertible as rev

    rev_tmp = tempfile.mkdtemp()
    sp_patch = sp.SourcePatch(
        id="g1", module="huginn.harness.meta_improver",
        symbol="_PROPOSE_EVERY_N",  # 已存在常量锚
        new_code="_PROPOSE_EVERY_N = 12345",
    )
    ctx = rev.RevertibleContext(journal_path=rev_tmp + "/j.json")
    _register_source_compensator()
    ok = sp.apply_source_patch(sp_patch, ctx)
    assert ok is True, "合法源补丁应应用"

    import huginn.harness.meta_improver as mi_mod
    assert mi_mod._PROPOSE_EVERY_N == 12345, "运行时实现应被换入"
    print("3. apply runtime self-mod OK")

    # 4. 可逆回复
    ctx.revert_all()
    # 原值不是 12345, 回复后应还原
    assert mi_mod._PROPOSE_EVERY_N != 12345, "revert_all 应恢复原实现"
    print("4. revert_all restore OK")

    # 5. 持久化
    assert sp.SourcePatchStore.get_instance().get("g1") is not None, "patch 应落盘"
    print("5. store persistence OK")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nA3 source_patch selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()