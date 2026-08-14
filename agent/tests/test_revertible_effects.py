"""T-BCSE-14: RevertibleEffect 抽象测试 (Cordis 论文 revertible effects).

验证:
- 每个可逆效应返回逆 (disposer), 由 RevertibleContext 累积.
- revert_all 后进先出地恢复上下文 (时间可组合).
- transaction 异常自动回滚, 正常退出保留.
- 复合逆按相反顺序执行 (扭结算子).
- SandboxExecutor 集成: create_file/set_env/run_revertible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.security.revertible import RevertibleContext
from huginn.security.sandbox import SandboxConfig, SandboxExecutor


def test_create_file_restores_missing_file(tmp_path: Path) -> None:
    ctx = RevertibleContext()
    p = tmp_path / "a.txt"
    ctx.create_file(p, "hello")
    assert p.read_text() == "hello"
    assert ctx.depth == 1

    ctx.revert_all()
    assert not p.exists(), "revert_all 应删除新建文件"
    assert ctx.depth == 0


def test_create_file_over_existing_restores_content(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("original")
    ctx = RevertibleContext()
    ctx.create_file(p, "overwritten")
    assert p.read_text() == "overwritten"

    ctx.revert_all()
    assert p.read_text() == "original", "revert 应恢复原内容"


def test_set_env_restores_old_value() -> None:
    env = {"K": "old"}
    ctx = RevertibleContext()
    ctx.set_env("K", "new", env)
    ctx.set_env("MISSING", "x", env)
    assert env["K"] == "new"
    assert env["MISSING"] == "x"

    ctx.revert_all()
    assert env["K"] == "old"
    assert "MISSING" not in env, "原本不存在的 key 应被删除"


def test_revert_all_is_lifo() -> None:
    """后进先出: 后注册的逆先执行 (扭结算子顺序)."""
    ctx = RevertibleContext()
    order: list[str] = []
    ctx.track(lambda: order.append("a"))
    ctx.track(lambda: order.append("b"))
    ctx.revert_all()
    assert order == ["b", "a"], f"应 LIFO, got {order}"


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    ctx = RevertibleContext()
    p = tmp_path / "t.txt"
    with pytest.raises(RuntimeError), ctx.transaction():
        ctx.create_file(p, "x")
        raise RuntimeError("boom")
    # 块内效应应被回滚, 但 ctx 栈清空 (逆已消费)
    assert not p.exists()
    assert ctx.depth == 0


def test_transaction_keeps_effects_on_success(tmp_path: Path) -> None:
    ctx = RevertibleContext()
    p = tmp_path / "t.txt"
    with ctx.transaction():
        ctx.create_file(p, "kept")
    assert p.exists(), "正常退出应保留效应"
    assert ctx.depth == 1

    # 外层 revert 仍可整体撤销
    ctx.revert_all()
    assert not p.exists()


def test_composite_reverses_order() -> None:
    ctx = RevertibleContext()
    order: list[str] = []
    composite = ctx.composite(
        lambda: order.append("1"),
        lambda: order.append("2"),
        lambda: order.append("3"),
    )
    composite()
    assert order == ["3", "2", "1"], "复合逆按相反顺序执行"


def test_register_coeffect_restores_store() -> None:
    store: dict[str, str] = {"db": "old"}
    ctx = RevertibleContext()
    ctx.register("db", "new", store)
    ctx.register("only_tmp", "x", store)
    assert store == {"db": "new", "only_tmp": "x"}

    ctx.revert_all()
    assert store == {"db": "old"}, "co-effect 注册应在卸载时被撤销"


# ── SandboxExecutor 集成 ──────────────────────────────────────────
def test_sandbox_create_file_and_revert(tmp_path: Path) -> None:
    sb = SandboxExecutor()
    p = tmp_path / "s.txt"
    sb.revertible.create_file(p, "data")
    assert p.read_text() == "data"
    sb.revertible.revert_all()
    assert not p.exists()


def test_sandbox_transaction_rolls_back(tmp_path: Path) -> None:
    sb = SandboxExecutor()
    p = tmp_path / "x.bin"
    with pytest.raises(ValueError), sb.transaction():
        sb.create_file(p, "bin")
        raise ValueError("nope")
    assert not p.exists()


def test_sandbox_run_revertible_removes_new_files(tmp_path: Path) -> None:
    cfg = SandboxConfig(allowed_executables={"sh"})
    sb = SandboxExecutor(cfg)
    env = {"PATH": "/usr/bin:/bin"}
    result, dispose = sb.run_revertible(
        ["sh", "-c", "echo hi > out.txt"],
        cwd=tmp_path,
        env=env,
    )
    assert result.success
    assert (tmp_path / "out.txt").exists()

    dispose()
    assert not (tmp_path / "out.txt").exists(), "run_revertible 的 dispose 应删除新文件"


def test_sandbox_run_revertible_keeps_preexisting(tmp_path: Path) -> None:
    cfg = SandboxConfig(allowed_executables={"sh"})
    sb = SandboxExecutor(cfg)
    pre = tmp_path / "keep.txt"
    pre.write_text("keep")
    env = {"PATH": "/usr/bin:/bin"}
    result, dispose = sb.run_revertible(
        ["sh", "-c", "echo n > new.txt"],
        cwd=tmp_path,
        env=env,
    )
    assert result.success
    dispose()
    assert pre.exists(), "既有文件不应被 dispose 删除"
    assert not (tmp_path / "new.txt").exists()
