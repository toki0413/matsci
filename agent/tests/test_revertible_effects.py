"""T-BCSE-14: RevertibleEffect 抽象测试 (Cordis 论文 revertible effects).

验证:
- 每个可逆效应返回逆 (disposer), 由 RevertibleContext 累积.
- revert_all 后进先出地恢复上下文 (时间可组合).
- transaction 异常自动回滚, 正常退出保留.
- 复合逆按相反顺序执行 (扭结算子).
- SandboxExecutor 集成: create_file/set_env/run_revertible.
"""

from __future__ import annotations

import asyncio
import os
import sys
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


# ── 工具执行路径集成 (code_tool / bash_tool 失败自动回滚) ─────────
def test_code_tool_failure_rolls_back_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """code_tool 执行失败 → 本次新建的文件自动回滚 (时间可组合).

    open()/pathlib 被 restricted_python 拦截, 用 numpy.save 创建文件 (允许),
    然后 raise 让脚本失败. 回滚后 partial.npy 应被删除.
    """
    monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
    # 让子进程解析到当前解释器 (.venv, 带 numpy), 否则 code_tool 用全局
    # python 缺 numpy 而无法执行到创建文件那一步.
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    from huginn.tools.code_tool import CodeTool

    tool = CodeTool()
    code = (
        "import numpy as np\n"
        "np.save('partial.npy', [1, 2, 3])\n"
        "raise ValueError('boom')\n"
    )
    res = tool.call({"code": code, "working_dir": str(tmp_path), "timeout": 30})
    assert not res.success
    assert not (tmp_path / "partial.npy").exists(), "失败应回滚新建文件"


def test_bash_tool_failure_rolls_back_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bash_tool 命令失败 → 本次在 work_dir 新建的文件自动回滚."""
    monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
    from huginn.tools.bash_tool import BashTool

    tool = BashTool()
    cmd = [
        sys.executable,
        "-c",
        "open('created.txt', 'w').write('x'); import sys; sys.exit(3)",
    ]
    res = asyncio.run(
        tool.call(
            {"command": cmd, "working_dir": str(tmp_path), "timeout": 30}
        )
    )
    assert not res.success
    assert not (tmp_path / "created.txt").exists(), "命令失败应回滚新建文件"


# ── 技能生态集成 (技能多步组合失败 → 原子回滚) ─────────────────
def test_skill_execution_failure_rolls_back_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """技能是原子事务: 多步组合整体失败 (abort) → 回滚本次技能新建的文件.

    验证 skill_tool 的 workspace 透传 + 快照回滚: 第一步成功产物文件,
    第二步失败, 整体失败后第一步的产物也被回滚 (时间可组合).
    """
    monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    from huginn.skills.base import DeclarativeSkillExecutor, SkillDefinition, SkillStep
    from huginn.skills.registry import SkillRegistry
    from huginn.tools.skill_tool import SkillTool, SkillToolInput
    from huginn.tools.registry import ToolRegistry
    from huginn.core_types import ToolContext

    # 技能: step1 用 code_tool 建文件, step2 必失败 (abort).
    skill = SkillDefinition(
        name="test_atomic_skill",
        description="atomic rollback test",
        category="test",
        parameters=[],
        steps=[
            SkillStep(
                name="make_file",
                tool="code_tool",
                input_mapping={
                    "code": (
                        "import numpy as np\n"
                        "np.save('step1_out.npy', [1])\n"
                        "'ok'\n"
                    ),
                    "working_dir": "$workspace",
                    "timeout": "30",
                },
                output_key="step1",
                on_failure="abort",
            ),
            SkillStep(
                name="fail_step",
                tool="code_tool",
                input_mapping={
                    "code": "raise RuntimeError('boom')\n",
                    "working_dir": "$workspace",
                    "timeout": "30",
                },
                output_key="step2",
                on_failure="abort",
            ),
        ],
    )
    tool = SkillTool(DeclarativeSkillExecutor(ToolRegistry))
    SkillRegistry.register(skill)
    ctx = ToolContext(session_id="test", workspace=str(tmp_path))
    inp = SkillToolInput(action="execute", skill_name=skill.name, parameters={})
    res = asyncio.run(tool.call(inp, ctx))
    assert not res.success
    assert not (tmp_path / "step1_out.npy").exists(), "技能整体失败应回滚第一步产物"
