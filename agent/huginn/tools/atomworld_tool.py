"""AtomWorld benchmark tool — verifiable 3D spatial reasoning.

AtomWorld (arXiv:2510.04704, NUS + MasterAI-EAM, ICML 2026) 是纯 CIF
benchmark: 输入 CIF + action, 输出 CIF, 用 StructureMatcher + RMSD 评估.
论文实证 Claude Opus 4.6 在 rotation 类操作成功率 < 12%, 这是 text-centric
VLM 丢失 3D 信息的直接证据.

接入点: CodeAct 沙箱 + RCBench runner. env flag HUGINN_USE_ATOMWORLD=1 控制.
默认 off, 行为完全不变 (跟 BranchIncubator gating 风格一致).

ponytail: optional 依赖, 包未安装时降级 log warning 不 raise (跟 ml_potential_tool 一致).
升级路径: 接入 RCBench 做 RL reward (P0 只 eval 不训练).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import atomworld as _atomworld
except ImportError:
    _atomworld = None


@dataclass
class EvaluateResult:
    """evaluate() 返回值 — 跟 atomworld API 字段一一对齐.

    wrong_type 取值: None / "OutputFormatError" / "CIFParsingError" /
    "AtomCountMismatch" / "StructureMismatch" (来自 README).
    """

    correct: bool
    wrong_type: str | None = None
    rmsd: float | None = None
    max_dist: float | None = None


def is_available() -> bool:
    """atomworld 包是否已安装."""
    return _atomworld is not None


def evaluate(target_cif: str, generated_output: str) -> EvaluateResult:
    """验证 generated_output CIF 是否匹配 target_cif.

    包装 atomworld.evaluate(). 包未安装时 raise RuntimeError.
    签名按 README: keyword args target_cif=, generated_output=.
    """
    if _atomworld is None:
        raise RuntimeError(
            "atomworld not installed, pip install -e \".[benchmark]\""
        )
    raw = _atomworld.evaluate(
        target_cif=target_cif,
        generated_output=generated_output,
    )
    return _coerce_evaluate_result(raw)


def _coerce_evaluate_result(raw: Any) -> EvaluateResult:
    """把 atomworld 原始返回值转成 EvaluateResult (defensive).

    README 显示返回的是带 .correct/.wrong_type/.rmsd/.max_dist 属性的对象,
    但也兼容 dict / namedtuple 以防版本变动.
    """
    if isinstance(raw, EvaluateResult):
        return raw
    if isinstance(raw, dict):
        return EvaluateResult(
            correct=raw.get("correct", False),
            wrong_type=raw.get("wrong_type"),
            rmsd=raw.get("rmsd"),
            max_dist=raw.get("max_dist"),
        )
    return EvaluateResult(
        correct=getattr(raw, "correct", False),
        wrong_type=getattr(raw, "wrong_type", None),
        rmsd=getattr(raw, "rmsd", None),
        max_dist=getattr(raw, "max_dist", None),
    )


# AtomWorld 全集 (15 actions, 来自 README). 带 _action 后缀跟 atomworld API 一致.
# ponytail: 名字直接跟 API 对齐, 不做 short-name 转换, 少一层映射.
_ACTION_NAMES = [
    "add_atom_action",
    "change_atom_action",
    "remove_atom_action",
    "move_atom_action",
    "move_towards_atom_action",
    "move_around_atom_action",
    "insert_between_atoms_action",
    "swap_atoms_action",
    "delete_below_atom_action",
    "delete_around_atom_action",
    "super_cell_action",
    "rotate_around_atom_action",
    "rotate_whole_action",
    "move_all_action",
    "move_selected_atoms_action",
]


def list_actions() -> list[str]:
    """返回支持的 action 名 (跟 atomworld CLI 的 -a 参数一致)."""
    return list(_ACTION_NAMES)


def apply_action(input_cif: str, action_name: str, **params: Any) -> str:
    """对 input_cif 应用 action, 返回新 CIF.

    包未安装时 raise RuntimeError.
    action_name 不在 list 时 raise ValueError.
    params 透传给 ActionClass (具体签名在真包安装后验证).
    """
    if _atomworld is None:
        raise RuntimeError(
            "atomworld not installed, pip install -e \".[benchmark]\""
        )
    if action_name not in _ACTION_NAMES:
        raise ValueError(
            f"unknown action {action_name!r}, supported: {_ACTION_NAMES}"
        )
    return _dispatch_apply(input_cif, action_name, params)


def _resolve_actions_module():
    """找 atomworld.actions 模块.

    README 写的是 `from atom_world.actions import ...`, 但包名叫 atomworld.
    两个都试, 真包装上后哪个 work 用哪个.
    ponytail: 不写 if-else 链, getattr 试两层就完事.
    """
    mod = getattr(_atomworld, "actions", None)
    if mod is None:
        try:
            import atom_world.actions as _aw  # type: ignore
            mod = _aw
        except ImportError:
            pass
    return mod


def _dispatch_apply(input_cif: str, action_name: str, params: dict) -> str:
    """从 actions 模块找对应 ActionClass 并 apply.

    README 只展示了 `AddAtomAction.apply_random(atoms, rng=rng) -> (prompt, result)`
    这种 classmethod 风格 (返回 prompt 字符串 + result atoms). 确定性 apply 的
    具体签名 (apply / __call__ / 构造器+apply) 没在 README 写明, 这里走 defensive
    fallback 链. 真包安装后 (Task 5) 跑一次 selfcheck 验证哪条路径生效.

    ponytail: 用 getattr + 短 fallback 链, 不写 15 个 if-elif.
    升级路径: atomworld 升级时如果 action 类名 / apply 签名变, 只改这一个函数.
    """
    actions_mod = _resolve_actions_module()
    if actions_mod is None:
        raise RuntimeError(
            "atomworld.actions module not found (README says atom_world.actions, "
            "package is atomworld — install a newer build)"
        )

    # action_name → ClassName: "add_atom_action" → "AddAtomAction"
    cls_name = "".join(p.capitalize() for p in action_name.split("_"))
    cls = getattr(actions_mod, cls_name, None)
    if cls is None:
        raise RuntimeError(f"atomworld.actions.{cls_name} not found")

    atoms = _cif_to_atoms(input_cif)

    # fallback 链: apply(atoms, **params) → cls(**params).apply(atoms) → apply_random
    # 第三条只在没有 params 时用 (随机生成测试样本).
    apply_cls = getattr(cls, "apply", None)
    if callable(apply_cls):
        new_atoms = apply_cls(atoms, **params)
        return _atoms_to_cif(_unwrap_result(new_atoms))

    try:
        instance = cls(**params)
    except TypeError:
        instance = None

    if instance is not None:
        apply_fn = getattr(instance, "apply", None)
        if callable(apply_fn):
            new_atoms = apply_fn(atoms)
            return _atoms_to_cif(_unwrap_result(new_atoms))

    if not params and hasattr(cls, "apply_random"):
        import numpy as np
        rng = np.random.default_rng()
        _prompt, new_atoms = cls.apply_random(atoms, rng=rng)
        return _atoms_to_cif(_unwrap_result(new_atoms))

    raise RuntimeError(
        f"could not apply {cls_name}: no apply/apply_random path matched. "
        f"params={params!r}"
    )


def _unwrap_result(result: Any):
    """apply 可能返回 atoms 或 (prompt, atoms) 元组 (apply_random 风格)."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    return result


def _cif_to_atoms(cif: str):
    """CIF string → ase.Atoms."""
    from io import StringIO
    from ase.io import read as ase_read
    return ase_read(StringIO(cif), format="cif")


def _atoms_to_cif(atoms) -> str:
    """ase.Atoms → CIF string.

    走临时文件 — ase 3.23+ 的 cif writer 末尾调 fd.detach(), StringIO 不支持,
    跟 packing_tool 一样用 temp file 规避.
    """
    import os
    import tempfile
    from ase.io import write as ase_write
    fd, path = tempfile.mkstemp(suffix=".cif")
    try:
        os.close(fd)
        ase_write(path, atoms, format="cif")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _selfcheck() -> None:
    """3 场景 selfcheck, mock CIF 不调真 LLM.

    ponytail: 非平凡逻辑留 runnable check. 不依赖真 atomworld 包,
    场景 3 在包未装时降级为 skip log (跟 ml_potential_tool 的 ImportError 降级一致).
    """
    # 用 ase 生成 mock CIF (简单立方 Si 晶胞)
    from ase.build import bulk
    si = bulk("Si", "diamond", a=5.43)
    mock_cif = _atoms_to_cif(si)

    # 1. list_actions 返回 15 个 action (README 全集)
    actions = list_actions()
    assert len(actions) == 15, f"应 15 个 action (README 全集), got {len(actions)}"
    assert "move_atom_action" in actions
    assert "rotate_around_atom_action" in actions
    assert "delete_around_atom_action" in actions, "delete_around 容易漏"
    print(f"1. list_actions ({len(actions)} actions) OK")

    # 2. evaluate 降级: 包未安装时 raise RuntimeError (mock 测试)
    global _atomworld
    orig = _atomworld
    try:
        _atomworld = None
        try:
            evaluate(mock_cif, mock_cif)
            raise AssertionError("包未安装应 raise RuntimeError")
        except RuntimeError as e:
            assert "pip install" in str(e), f"错误信息应含安装提示, got {e}"
        print("2. evaluate 降级 (包未安装 raise RuntimeError) OK")
    finally:
        _atomworld = orig

    # 3. apply_action 未知 action raise ValueError
    if _atomworld is not None:
        try:
            apply_action(mock_cif, "fly_to_moon_action")
            raise AssertionError("未知 action 应 raise ValueError")
        except ValueError as e:
            assert "unknown action" in str(e)
        print("3. apply_action 未知 action raise ValueError OK")
    else:
        # 包未安装时跳过, 只 log
        print("3. apply_action 未知 action (跳过, 包未安装)")

    print("atomworld_tool selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
