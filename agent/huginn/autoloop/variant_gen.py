"""H2: Workflow variant 生成器.

两种模式:
1. 参数扰动 (默认): 对 base_script 的 subtasks.args 做数值 ±10% / kpoints +1
   扰动, 不需要 LLM. 适合 VASP/Quantum ESPRESSO 等结构化参数.
2. LLM 生成 (fallback): base_script=None 时调 LLM 一次写 N 个 script dict.

toggle: cfg.feature_flags.harness_workflow_evolution (默认 off).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from huginn.autoloop.bandit import _harness_enabled
from huginn.autoloop.dynamic_workflow import WorkflowScript

logger = logging.getLogger(__name__)

# 参数扰动规则. key = 参数名, value = (扰动函数, 适用工具集或 None)
# ponytail: 硬编码 VASP 常见参数, 升级路径: 从 ToolRegistry schema 动态读.

def _perturb_encut(v: Any) -> Any:
    """encut ±50 eV, 钳到 [200, 2000]."""
    try:
        base = float(v)
    except (TypeError, ValueError):
        return v
    delta = 50.0 * (1 if uuid.uuid4().hex[0] < "8" else -1)
    return int(max(200, min(2000, base + delta)))


def _perturb_kpoints(v: Any) -> Any:
    """kpoints grid 'a b c' → 每维 +1 或 -1 (钳到 [1, 12])."""
    if not isinstance(v, str):
        return v
    parts = v.split()
    if len(parts) < 3:
        return v
    out: list[str] = []
    for p in parts[:3]:
        try:
            n = int(p)
        except ValueError:
            out.append(p)
            continue
        delta = 1 if uuid.uuid4().hex[0] < "8" else -1
        out.append(str(max(1, min(12, n + delta))))
    return " ".join(out) + (" " + " ".join(parts[3:]) if len(parts) > 3 else "")


def _perturb_sigma(v: Any) -> Any:
    """sigma 0.05 → 0.02 / 0.1 (二选一)."""
    try:
        base = float(v)
    except (TypeError, ValueError):
        return v
    return 0.02 if uuid.uuid4().hex[0] < "8" else 0.1


def _perturb_numeric(v: Any) -> Any:
    """通用数值参数 ±10%."""
    try:
        base = float(v)
    except (TypeError, ValueError):
        return v
    delta = base * 0.1 * (1 if uuid.uuid4().hex[0] < "8" else -1)
    if isinstance(v, int):
        return int(base + delta)
    return round(base + delta, 4)


# 参数名 → 扰动函数. 不在表里的数值参数走 _perturb_numeric.
_PERTURB_RULES = {
    "encut": _perturb_encut,
    "ENCUT": _perturb_encut,
    "kpoints": _perturb_kpoints,
    "kgrid": _perturb_kpoints,
    "k_grid": _perturb_kpoints,
    "sigma": _perturb_sigma,
    "SIGMA": _perturb_sigma,
    "ediff": lambda v: _perturb_numeric(v),
    "EDIFF": lambda v: _perturb_numeric(v),
}


def _perturb_args(args: dict[str, Any]) -> dict[str, Any]:
    """对 args dict 做参数扰动. 返回新 dict, 原地不动."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        fn = _PERTURB_RULES.get(k)
        if fn is not None:
            out[k] = fn(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = _perturb_numeric(v)
        else:
            out[k] = v
    return out


def _perturb_script(script: WorkflowScript) -> WorkflowScript:
    """对 base_script 做参数扰动, 返回新 WorkflowScript (新 id)."""
    # H3 接入: toggle on 时用 UCB 调参 替代随机扰动 (stage = subtask tool_name)
    h3_on = _harness_enabled("harness_joint_optimizer")
    new_subtasks = []
    for st in script.subtasks:
        if h3_on:
            try:
                from huginn.harness.joint_optimizer import select_workflow_params_for_stage
                new_args = select_workflow_params_for_stage(st.tool_name, st.args)
            except Exception:
                new_args = _perturb_args(st.args)
        else:
            new_args = _perturb_args(st.args)
        new_subtasks.append({
            "id": st.id,
            "tool": st.tool_name,
            "args": new_args,
            "description": st.description,
        })
    raw = {
        "objective": script.objective,
        "max_concurrent": script.max_concurrent,
        "subtasks": new_subtasks,
    }
    return WorkflowScript.from_dict(raw)


async def _llm_generate_variants(
    objective: str,
    n: int,
    llm_chat_fn: Any,
) -> list[WorkflowScript]:
    """LLM 一次写 N 个 script dict. 失败返回空 list."""
    prompt = (
        "You are generating workflow variants for a materials science agent. "
        "Each variant is a declarative script of independent tool subtasks.\n\n"
        f"Objective: {objective}\n"
        f"Generate {n} distinct variants with different parameter choices.\n\n"
        "Output JSON only:\n"
        '{"variants": [{"objective": "...", "max_concurrent": 8, "subtasks": '
        '[{"id": "s1", "tool": "vasp_tool", "args": {"encut": 520, "kpoints": "2 2 2"}, '
        '"description": "..."}]}]}\n'
        "Rules:\n"
        "- Each variant must have at least 1 subtask\n"
        "- Variants must differ in args (encut/kpoints/sigma/etc.)\n"
        "- tool names must be valid registered tools\n"
    )
    try:
        response = await llm_chat_fn(prompt, task="summarize")
    except Exception:
        logger.debug("llm_generate_variants LLM fail", exc_info=True)
        return []
    if not (response and response.strip()):
        return []
    txt = response.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(txt)
    except Exception:
        logger.debug("llm_generate_variants JSON parse fail: %s", txt[:200])
        return []
    variants_raw = d.get("variants", [])
    out: list[WorkflowScript] = []
    for v in variants_raw:
        if not isinstance(v, dict):
            continue
        try:
            out.append(WorkflowScript.from_dict(v))
        except Exception:
            logger.debug("llm variant parse fail", exc_info=True)
    return out[:n]


async def generate_variants(
    objective: str,
    n: int = 3,
    base_script: WorkflowScript | None = None,
    llm_chat_fn: Any = None,
) -> list[WorkflowScript]:
    """生成 n 个 workflow variant.

    优先参数扰动 (base_script 非空), fallback LLM 生成 (base_script=None).
    toggle off 返回空 list.
    """
    if not _harness_enabled("harness_workflow_evolution"):
        return []
    if n <= 0:
        return []
    if base_script is not None:
        # 参数扰动: 同一 base_script 扰动 n 次
        out: list[WorkflowScript] = []
        for _ in range(n):
            out.append(_perturb_script(base_script))
        return out
    if llm_chat_fn is None:
        return []
    return await _llm_generate_variants(objective, n, llm_chat_fn)


def _selfcheck() -> None:
    """variant_gen selfcheck: 参数扰动 + LLM fallback 路径."""
    from huginn.autoloop.dynamic_workflow import WorkflowScript
    import huginn.autoloop.variant_gen as vg
    import huginn.harness.joint_optimizer as jo

    # toggle off → 空 list. patch vg + jo 两处 (H3 接入后两模块各自有 _harness_enabled)
    orig_vg = vg._harness_enabled
    orig_jo = jo._harness_enabled
    vg._harness_enabled = lambda key, default=False: False
    jo._harness_enabled = lambda key, default=False: False
    import asyncio
    out = asyncio.run(vg.generate_variants("test", n=3))
    assert out == [], f"toggle off should return []: {out}"
    print("1. toggle off → [] OK")

    # toggle on + base_script → 参数扰动 (H2 workflow_evolution on, H3 off 走原扰动)
    vg._harness_enabled = lambda key, default=False: (
        True if key == "harness_workflow_evolution" else False
    )
    jo._harness_enabled = lambda key, default=False: False
    base = WorkflowScript.from_dict({
        "objective": "test Si band gap",
        "subtasks": [
            {"id": "s1", "tool": "vasp_tool", "args": {"encut": 520, "kpoints": "2 2 2", "sigma": 0.05}},
        ],
    })
    out = asyncio.run(vg.generate_variants("test", n=3, base_script=base))
    assert len(out) == 3, f"should generate 3 variants: {len(out)}"
    # 每个 variant 的 encut/kpoints 应该跟 base 不同 (至少有一个不同)
    base_encut = base.subtasks[0].args["encut"]
    diff_count = sum(1 for v in out if v.subtasks[0].args["encut"] != base_encut)
    assert diff_count > 0, "at least one variant should differ in encut"
    print(f"2. param perturbation: 3 variants, {diff_count} differ in encut OK")

    # 扰动后的 args 应该在合理范围
    for v in out:
        encut = v.subtasks[0].args["encut"]
        assert 200 <= encut <= 2000, f"encut out of range: {encut}"
        kp = v.subtasks[0].args["kpoints"]
        parts = kp.split()
        assert len(parts) >= 3, f"kpoints wrong: {kp}"
        for p in parts[:3]:
            assert 1 <= int(p) <= 12, f"kpoint out of range: {p}"
    print("3. perturbed args in valid range OK")

    # base_script=None + llm_chat_fn=None → 空 list
    out = asyncio.run(vg.generate_variants("test", n=3, base_script=None, llm_chat_fn=None))
    assert out == [], f"no base + no llm should return []: {out}"
    print("4. no base + no llm → [] OK")

    # 5. H3 接入: H3 toggle on 时 _perturb_script 走 UCB 调参路径
    jo._harness_enabled = lambda key, default=False: (
        True if key == "harness_joint_optimizer" else False
    )
    import os as _os, tempfile as _tf
    _tmp = _tf.mkdtemp()
    _os.environ["HUGINN_CACHE_DIR"] = _tmp
    jo.JointBandit._instance = None
    out_h3 = asyncio.run(vg.generate_variants("h3 test", n=2, base_script=base))
    assert len(out_h3) == 2, f"H3 should still gen 2 variants: {len(out_h3)}"
    h3_diff = sum(1 for v in out_h3 if v.subtasks[0].args["encut"] != base_encut)
    assert h3_diff > 0, f"H3 path should also perturb encut: {h3_diff}"
    # H3 ±10% → encut ∈ [468, 572]
    for v in out_h3:
        _e = v.subtasks[0].args["encut"]
        assert 468 <= _e <= 572, f"H3 encut out of ±10% range: {_e}"
    print(f"5. H3 path: 2 variants, {h3_diff} differ in encut (±10%) OK")
    _os.environ.pop("HUGINN_CACHE_DIR", None)
    jo.JointBandit._instance = None

    vg._harness_enabled = orig_vg
    jo._harness_enabled = orig_jo
    print("\nH2 variant_gen selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()
