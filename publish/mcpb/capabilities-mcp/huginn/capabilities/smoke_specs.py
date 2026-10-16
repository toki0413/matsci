"""L1 精选端到端 smoke: 对科学关键能力各跑一条**最小真跑**.

L0 (runtime_probe) 只证「依赖可加载」; L1 证「最小任务能产出可判定结果」。
缺 ``requires`` 的能力 **skip 不 fail** (对齐既有 rcb/hypothesis 模式)。
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SmokeSpec:
    name: str
    capability: str
    runner: Callable[[], Any]
    requires: tuple[str, ...] = ()
    expect: Callable[[Any], bool] | None = None


@dataclass
class SmokeResult:
    name: str
    capability: str
    status: str  # passed | failed | skipped
    detail: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capability": self.capability,
            "status": self.status,
            "detail": self.detail,
            "elapsed": round(self.elapsed, 4),
        }


def _missing_requires(requires: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in requires:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:  # 点分名父包缺失等 → 视为缺失, 绝不向上抛
            missing.append(name)
    return missing


# ── 最小真跑 runners (只碰 requires 里声明过的库) ──────────────────────────


def _run_stdlib_smoke() -> dict:
    """机制自检: 纯 stdlib, 永远可跑, 保证 smoke 机制本身可测."""
    import json

    payload = {"ok": True, "n": 3}
    return json.loads(json.dumps(payload))


def _run_curve_fit() -> tuple[float, float]:
    import numpy as np

    x = np.arange(10.0)
    y = 2.0 * x + 1.0
    slope, intercept = np.polyfit(x, y, 1)
    return round(float(slope), 6), round(float(intercept), 6)


def _expect_curve_fit(value: Any) -> bool:
    return isinstance(value, tuple) and abs(value[0] - 2.0) < 1e-6


def _run_xrd_peak_detect() -> list[float]:
    import numpy as np

    x = np.arange(0.0, 10.0, 0.01)
    y = np.exp(-0.5 * ((x - 3.0) / 0.1) ** 2) + np.exp(-0.5 * ((x - 7.0) / 0.1) ** 2)
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0.5))[0] + 1
    return sorted(round(float(x[i]), 1) for i in idx)


def _expect_xrd_peak_detect(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2


def _run_rdkit_mol_props() -> float:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles("c1ccccc1")
    return round(float(Descriptors.MolWt(mol)), 1)


def _expect_rdkit_mol_props(value: Any) -> bool:
    return isinstance(value, float) and 70.0 < value < 90.0


# 台账探针 (governance_audit) 用 ast 读这个 tuple 取清单条目数 —— 必须与 SMOKE_SPECS 同步.
SMOKE_SPEC_NAMES: tuple[str, ...] = (
    "stdlib_smoke",
    "curve_fit",
    "xrd_peak_detect",
    "rdkit_mol_props",
)

SMOKE_SPECS: tuple[SmokeSpec, ...] = (
    SmokeSpec(name="stdlib_smoke", capability="mechanism", runner=_run_stdlib_smoke),
    SmokeSpec(
        name="curve_fit",
        capability="experiment.curve",
        requires=("numpy",),
        runner=_run_curve_fit,
        expect=_expect_curve_fit,
    ),
    SmokeSpec(
        name="xrd_peak_detect",
        capability="characterization.xrd",
        requires=("numpy",),
        runner=_run_xrd_peak_detect,
        expect=_expect_xrd_peak_detect,
    ),
    SmokeSpec(
        name="rdkit_mol_props",
        capability="molecule.property",
        requires=("rdkit",),
        runner=_run_rdkit_mol_props,
        expect=_expect_rdkit_mol_props,
    ),
)


def run_smoke(specs: tuple[SmokeSpec, ...] | None = None) -> list[SmokeResult]:
    """逐条跑 smoke; 缺依赖 skip, 异常/expect 不通过为 failed. 绝不抛."""
    specs = SMOKE_SPECS if specs is None else specs
    out: list[SmokeResult] = []
    for spec in specs:
        miss = _missing_requires(spec.requires)
        if miss:
            out.append(
                SmokeResult(
                    spec.name, spec.capability, "skipped", f"missing: {', '.join(miss)}"
                )
            )
            continue
        t0 = time.monotonic()
        try:
            value = spec.runner()
            ok = spec.expect(value) if spec.expect else True
            status = "passed" if ok else "failed"
            detail = "" if ok else f"expect() rejected: {value!r}"
        except Exception as exc:  # 真跑失败不得拖垮调用方
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
        out.append(
            SmokeResult(
                spec.name, spec.capability, status, detail, time.monotonic() - t0
            )
        )
    return sorted(out, key=lambda r: r.name)
