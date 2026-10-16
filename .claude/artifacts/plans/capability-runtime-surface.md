# 能力可运行性面 (Capability Runtime Surface) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让系统对「哪些能力因缺依赖而跑不了、缺什么、怎么装」具备**可观察 + 可自愈**的能力面,并把探针清单纳入低熵治理棘轮。

**Architecture:** 新增 `huginn/capabilities/runtime_probe.py`(L0 静态依赖探针 + 报告 + 注入行)与 `huginn/capabilities/smoke_specs.py`(L1 精选端到端 smoke)。报告常驻一行注入 system prompt,详单走 `system_diagnostic_tool` 与 `huginn/cli/capability_runtime.py`。治理侧只在 `governance_audit.BUDGET_SPECS` 登记「探针清单下限」棘轮(weightless,不阻断 CI)。

**Tech Stack:** Python 3.11+ (stdlib `ast`/`importlib.util`/`dataclasses`/`argparse`), pytest, 复用 `huginn.tools._collect_top_level_imports` 与现有治理台账范式。

> 规格来源: [agent/docs/capability-runtime-surface-spec.md](../docs/capability-runtime-surface-spec.md)

---

## File Structure

| 文件 | 责任 |
|---|---|
| `agent/huginn/capabilities/runtime_probe.py` | 新建: L0 探针 + `classify` + 报告组装 + `summary_line` |
| `agent/huginn/capabilities/smoke_specs.py` | 新建: L1 `SmokeSpec`/`SmokeResult`/`SMOKE_SPECS`/`run_smoke` |
| `agent/huginn/cli/capability_runtime.py` | 新建: `--json/--smoke/--out/--check` CLI |
| `agent/huginn/agent/context.py` | 修改: 注入能力面汇总块 |
| `agent/huginn/tools/system_diagnostic_tool.py` | 修改: 新增 `capability_runtime` action |
| `agent/huginn/cli/governance_audit.py` | 修改: 新增 `runtime_probe_floor` 预算 + `runtime_probe_live_count` 探针 |
| `agent/tests/test_capability_runtime_probe.py` | 新建: L0 单测(weightless) |
| `agent/tests/test_capability_runtime_smoke.py` | 新建: L1 单测(weightless) |
| `agent/tests/test_arch_gate_registry.py` | 修改: 冻结集加 `runtime_probe_floor` + 新棘轮测试 |

**约定:** 所有测试只用 stdlib / 合成临时文件,**不依赖 numpy/rdkit**;`agent/` 是 pytest 的 rootdir 基准路径(下文一律相对 `agent/`)。

---

## Task 1: L0 依赖探针 `runtime_probe.py`

**Files:**
- Create: `agent/huginn/capabilities/runtime_probe.py`
- Test: `agent/tests/test_capability_runtime_probe.py`

- [ ] **Step 1: 写失败测试**

创建 `agent/tests/test_capability_runtime_probe.py`:

```python
"""L0 依赖探针单测 —— 全部 weightless: 只用合成临时文件, 不碰重依赖."""

from __future__ import annotations

import pathlib

from huginn.capabilities import runtime_probe as rp


def test_classify_precedence():
    """分类优先级: 缺依赖 > 不可实例化 > 隐藏 > 可运行."""
    assert rp.classify(missing=["numpy"]) == rp.STATUS_MISSING
    assert rp.classify(missing=[], constructed=False) == rp.STATUS_NOT_INSTANTIABLE
    assert rp.classify(missing=[], visible=False) == rp.STATUS_HIDDEN
    assert rp.classify(missing=[], visible=True) == rp.STATUS_RUNNABLE
    assert rp.classify(missing=[], imported=False) == rp.STATUS_MISSING


def test_probe_detects_missing_dep_on_synthetic_module(tmp_path: pathlib.Path):
    mod = tmp_path / "fake_heavy.py"
    mod.write_text("import definitely_absent_pkg_xyz\n", encoding="utf-8")
    results = rp.probe_runtime([rp.ToolSpec(module="fake_heavy", cls="FakeHeavy", path=mod)])
    assert results[0].status == rp.STATUS_MISSING
    assert results[0].missing_deps == ["definitely_absent_pkg_xyz"]
    assert results[0].install_hint == "pip install definitely_absent_pkg_xyz"


def test_probe_marks_stdlib_only_module_runnable(tmp_path: pathlib.Path):
    mod = tmp_path / "fake_ok.py"
    mod.write_text("import json\nimport os\n", encoding="utf-8")
    results = rp.probe_runtime([rp.ToolSpec(module="fake_ok", cls="FakeOk", path=mod)])
    assert results[0].status == rp.STATUS_RUNNABLE
    assert results[0].missing_deps == []


def test_probe_never_raises_on_missing_path():
    """路径不存在时不得抛 —— 收敛为 missing_deps(无法验证)."""
    results = rp.probe_runtime(
        [rp.ToolSpec(module="nope", cls="Nope", path=pathlib.Path("/no/such/file.py"))]
    )
    assert len(results) == 1
    assert results[0].status == rp.STATUS_MISSING


def test_probe_orders_deterministically():
    """相同输入 → 输出逐字一致 (确定性)."""
    a = rp.probe_runtime([rp.ToolSpec(module="m", cls="B"), rp.ToolSpec(module="m", cls="A")])
    b = rp.probe_runtime([rp.ToolSpec(module="m", cls="A"), rp.ToolSpec(module="m", cls="B")])
    assert [r.name for r in a] == [r.name for r in b] == ["A", "B"]


def test_summary_line_all_ok():
    rep = rp.build_report([rp.ProbeResult(name="A", kind="tool", status=rp.STATUS_RUNNABLE)])
    assert rp.summary_line(rep) == rp.ALL_OK_LINE


def test_summary_line_reports_degraded_with_hint():
    rep = rp.build_report(
        [
            rp.ProbeResult(
                name="RDKitTool",
                kind="tool",
                status=rp.STATUS_MISSING,
                missing_deps=["rdkit"],
                install_hint="pip install rdkit",
            )
        ]
    )
    line = rp.summary_line(rep)
    assert "degraded: 1" in line
    assert "rdkit" in line
    assert "pip install rdkit" in line
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_capability_runtime_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'huginn.capabilities.runtime_probe'`

- [ ] **Step 3: 实现 `runtime_probe.py`**

创建 `agent/huginn/capabilities/runtime_probe.py`:

```python
"""能力可运行性面 (runtime_probe): L0 依赖探针 + 报告组装.

与 introspection 的分工: introspection 管**名字面** (工具是否登记/可见),
本模块管**依赖面** (依赖是否可加载 → 能力能否真跑)。纯静态扫描 + 单次
``find_spec``, 不 import 工具模块本体, 保证 weightless、可注入、可测、确定性。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# huginn/capabilities/runtime_probe.py -> parents[2] == agent/.
AGENT_ROOT = Path(__file__).resolve().parents[2]

STATUS_RUNNABLE = "runnable"
STATUS_MISSING = "missing_deps"
STATUS_NOT_INSTANTIABLE = "not_instantiable"
STATUS_HIDDEN = "hidden"

# 全部就绪时注入 context 的一行 (空串表示"不注入", 省 token).
ALL_OK_LINE = "Tool capabilities: all runnable."


@dataclass(frozen=True)
class ToolSpec:
    """一条待探测声明: (模块, 类, 可选源码路径)."""

    module: str
    cls: str
    path: Path | None = None
    kind: str = "tool"


@dataclass
class ProbeResult:
    name: str
    kind: str
    status: str
    missing_deps: list[str] = field(default_factory=list)
    install_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "missing_deps": list(self.missing_deps),
            "install_hint": self.install_hint,
        }


@dataclass
class CapabilityRuntimeReport:
    l0: list[ProbeResult] = field(default_factory=list)
    l1: list = field(default_factory=list)  # list[SmokeResult], 避免循环导入

    def degraded(self) -> list[ProbeResult]:
        return [r for r in self.l0 if r.status != STATUS_RUNNABLE]


def _filter_deps(deps: set[str]) -> list[str]:
    """去掉 stdlib 与 huginn 内部导入, 只留第三方依赖名 (排序)."""
    stdlib = sys.stdlib_module_names
    return sorted(d for d in deps if not d.startswith("huginn") and d not in stdlib)


def _scan(path: Path | None) -> tuple[bool, list[str]]:
    """静态扫描: 返回 (是否可解析, 缺失的第三方依赖)."""
    if path is None or not path.is_file():
        return False, []
    try:
        from huginn.tools import _collect_top_level_imports  # 惰性, 免包级循环导入

        deps = _filter_deps(_collect_top_level_imports(str(path)))
    except Exception:  # 非致命: 解析失败 = 无信息
        logger.debug("dep scan failed: %s", path, exc_info=True)
        return False, []
    missing = [d for d in deps if importlib.util.find_spec(d) is None]
    return True, missing


def classify(
    *,
    missing: list[str],
    imported: bool = True,
    visible: bool | None = None,
    constructed: bool | None = None,
) -> str:
    """纯判据: 由事实推 status. 顺序即优先级."""
    if not imported or missing:
        return STATUS_MISSING
    if constructed is False:
        return STATUS_NOT_INSTANTIABLE
    if visible is False:
        return STATUS_HIDDEN
    return STATUS_RUNNABLE


def install_hint(missing: list[str]) -> str:
    return f"pip install {' '.join(missing)}" if missing else ""


def _module_path(module: str) -> Path | None:
    rel = module.replace(".", "/")
    for cand in (AGENT_ROOT / f"{rel}.py", AGENT_ROOT / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def default_tool_specs() -> list[ToolSpec]:
    """从既有 builtin 工具清单派生 spec (惰性 import, 只取名字 + 静态解析路径)."""
    from huginn.tools import _CORE_MODULES, _OPTIONAL_MODULES

    return [
        ToolSpec(module=m, cls=c, path=_module_path(m))
        for m, c in (*_CORE_MODULES, *_OPTIONAL_MODULES)
    ]


def probe_runtime(
    specs: list[ToolSpec] | None = None,
    *,
    available_names: set[str] | None = None,
) -> list[ProbeResult]:
    """对每条 spec 探测依赖可运行性. 任何异常收敛为 status, 绝不抛."""
    specs = specs if specs is not None else default_tool_specs()
    out: list[ProbeResult] = []
    for spec in specs:
        try:
            imported, missing = _scan(spec.path)
            visible = None if available_names is None else (spec.cls in available_names)
            out.append(
                ProbeResult(
                    name=spec.cls,
                    kind=spec.kind,
                    status=classify(missing=missing, imported=imported, visible=visible),
                    missing_deps=missing,
                    install_hint=install_hint(missing),
                )
            )
        except Exception:  # 探针自身绝不抛
            logger.debug("probe failed for %s", spec, exc_info=True)
            out.append(ProbeResult(name=spec.cls, kind=spec.kind, status=STATUS_MISSING))
    return sorted(out, key=lambda r: (r.kind, r.name))


def build_report(l0: list[ProbeResult] | None = None, l1: list | None = None) -> CapabilityRuntimeReport:
    return CapabilityRuntimeReport(l0=list(l0 or []), l1=list(l1 or []))


def summary_line(report: CapabilityRuntimeReport) -> str:
    """一行汇总, 供注入 system prompt. 全就绪时返回 ALL_OK_LINE."""
    bad = report.degraded()
    if not bad:
        return ALL_OK_LINE
    deps = sorted({d for r in bad for d in r.missing_deps})
    names = ", ".join(r.name for r in bad[:8])
    more = "" if len(bad) <= 8 else f" (+{len(bad) - 8})"
    hint = install_hint(deps)
    tail = f" Install with: {hint}." if hint else ""
    return (
        f"Tool capabilities degraded: {len(bad)} ({names}{more}); "
        f"missing deps: {', '.join(deps) or 'unknown'}.{tail}"
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_capability_runtime_probe.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: ruff + 提交**

Run: `cd agent && python -m ruff check huginn/capabilities/runtime_probe.py tests/test_capability_runtime_probe.py`
Expected: All checks passed

```bash
cd /workspace && git add agent/huginn/capabilities/runtime_probe.py agent/tests/test_capability_runtime_probe.py
git commit -m "feat(capabilities): add L0 runtime dependency probe"
```

---

## Task 2: L1 端到端 smoke `smoke_specs.py`

**Files:**
- Create: `agent/huginn/capabilities/smoke_specs.py`
- Test: `agent/tests/test_capability_runtime_smoke.py`

- [ ] **Step 1: 写失败测试**

创建 `agent/tests/test_capability_runtime_smoke.py`:

```python
"""L1 端到端 smoke 单测 —— weightless: 用注入的假 spec, 不依赖 numpy/rdkit."""

from __future__ import annotations

from huginn.capabilities import smoke_specs as ss


def test_run_smoke_passes():
    specs = (ss.SmokeSpec(name="ok", capability="m", runner=lambda: 42, expect=lambda v: v == 42),)
    res = ss.run_smoke(specs)
    assert [r.status for r in res] == ["passed"]


def test_run_smoke_fails_on_exception():
    def boom():
        raise ValueError("x")

    specs = (ss.SmokeSpec(name="boom", capability="m", runner=boom),)
    res = ss.run_smoke(specs)
    assert res[0].status == "failed"
    assert "ValueError" in res[0].detail


def test_run_smoke_fails_on_expect_reject():
    specs = (ss.SmokeSpec(name="bad", capability="m", runner=lambda: 1, expect=lambda v: v == 999),)
    res = ss.run_smoke(specs)
    assert res[0].status == "failed"


def test_run_smoke_skips_when_requires_missing():
    specs = (
        ss.SmokeSpec(
            name="need",
            capability="m",
            requires=("definitely_absent_pkg_xyz",),
            runner=lambda: 1,
        ),
    )
    res = ss.run_smoke(specs)
    assert res[0].status == "skipped"
    assert "definitely_absent_pkg_xyz" in res[0].detail


def test_smoke_names_tuple_matches_specs():
    """台账探针读 SMOKE_SPEC_NAMES; 必须与 SMOKE_SPECS 一致 (单一事实来源)."""
    assert tuple(s.name for s in ss.SMOKE_SPECS) == ss.SMOKE_SPEC_NAMES


def test_smoke_results_ordered_deterministically():
    specs = (
        ss.SmokeSpec(name="z", capability="m", runner=lambda: 1),
        ss.SmokeSpec(name="a", capability="m", runner=lambda: 1),
    )
    assert [r.name for r in ss.run_smoke(specs)] == ["a", "z"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_capability_runtime_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'huginn.capabilities.smoke_specs'`

- [ ] **Step 3: 实现 `smoke_specs.py`**

创建 `agent/huginn/capabilities/smoke_specs.py`:

```python
"""L1 精选端到端 smoke: 对科学关键能力各跑一条**最小真跑**.

L0 (runtime_probe) 只证「依赖可加载」; L1 证「最小任务能产出可判定结果」。
缺 ``requires`` 的能力 **skip 不 fail** (对齐既有 rcb/hypothesis 模式)。
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
    return [m for m in requires if importlib.util.find_spec(m) is None]


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
                SmokeResult(spec.name, spec.capability, "skipped", f"missing: {', '.join(miss)}")
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
            SmokeResult(spec.name, spec.capability, status, detail, time.monotonic() - t0)
        )
    return sorted(out, key=lambda r: r.name)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_capability_runtime_smoke.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: ruff + 提交**

Run: `cd agent && python -m ruff check huginn/capabilities/smoke_specs.py tests/test_capability_runtime_smoke.py`

```bash
cd /workspace && git add agent/huginn/capabilities/smoke_specs.py agent/tests/test_capability_runtime_smoke.py
git commit -m "feat(capabilities): add L1 science smoke specs"
```

---

## Task 3: CLI `capability_runtime.py`

**Files:**
- Create: `agent/huginn/cli/capability_runtime.py`
- Test: `agent/tests/test_capability_runtime_cli.py`

- [ ] **Step 1: 写失败测试**

创建 `agent/tests/test_capability_runtime_cli.py`:

```python
"""CLI 单测: 只验证 --json 输出结构与 --check 语义 (不真跑 L1)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

# 按文件路径加载: 避免触发 huginn.cli 包 __init__ (它 import rich, 轻量环境可能缺).
_CLI_PATH = pathlib.Path(__file__).resolve().parents[1] / "huginn" / "cli" / "capability_runtime.py"
_spec = importlib.util.spec_from_file_location("capability_runtime_cli", _CLI_PATH)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def test_json_output_has_l0_and_summary(capsys):
    rc = cr.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "l0" in payload and "summary" in payload
    assert payload["l1"] == []


def test_json_is_deterministic(capsys):
    cr.main(["--json"])
    first = capsys.readouterr().out
    cr.main(["--json"])
    second = capsys.readouterr().out
    assert first == second


def test_check_returns_zero_when_everything_runnable(monkeypatch, capsys):
    monkeypatch.setattr(cr, "probe_runtime", lambda *a, **k: [])
    assert cr.main(["--json", "--check"]) == 0


def test_out_writes_file(tmp_path, capsys):
    out = tmp_path / "state.json"
    rc = cr.main(["--json", "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["l0"] is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_capability_runtime_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'huginn.cli.capability_runtime'`

- [ ] **Step 3: 实现 CLI**

创建 `agent/huginn/cli/capability_runtime.py`:

```python
"""能力可运行性 CLI: L0 依赖盘点 (+ 可选 L1 端到端 smoke).

用法:
    python -m huginn.cli.capability_runtime --json        # 打印 JSON
    python -m huginn.cli.capability_runtime --smoke       # 附带跑 L1
    python -m huginn.cli.capability_runtime --check       # 有降级能力则 exit 1
"""

from __future__ import annotations

import argparse
import json
import sys

from huginn.capabilities.runtime_probe import (
    build_report,
    probe_runtime,
    summary_line,
)


def _payload(report) -> dict:
    return {
        "l0": [r.to_dict() for r in report.l0],
        "l1": [r.to_dict() for r in report.l1],
        "summary": summary_line(report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--smoke", action="store_true", help="附带跑 L1 端到端 smoke")
    parser.add_argument("--out", default="", help="写到文件 (默认 stdout)")
    parser.add_argument("--check", action="store_true", help="有降级能力则 exit 1")
    args = parser.parse_args(argv)

    l0 = probe_runtime()
    l1 = []
    if args.smoke:
        from huginn.capabilities.smoke_specs import run_smoke

        l1 = run_smoke()
    report = build_report(l0, l1)
    payload = _payload(report)
    text = json.dumps(payload, ensure_ascii=False, indent=2) if args.json else summary_line(report)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.check and report.degraded():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_capability_runtime_cli.py -q`
Expected: PASS (4 passed)

> 注: 本 CLI 位于 `huginn.cli` 包下, `python -m` 会先加载 `huginn/cli/__init__.py`(依赖 `rich`)。CI 环境已装 `rich`; 轻量沙箱若缺 `rich`, 测试通过直接 import 模块完成(不触发包级 `-m` 路径)。

- [ ] **Step 5: ruff + 提交**

Run: `cd agent && python -m ruff check huginn/cli/capability_runtime.py tests/test_capability_runtime_cli.py`

```bash
cd /workspace && git add agent/huginn/cli/capability_runtime.py agent/tests/test_capability_runtime_cli.py
git commit -m "feat(cli): add capability_runtime inventory CLI"
```

---

## Task 4: 注入 system prompt

**Files:**
- Modify: `agent/huginn/agent/context.py` (现有 missing backends 块之后, 约 L152)
- Test: `agent/tests/test_capability_context_note.py`

- [ ] **Step 1: 写失败测试**

创建 `agent/tests/test_capability_context_note.py`:

```python
"""能力面汇总注入块的单测 —— 只测纯函数, 不构造 AgentContext."""

from __future__ import annotations

from huginn.agent import context as ctx


def test_note_empty_when_all_runnable(monkeypatch):
    monkeypatch.setattr(
        "huginn.capabilities.runtime_probe.probe_runtime", lambda *a, **k: []
    )
    assert ctx._capability_note() == ""


def test_note_includes_degraded_line(monkeypatch):
    from huginn.capabilities import runtime_probe as rp

    monkeypatch.setattr(
        rp,
        "probe_runtime",
        lambda *a, **k: [
            rp.ProbeResult(
                name="RDKitTool",
                kind="tool",
                status=rp.STATUS_MISSING,
                missing_deps=["rdkit"],
                install_hint="pip install rdkit",
            )
        ],
    )
    note = ctx._capability_note()
    assert "Tool capabilities degraded" in note
    assert "rdkit" in note


def test_note_swallows_probe_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("huginn.capabilities.runtime_probe.probe_runtime", boom)
    assert ctx._capability_note() == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_capability_context_note.py -q`
Expected: FAIL — `AttributeError: module 'huginn.agent.context' has no attribute '_capability_note'`

- [ ] **Step 3: 加纯函数 + 接线**

在 `agent/huginn/agent/context.py` 顶部(与其他模块级 helper 并列)新增:

```python
def _capability_note() -> str:
    """能力面汇总块: 有降级能力时返回可注入的一段, 全就绪/失败时返回空串.

    诚实边界: 只陈述「哪些能力因缺依赖不可用、可 pip 装什么」, 不暗示能力已可用.
    """
    try:
        from huginn.capabilities.runtime_probe import (
            build_report,
            probe_runtime,
            summary_line,
        )

        line = summary_line(build_report(probe_runtime()))
    except Exception:  # 非致命: 能力盘点失败则回落不注入
        logger.debug("capability runtime probe failed", exc_info=True)
        return ""
    if line.endswith("all runnable."):
        return ""  # 全就绪不注入, 省 token
    return f"\n\n# Tool Capabilities\n{line}"
```

在现有 missing backends 块之后(即 `if _missing: base += ...` 之后)接线:

```python
        # 能力面汇总 — 每次构建重算(不缓存), 保证装完依赖后不注入陈旧状态 (AC-4).
        # L0 为纯静态扫描 (不 import 工具模块、无网络), 开销可控.
        base += _capability_note()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_capability_context_note.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: ruff + 提交**

Run: `cd agent && python -m ruff check huginn/agent/context.py tests/test_capability_context_note.py`

```bash
cd /workspace && git add agent/huginn/agent/context.py agent/tests/test_capability_context_note.py
git commit -m "feat(agent): inject capability runtime summary into system prompt"
```

---

## Task 5: 诊断工具按需详单

**Files:**
- Modify: `agent/huginn/tools/system_diagnostic_tool.py:21-30` (action Literal) 与 `:114-129` (dispatch)
- Test: `agent/tests/test_system_diagnostic_capability_action.py`

- [ ] **Step 1: 写失败测试**

创建 `agent/tests/test_system_diagnostic_capability_action.py`:

```python
"""system_diagnostic_tool 新增 capability_runtime action 的行为."""

from __future__ import annotations

import asyncio

from huginn.tools.system_diagnostic_tool import SystemDiagnosticInput, SystemDiagnosticTool


def test_action_is_accepted_by_schema():
    SystemDiagnosticInput(action="capability_runtime")  # 不得抛 ValidationError


def test_action_returns_degraded_list():
    tool = SystemDiagnosticTool()
    result = asyncio.run(tool.call({"action": "capability_runtime"}))
    assert result.success is True
    assert "summary" in result.data
    assert isinstance(result.data["degraded"], list)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_system_diagnostic_capability_action.py -q`
Expected: FAIL — `pydantic.ValidationError`(action 不在 Literal 内)

- [ ] **Step 3: 扩 Literal + dispatch**

在 `SystemDiagnosticInput.action` 的 `Literal[...]` 中追加 `"capability_runtime",`,并更新 `description`:

```python
    action: Literal[
        "snapshot",
        "diagnose",
        "recent_anomalies",
        "top_processes",
        "capability_runtime",
    ] = Field(
        ...,
        description="snapshot: 当前资源快照; diagnose: 诊断异常+根因; "
        "recent_anomalies: 最近记录的异常事件; top_processes: 吃资源最多的进程; "
        "capability_runtime: 能力可运行性详单 (哪些工具因缺依赖降级、可 pip 装什么).",
    )
```

在 `call` 内 `top_processes` 分支之后、最终 `return ToolResult(... Unknown action ...)` 之前插入:

```python
        if input_data.action == "capability_runtime":
            from huginn.capabilities.runtime_probe import (
                build_report,
                probe_runtime,
                summary_line,
            )

            report = build_report(probe_runtime())
            return ToolResult(
                data={
                    "summary": summary_line(report),
                    "degraded": [r.to_dict() for r in report.degraded()],
                },
                success=True,
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_system_diagnostic_capability_action.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: ruff + 提交**

Run: `cd agent && python -m ruff check huginn/tools/system_diagnostic_tool.py tests/test_system_diagnostic_capability_action.py`

```bash
cd /workspace && git add agent/huginn/tools/system_diagnostic_tool.py agent/tests/test_system_diagnostic_capability_action.py
git commit -m "feat(tools): expose capability_runtime detail via system_diagnostic_tool"
```

---

## Task 6: 治理台账棘轮

**Files:**
- Modify: `agent/huginn/cli/governance_audit.py` — `BUDGET_SPECS`(末尾 `loop_cap_floor` 项之后, 约 L1489) + `_probe_current`(约 L1650 `loop_cap_live_count` 分支之后)
- Modify: `agent/tests/test_arch_gate_registry.py` — `_EXPECTED_BUDGET_IDS`(L632 后) + 新增棘轮测试
- Test: `agent/tests/test_arch_gate_registry.py`(新增用例)

- [ ] **Step 1: 写失败测试**

在 `agent/tests/test_arch_gate_registry.py` 的 `_EXPECTED_BUDGET_IDS` 里, 于 `"loop_cap_floor",` 之后追加一行:

```python
        "runtime_probe_floor",
```

并在文件末尾(fixture `_AGENT` 之后)追加:

```python
def test_runtime_probe_floor_ratchet_only_shrinks():
    """可运行性探针清单只增不减: SMOKE_SPECS 条目数 ≥ 台账下限."""
    floor = ga.budget_value("runtime_probe_floor")
    spec = next(s for s in ga.BUDGET_SPECS if s["id"] == "runtime_probe_floor")
    live = ga._probe_current(spec, _AGENT, heavy=False)
    assert live is not None, "探针清单活值读不到 —— path/变量名失效, 棘轮静默失效."
    assert live >= floor, (
        f"可运行性探针清单从 {floor} 降到 {live} —— 探针被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 runtime_probe_floor 的 value."
    )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd agent && python -m pytest tests/test_arch_gate_registry.py::test_budget_ids_unique_and_complete tests/test_arch_gate_registry.py::test_runtime_probe_floor_ratchet_only_shrinks -q`
Expected: FAIL — `KeyError: 'runtime_probe_floor'`(预算 id 未登记)

- [ ] **Step 3: 登记预算 + 探针**

在 `governance_audit.py` 的 `BUDGET_SPECS` 中 `loop_cap_floor` 项之后(`)` 收尾之前)插入:

```python
    {
        "id": "runtime_probe_floor",
        "name": "可运行性探针清单下限",
        "path": "huginn/capabilities/smoke_specs.py",
        "group": "runtime",
        "kind": "const",
        "source": "inline",
        "value": 4,
        "key": "SMOKE_SPEC_NAMES",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "runtime_probe_live_count",
        "direction": "floor",
    },
```

在 `_probe_current` 内 `loop_cap_live_count` 分支之后插入:

```python
    if probe == "runtime_probe_live_count":  # 可运行性探针清单条目数 (floor 棘轮)
        path = agent_root / spec["path"]
        if not path.is_file():
            return None
        return _extract_collection_len(path.read_text(encoding="utf-8"), spec["key"])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd agent && python -m pytest tests/test_arch_gate_registry.py -q`
Expected: PASS (含新用例; 现有 45 项不变)

> 若 `test_budget_frozen_values_extract_cleanly` / `test_budget_probe_headroom_is_nonneg` 失败: 说明 `SMOKE_SPEC_NAMES` 条目数与台账 `value: 4` 不一致 —— 二者必须相等。

- [ ] **Step 5: 重生成治理文档**

Run: `cd agent && python -m huginn.cli.governance_audit --out docs/governance-state.md`
Expected: 无报错; `docs/governance-state.md` 的「预算」表新增一行 `runtime_probe_floor`(模式=下限)。

- [ ] **Step 6: 确定性校验 + 提交**

Run: `cd agent && python -m pytest tests/test_arch_gate_registry.py::test_committed_state_doc_matches_registry -q`
Expected: PASS

```bash
cd /workspace && git add agent/huginn/cli/governance_audit.py agent/tests/test_arch_gate_registry.py agent/docs/governance-state.md
git commit -m "feat(governance): register runtime probe floor ratchet"
```

---

## Task 7: 索引登记 + 全量收尾

**Files:**
- Modify: `agent/docs/INDEX.md`(第 2 节, spec 行之后)
- Modify: `agent/docs/capability-runtime-surface-spec.md`(Status → APPROVED)

- [ ] **Step 1: 登记 CLI 与 spec 状态**

在 `agent/docs/INDEX.md` 第 2 节 spec 行之后追加:

```markdown
| [capability-runtime-surface-spec.md](capability-runtime-surface-spec.md) | active | 能力可运行性面（L0 依赖探针 + L1 精选 smoke + 运行时自愈 + 台账棘轮；CLI: `python -m huginn.cli.capability_runtime --json`） |
```

> 若原行已存在(staging), 改为 `active` 并补 CLI 说明, **不要重复登记**。

在 `agent/docs/capability-runtime-surface-spec.md` 顶部把 `> Status: DRAFT (待用户评审)` 改为 `> Status: APPROVED`。

- [ ] **Step 2: 全量相关门**

Run: `cd agent && python -m pytest tests/test_capability_runtime_probe.py tests/test_capability_runtime_smoke.py tests/test_capability_runtime_cli.py tests/test_capability_context_note.py tests/test_system_diagnostic_capability_action.py tests/test_arch_gate_registry.py -q`
Expected: 全部 PASS

Run: `cd agent && python -m ruff check huginn/capabilities/runtime_probe.py huginn/capabilities/smoke_specs.py huginn/cli/capability_runtime.py huginn/agent/context.py huginn/tools/system_diagnostic_tool.py huginn/cli/governance_audit.py tests/test_capability_runtime_probe.py tests/test_capability_runtime_smoke.py tests/test_capability_runtime_cli.py tests/test_capability_context_note.py tests/test_system_diagnostic_capability_action.py`
Expected: All checks passed

- [ ] **Step 3: 提交**

```bash
cd /workspace && git add agent/docs/INDEX.md agent/docs/capability-runtime-surface-spec.md
git commit -m "docs: register capability runtime surface as active"
```

---

## 覆盖对照(spec → task)

| spec AC | Task |
|---|---|
| AC-1 CLI `--json` 确定性 | T3 |
| AC-2 四类 status 分类 + 不抛 | T1 |
| AC-3 `summary_line` 半空/含 hint | T1 |
| AC-4 会话注入与 L0 一致 | T4 |
| AC-5 L1 `--smoke` passed/skip | T2, T3 |
| AC-6 自愈走 ask 门禁 | 复用既有 `bash_tool` + `default_policy`(无代码改动, 由 `install_hint` 触发) |
| AC-7 `governance-state.md` 逐字相等 | T6 |
| AC-8 探针清单下限棘轮 | T6 |

## 有意偏离 spec 之处(诚实记录)

- **L1 起始清单不含文献/网络 smoke**: spec §6 原列 `lit_arxiv_reach`。网络能力在离线环境无法确定性 skip(会误报 `failed`), 故本期以**依赖可判定**的 3 条科学 smoke(`curve_fit` / `xrd_peak_detect` / `rdkit_mol_props`)+ 1 条机制 smoke(`stdlib_smoke`)替代; 网络 smoke 留待 spec §13 后续。
- **`not_instantiable` 状态**: L0 静态探针默认不构造实例, 该状态由 `classify(constructed=False)` 提供接口供后续动态探测使用; 本期仅在单测覆盖判据, 不接真实构造(避免会话构建变重)。
- **不缓存能力面汇总(偏离 spec §7 的「缓存失效」措辞)**: 改为每次构建重算, 以满足 AC-4「不缓陈旧值」。代价是每次构建一次纯静态扫描(不 import、无网络); 若后续实测发现构建频繁导致开销不可忽略, 再引入「装包后显式失效」的缓存(见 spec §12 风险)。

## 执行期评审驱动修正(诚实记录)

### Task 1 (L0 runtime_probe) — 分支 `fix/mece-hook-audit`, commit `b25bf7d8` / `a51c9aaa` / `b05f2440`
- **install_hint 增加 import→分发名映射**(评审发现: 直接拼 `pip install <import 名>` 会产出 `pip install PIL / sklearn / yaml / z3` 等错误命令 —— `pip install PIL` 根本不存在, `yaml` 装错包, 破坏「缺什么、怎么装」的诚实边界与 AC-6 自愈)。新增 `_IMPORT_TO_DIST`(仅登记本仓库工具集确认的 7 对: `PIL→Pillow`、`sklearn→scikit-learn`、`yaml→PyYAML`、`mace→mace-torch`、`fairchem→fairchem-core`、`tavily→tavily-python`、`z3→z3-solver`); 未登记项原样透传(依赖 pip 对 `_`/`-`/`.` 的归一化)。附单测 `test_install_hint_maps_import_names_to_distributions` 守护。
- **summary_line 去重**: 原实现把同一批缺失依赖列两遍(「missing deps」+「pip install」), 实测注入行 ~1KB; 改为只保留可执行的 `Install with: ...` 一处, 仍是单行(符合 spec §12「汇总行须保持一行」)。
- **未采纳的评审项(判定为已知限制 / 不构成缺陷)**:
  - `model` 之类运行时 vendored/本地包被静态扫描误报为缺失三方依赖 —— 与既有 `probe_tool_dependencies`(`huginn/tools/__init__.py:420-425`)同口径(§5 要求复用其逻辑), 属静态扫描固有边界; 见 spec §12。
  - 每次构建 ~1s 的探针开销(157 spec 实测于裸沙箱) —— 计划已显式选择「不缓存」以满足 AC-4; 依赖齐备环境汇总为「all runnable」不注入, 代价仅出现在降级环境。若后续实测高频构建成为瓶颈, 按 spec §12 引入「装包后显式失效」的缓存。
  - `default_tool_specs()` 保持在 try 之外 —— 若 `huginn.tools` 不可导入, 让其抛出并由调用方(`_capability_note`/CLI/诊断工具)捕获回落「不注入」, 优于收敛为空清单后误报「all runnable」。

### Task 2 (L1 smoke_specs) — commit `30f3f5ea` / `a1b1cf86`
- **`Callable` 改从 `collections.abc` 导入**(计划原写 `from typing import Any, Callable`, 触发 ruff `UP035`; 与仓库既有约定一致, 行为等价)。
- **`_missing_requires` 加固**: 原用列表推导直接调 `importlib.util.find_spec`, 对「父包缺失的点分名」会抛 `ModuleNotFoundError` —— 与模块 docstring/计划的「`run_smoke` 绝不抛」契约相悖(评审发现)。改为逐名 try/except, 异常收敛为「缺失」。附回归测试 `test_run_smoke_dotted_missing_require_does_not_raise`。
- **删除死代码**: 计划样例自带但从未使用的 `import logging` + `logger` 移除(评审发现)。
- **补 4 条边界测试**: None-expect 通过 / 空清单 / 多缺失 detail / 点分名不抛; 并给 expect-reject 用例补 detail 断言(6 → 10)。
- **`ruff format` 归一化**: 两文件在原固定版本 `ruff==0.6.9` 下本就未 format-clean; 顺带归一化(纯排版, 无逻辑变化)。
- **未采纳的评审项**: 不给 `_expect_curve_fit` 补截距断言(计划刻意保持「最小真跑」); `to_dict() -> dict` 不收紧为参数化(与 Task 1 `ProbeResult.to_dict` 风格一致, mypy 不强制); `elapsed` 使含 `--smoke` 的 L1 报告失去字节级确定性 —— 属预期(AC-1 只要求 L0 `--json` 确定性, 且 CLI 确定性测试不跑 `--smoke`)。

### Task 3 (CLI capability_runtime) — 质量评审补齐
- **`--out` 缺父目录创建**: 原实现 `Path(args.out).write_text(...)` 在目标父目录不存在时抛 `FileNotFoundError`(评审发现)。改为先 `out_path.parent.mkdir(parents=True, exist_ok=True)` 再写; 附单测 `test_out_creates_missing_parent_dirs`。
- **`--check` exit=1 路径无测试守护**: 原测试只覆盖「全可运行 → 0」, 未覆盖「有降级 → 1」。补单测 `test_check_returns_one_when_degraded`(注入一个 `STATUS_MISSING` 的 `ProbeResult`)。

### Task 4 (注入 system prompt) — 计划缺陷修正 (weightless)
- **计划缺陷**: 计划让单测 `from huginn.agent import context` —— 这会触发 `huginn/agent/__init__.py → core → langchain_core`, 且 `context.py` 顶层 `import huginn.utils.prompt_cache` 也拉 `PromptCacheBuilder`(→ langchain_core)。裸沙箱无 `langchain_core`, 与仓库既定「weightless 测试须在裸环境可跑」(conftest 显式支持)相悖。
- **修正**: 把「能力面汇总块」的**纯逻辑**落到 weightless 模块 `runtime_probe.capability_note()`(单源、绝不抛、每次重算), `context._capability_note()` 退化为**薄导入隔离封装**(仅 `try import` + 回落空串)。单测 `test_capability_context_note.py` 直接覆盖 `runtime_probe.capability_note`(weightless), 不再 import `huginn.agent`。AC-4 语义不变(汇总=实时 L0, 不缓陈旧值)。
- **偏差说明**: 计划里 `context._capability_note()` 内联 `build_report/probe_runtime/summary_line` 的实现被替换为委托 `runtime_probe.capability_note()`; 文件名 `test_capability_context_note.py` 保留(仍指「注入 context 的那段 note」)。

### Task 5 (诊断工具 capability_runtime action)
- 按计划扩 `SystemDiagnosticInput.action` 的 `Literal` 并新增 dispatch 分支; 附 2 项单测(schema 接受 + 返回 `degraded` 列表)。无偏离。

### Task 6 (治理台账棘轮) — 评审驱动修正
- **`kind` 选用 `collection-len` 而非计划的 `const`**: 探针计数对象 `SMOKE_SPEC_NAMES` 是**元组**(非整数常量), `_frozen_value` 对 `kind=const` 走 `_extract_int_const`(只认 `KEY = <int>`)。虽本项 `source=inline` 使冻结值直接取台账 `value`, 不触发该分支, 但 `const` 语义错配。改为 `kind=collection-len` + `var=SMOKE_SPEC_NAMES`(与 `_frozen_value` 对 `collection-len` 的 `_extract_collection_len` 一致), 探针分支读 `spec["var"]`。
- **镜像分叉同步(计划未提)**: `governance_audit.py` 在 `publish/mcpb/{capabilities,workflows}-mcp/` 有 vendored 镜像(only-shrink 漂移门)。本轮把新增的 `runtime_probe.py`/`smoke_specs.py`/`cli/capability_runtime.py` 及改动的 `context.py`/`system_diagnostic_tool.py`/`governance_audit.py` 经 `scripts/mcpb_vendor.py --write` 同步两镜像; `--check` 后漂移 0。
- **裸环境无法 `python -m huginn.cli.governance_audit`**(`huginn/cli/__init__.py` 急切 import `rich`, 裸沙箱缺): 改按文件路径加载 `governance_audit` 调 `main(["--out", ...])` 再生成 `docs/governance-state.md`; 与 CI 下 `-m` 路径产物一致(逐字校验门 `test_committed_state_doc_matches_registry` 通过)。
