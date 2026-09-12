"""A线·第一步: 客观题"真执行"回归基线 (只读, 不动引擎不加闸).

目的: 量化 uner同一套 AutoloopEngine 面对"确定性能算的客观题"时, 到底能真算出结果多少。
把"换名空转"从印象变成数字, 作为后续 A 线改动的对比锚点。

只做一件事: 对 5 道闭式可判对错的客观题, 各跑一次 run_cognitive(max_iter 8),
记录 每轮真工具调用数 / goal_achieved / surprise / 换名标志 / 与闭式答案比对。
不修改任何引擎/门禁代码.

用法(agent 根目录, 已装 deps venv):
    export HUGINN_PROVIDER=deepseek HUGINN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=...
    PYTHONPATH=. python -m huginn.run_aline_baseline [max_iter]
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

from huginn.autoloop.engine import AutoloopEngine
from huginn.memory.manager import MemoryManager


def _run_py(src: str) -> str:
    import subprocess
    from pathlib import Path as P
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "probe.py"
        f.write_text(src, encoding="utf-8")
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr).strip()


# 5 道客观题: (name, objective 文案, 闭式校验脚本[打印标准答案 → check 用它比对])
ALINE_OBJECTIVES = [
    ("stress_f_div_a",
     "A tensile force F=500 N over area A=0.01 m^2 produces normal stress sigma=F/A. Predict the numeric stress in Pa.",
     "print(f'{500/0.01:.6g}')"),                      # 5e4
    ("stress_safety_pass",
     "A rod (sigma=F/A) under F=500 N, A=0.01 m^2 has a yield stress of 250 MPa. Predict the applied stress in MPa and give a PASS/FAIL safety verdict.",
     "s=500/0.01; print(f'{s/1e6:.6g} {(\"PASS\" if s<250e6 else \"FAIL\")}')"),  # 0.05 PASS
    ("pendulum_period",
     "An ideal pendulum of length L=1.0 m in g=9.81 m/s^2 has period T=2*pi*sqrt(L/g). Predict the numeric period in seconds.",
     "import math\nprint(f'{2*math.pi*math.sqrt(1/9.81):.6g}')"),    # ~2.0064
    ("ideal_gas_volume",
     "n=0.5 mol ideal gas at T=310 K and P=101325 Pa (R=8.314) has volume V=nRT/P. Predict the numeric volume in m^3.",
     "print(f'{0.5*8.314*310/101325:.6g}')"),          # ~0.012715
    ("electric_power_vi",
     "A resistor across V=12 V carrying I=2 A dissipates power P=V*I. Predict the numeric power in watts.",
     "print(f'{12*2:.6g}')"),                          # 24
]


# 换名/空断言 标志词 (A线核心病灶指标): 出现在 phase result/plan/hypothesis 即记一次
_RENAME_MARKERS = ("DIM[:", "换名", "空断言", "renaming", "restate")


async def _run_one(eng: AutoloopEngine, name: str, objective: str, check: str) -> dict:
    try:
        result = await eng.run_cognitive(objective, max_iterations=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    except Exception as exc:  # 单题失败不拖垮整次基线
        return {"name": name, "objective": objective, "error": str(exc)}
    phases = getattr(result, "phases", []) or []
    tool_hits = 0
    rename_hits = 0
    rend_txt = []
    for p in phases:
        txt = str(getattr(p, "result", "") or "") + " " + str(getattr(p, "summary", "") or "")
        # 粗判"真执行": phase 结果带数值/代码样标记
        if re.search(r"\d", txt) and re.search(r"(=|print|import|VERDICT|sigma|PASS|FAIL)", txt):
            tool_hits += 1
        if any(m in txt for m in _RENAME_MARKERS):
            rename_hits += 1
        if getattr(p, "name", "") == "execute" and re.search(r"\d", txt):
            rend_txt.append((str(getattr(p, "name", "")), txt[:80]))
    exp = _run_py(check).strip().split()  # 闭式标准答案(可变长)
    return {
        "name": name,
        "objective": objective[:60],
        "goal_achieved": bool(getattr(result, "goal_achieved", None)),
        "surprise": getattr(result, "surprise_score", getattr(result, "surprise", None)),
        "n_phases": len(phases),
        "phase_names": [getattr(p, "name", "") for p in phases],
        "TRUE-EXEC (num+code marks in phase)": tool_hits,
        "RENAME marks": rename_hits,
        "closed_form_expected": " ".join(exp),
        "any_phase_bears_expected": any(e in txt for txt in [str(getattr(p, "result", "")) for p in phases] for e in exp),
    }


async def _main() -> None:
    ws_dir = Path("/tmp/huginn_aline_baseline")
    ws_dir.mkdir(parents=True, exist_ok=True)
    eng = AutoloopEngine(workspace=str(ws_dir), memory_manager=MemoryManager())
    rows = []
    for name, objective, chk in ALINE_OBJECTIVES:
        print(f"\n===== [{name}] 开始 =====", flush=True)
        rows.append(await _run_one(eng, name, objective, chk))
    out = ws_dir / "aline_baseline.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ALINE_BASELINE_RESULT ===")
    print(json.dumps(rows, ensure_ascii=False))
    print("=== END ===", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())