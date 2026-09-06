"""Huginn 能力演示 — 缺度追问 + 对象级取证 + 门禁闭环 (零网络/零凭据/确定性).

这是 Huginn 区别于"只会检索"的 agent 的核心叙事的可独立复现 demo:

  1. 局部化   : 把跨文献报道值按真实物理自由度 (方法族/泛函/温度) 分组, 而非
                笼统塞进同一个 "eV" 桶。
  2. 缺度追问 : 暴露"哪些自由度没定义" (缺温度/泛函/方法族), 生成补全查询。
  3. 对象级取证: 每条值绑定来源论文的强引用证据标签 (fid + sha256 快照), 可独立核实。
  4. 门禁不变量: 补后仍缺的关键自由度不再 silent 接受 —— 落显式豁免决策档 (哈希链),
                由 assess_gate 判能否放行 (pass / pass_with_waiver / needs_waiver)。

运行:
    cd agent && python ../examples/demo_evidence_chain.py
    # 或:  python examples/demo_evidence_chain.py  (从仓库根)

纯库级 Python, 不需要 API key, 不需要网络, 输出确定性可重现。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能从仓库根或 agent/ 两个位置运行都找得到包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from huginn.experimental.local_global_compat import run_compat_experiment
from huginn.tools.literature.completion_evidence import (
    assess_gate,
    attach_evidence,
    build_waivers,
    critical_missing,
)
from huginn.tools.literature.query_completion import _URGENCY


# ── 合成的文献报道值: 同一体系 Li2O band_gap, 但方法自由度故意混杂 ──
# PBE@0K 系统性低估带隙; 实验@室温; HSE06 (更准的杂化泛函).
# 一条故意缺 method/note → 缺失自由度.
def _synthetic_reported() -> list[dict]:
    rows = [
        {"value": 4.10, "unit": "eV", "method": "DFT-PBE", "note": "", "doi": "10.1000/pbe"},
        {"value": 4.05, "unit": "eV", "method": "DFT-PBE", "note": "", "doi": "10.1000/pbe2"},
        {"value": 5.81, "unit": "eV", "method": "experiment", "note": "room temperature", "doi": "10.1000/exp"},
        {"value": 5.79, "unit": "eV", "method": "experiment", "note": "room temperature", "doi": "10.1000/exp2"},
        {"value": 5.98, "unit": "eV", "method": "HSE06", "note": "", "doi": "10.1000/hse"},
        {"value": 6.02, "unit": "eV", "method": "HSE06", "note": "", "doi": "10.1000/hse2"},
        {"value": 47.5, "unit": "GPa", "method": "", "note": "", "doi": "10.1000/shear"},  # 缺自由度
    ]
    # 附原文 (供对象级取证 sha256 快照)
    papers = [
        {"doi": "10.1000/pbe", "title": "PBE gap", "abstract": "GGA gives 4.10 eV for Li2O."},
        {"doi": "10.1000/pbe2", "title": "PBE gap 2", "abstract": "PBE band gap about 4.05 eV."},
        {"doi": "10.1000/exp", "title": "Exp gap", "abstract": "Measured 5.81 eV at room temperature."},
        {"doi": "10.1000/exp2", "title": "Exp gap 2", "abstract": "5.79 eV optical gap."},
        {"doi": "10.1000/hse", "title": "HSE gap", "abstract": "HSE06 predicts 5.98 eV."},
        {"doi": "10.1000/hse2", "title": "HSE gap 2", "abstract": "Hybrid functional yields 6.02 eV."},
        {"doi": "10.1000/shear", "title": "Shear mod", "abstract": "Shear modulus."},
    ]
    return attach_evidence(rows, papers)


def main() -> int:
    reported = _synthetic_reported()

    # 1. 局部-整体兼容: 分组 + 缺度暴露
    compat = run_compat_experiment(reported)
    missing = compat["missing_dims"]

    print("=" * 70)
    print("HUGINN — 缺度追问 + 对象级取证 + 门禁闭环 演示")
    print("=" * 70)
    print("\n[1] 局部化: 按物理自由度分组 (每个条件组内部是否收敛)")
    print(f"{'条件组':<28}{'n':<4}{'median':<10}{'组内判定':<18}{'缺自由度'}")
    for key, st in sorted(compat["local"].items()):
        miss = ','.join(st["missing_dims"]) or "无"
        print(f"{key:<28}{st['n_sources']:<4}{st['median']:<10.4f}"
              f"{st['consistency']['overall']['verdict']:<18}{miss}")

    print("\n[2] 缺度追问: 暴露未定义的物理自由度 → 生成补全查询")
    from huginn.tools.literature.query_completion import completion_query

    for dim in sorted(missing):
        q = completion_query([dim], "Li2O", "band gap", "eV")
        urgent = _URGENCY.get(dim, 0)
        shown = q[0]["query"] if q else "(已满足)"
        print(f"  - 缺 {dim:<16} (urgency={urgent}) -> {shown}")

    print("\n[3] 对象级取证: 每条值绑定来源论文强引用证据 (fid + sha256 快照)")
    for r in reported[:4]:
        ev = r.get("evidence") or {}
        print(f"  - {r.get('doi')!s:<18} value={r['value']:<6} fid={ev.get('fid','')} sha256={str(ev.get('sha256'))[:12]}…")

    # 4. 门禁不变量: 补后仍缺的关键自由度 → 显式豁免决策档 + 门禁放行
    blocking = critical_missing(missing, _URGENCY)
    waivers = build_waivers(blocking, reason_var="demo")
    waived = [w["dim"] for w in waivers]
    gate = assess_gate(blocking, waived, verdict=compat["overall_verdict"])

    print("\n[4] 门禁不变量: 补后仍缺的关键自由度不再 silent 接受")
    print(f"  - 关键缺度(urgency>=2): {blocking or '无'}")
    if waivers:
        for w in waivers:
            print(f"  - 豁免决策 {w['decision_id']}: dim={w['dim']} kind={w['kind']} "
                  f"decision={w['decision']}")
    print(f"  - 门禁状态: {gate['status']}  verdict={gate['verdict']}")
    if gate["issues"]:
        for issue in gate["issues"]:
            print(f"      ⚠ {issue}")

    print("\n[5] 判别对比: 若不分自由直接全塞一起, 会被误判成什么")
    print(f"  未分组 verdict = {compat['flat_confounding']['overall']['verdict']}  "
          f"(分组后能归因为方法/温度差异, 而非数值冲突)")

    print("\n" + "=" * 70)
    print("关键结论: Huginn 不假装一致, 也不捏造自由度。")
    print("Pass 条件: 分组内收敛 + 组间差异归因于缺省自由度 + 缺度补全取证。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())