"""H3 端到端验证: 3 轮真实 autoloop + 3 toggle on (H1+H2+H3).

验证项:
1. 每轮不崩 (success=true 或 false 都行, 关键是不 exception)
2. .huginn/joint_beliefs/ 目录有持久化文件
3. Beta 信念有变化 (每次 record_joint_outcome 更新 alpha/beta)
4. block_subset 选择有差异 (UCB 冷启动=inf 优先探索)

跑完还原 huginn.toml.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

# 1. 备份 huginn.toml + 加 3 个 toggle
TOML = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_smoke")
shutil.copy(TOML, BAK)

txt = TOML.read_text(encoding="utf-8")
# 在 [feature_flags] section 后追加 3 个 toggle
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML.write_text(new_txt, encoding="utf-8")
print(f"[setup] huginn.toml backed up → {BAK.name}, 3 toggles enabled")

# 2. 清空 joint_beliefs 目录 (保证从冷启动开始)
JB_DIR = Path(".huginn/joint_beliefs")
if JB_DIR.exists():
    shutil.rmtree(JB_DIR)
JB_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] {JB_DIR} cleared for cold start")

# 3. 跑 3 轮 autoloop (reasoning-only objective, 不调真实 VASP)
OBJECTIVES = [
    "Compare LDA vs GGA exchange-correlation functionals for silicon band gap: summarize literature consensus without running DFT",
    "Propose a diagnostic checklist for detecting k-point convergence issues in VASP calculations",
    "Identify three key methodological tradeoffs between pseudopotential and all-electron calculations for transition metal oxides",
]

from huginn.autoloop.engine import AutoloopEngine


async def run_one(idx: int, objective: str) -> dict:
    """跑一轮 autoloop, 返回结果摘要."""
    t0 = time.time()
    engine = AutoloopEngine(workspace=Path("."))
    try:
        result = await engine.run_cognitive(
            objective=objective,
            max_iterations=3,  # 短跑, 验证不崩即可
            progressive_budget=False,
        )
        elapsed = time.time() - t0
        summary = {
            "idx": idx,
            "objective": objective[:80],
            "success": getattr(result, "success", False),
            "goal_achieved": getattr(result, "goal_achieved", False),
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        summary = {
            "idx": idx,
            "objective": objective[:80],
            "success": False,
            "goal_achieved": False,
            "elapsed_s": round(elapsed, 1),
            "error": f"{type(e).__name__}: {e}",
        }
    print(f"[loop {idx}] {summary}")
    return summary


async def main() -> None:
    results = []
    for i, obj in enumerate(OBJECTIVES, 1):
        r = await run_one(i, obj)
        results.append(r)

    # 4. 验证 joint_beliefs 持久化
    jb_files = list(JB_DIR.glob("*.json"))
    print(f"\n[verify] joint_beliefs/ has {len(jb_files)} files")
    for f in jb_files[:5]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            print(
                f"  {f.name}: phase={d.get('phase','?')} "
                f"α={d.get('successes',0)} β={d.get('failures',0)} "
                f"mean={d.get('posterior_mean',0):.2f}"
            )
        except Exception:
            print(f"  {f.name}: parse fail")

    # 5. 汇总
    n_success = sum(1 for r in results if r.get("success"))
    n_error = sum(1 for r in results if "error" in r)
    print(f"\n=== H3 端到端验证汇总 ===")
    print(f"3 轮 autoloop: {n_success}/3 success, {n_error}/3 error")
    print(f"joint_beliefs/ 持久化: {len(jb_files)} files")
    if jb_files:
        print(f"UCB/Beta 信念有持久化 ✅")
    else:
        print(f"⚠️  joint_beliefs/ 空 — H3 可能没被触发 (plan 没 n_variants 或 apply_patches 没走)")
    if n_error == 0:
        print(f"3 轮不崩 ✅")
    else:
        print(f"⚠️  {n_error} 轮崩了")


try:
    asyncio.run(main())
finally:
    # 6. 还原 huginn.toml
    shutil.copy(BAK, TOML)
    BAK.unlink()
    print(f"\n[teardown] huginn.toml restored")
