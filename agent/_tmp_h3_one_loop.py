"""Minimal: 1 loop, 1 iteration, toggle on."""
from __future__ import annotations
import asyncio
import shutil
import time
from pathlib import Path

TOML = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_one")
shutil.copy(TOML, BAK)
txt = TOML.read_text(encoding="utf-8")
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML.write_text(new_txt, encoding="utf-8")
print("[setup] toggles on", flush=True)

# 清空 joint_beliefs
JB_DIR = Path(".huginn/joint_beliefs")
if JB_DIR.exists():
    shutil.rmtree(JB_DIR)
JB_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] {JB_DIR} cleared", flush=True)

try:
    from huginn.autoloop.engine import AutoloopEngine
    print("[import] OK", flush=True)

    async def run_one():
        t0 = time.time()
        engine = AutoloopEngine(workspace=Path("."))
        print("[run] engine ready, calling run_cognitive...", flush=True)
        result = await engine.run_cognitive(
            objective="Compare LDA vs GGA exchange-correlation functionals for silicon band gap: summarize literature consensus without running DFT",
            max_iterations=1,
            progressive_budget=False,
        )
        print(f"[run] done in {time.time()-t0:.1f}s, success={getattr(result,'success',False)}", flush=True)

    asyncio.run(run_one())
    print("[done] OK", flush=True)
finally:
    shutil.copy(BAK, TOML)
    BAK.unlink()
    print("[teardown] toml restored", flush=True)
