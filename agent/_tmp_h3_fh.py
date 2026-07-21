"""1 loop, 1 iter, toggle on, with faulthandler to catch native crash."""
from __future__ import annotations
import asyncio
import faulthandler
import shutil
import sys
import time
from pathlib import Path

# 启用 faulthandler, crash 时 dump Python stack 到 stderr
faulthandler.enable()
faulthandler.dump_traceback_later(30, exit=False)  # 30s 后 dump 一次 stack

TOML = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_fh")
shutil.copy(TOML, BAK)
txt = TOML.read_text(encoding="utf-8")
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML.write_text(new_txt, encoding="utf-8")
print("[setup] toggles on", flush=True)

JB_DIR = Path(".huginn/joint_beliefs")
if JB_DIR.exists():
    shutil.rmtree(JB_DIR)
JB_DIR.mkdir(parents=True, exist_ok=True)

try:
    from huginn.autoloop.engine import AutoloopEngine
    print("[import] OK", flush=True)

    # patch _apply_block_patches 加 trace
    import huginn.autoloop.engine as eng_mod
    _orig_abp = eng_mod.AutoloopEngine._apply_block_patches
    def _traced_abp(self, blocks, phase):
        print(f"[trace] _apply_block_patches phase={phase} n_blocks={len(blocks)}", flush=True)
        out = _orig_abp(self, blocks, phase)
        print(f"[trace] _apply_block_patches done n_out={len(out)}", flush=True)
        return out
    eng_mod.AutoloopEngine._apply_block_patches = _traced_abp

    async def run_one():
        t0 = time.time()
        engine = AutoloopEngine(workspace=Path("."))
        print("[run] engine ready", flush=True)
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
