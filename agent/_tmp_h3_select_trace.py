"""H3 验证: trace select_block_subset_for_phase + select_workflow_params_for_stage 是否被调到.

action_history 显示 cognitive_loop 走了 hypothesize→plan→execute→validate,
H3 select helper 应该在 hypothesize/plan (H1 apply_patches) 和 execute (H2 _perturb_script) 触发.
_learn 没被调是 LLM 选了 pivot, 不是 H3 bug.
"""
from __future__ import annotations
import asyncio
import faulthandler
import json
import os
import shutil
import time
from pathlib import Path

faulthandler.enable()
os.environ["HUGINN_CACHE_DIR"] = str(Path(".huginn").resolve())

LOG = open("_tmp_h3_select_trace.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()

TOML_PATH = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_strace")
shutil.copy(TOML_PATH, BAK)
txt = TOML_PATH.read_text(encoding="utf-8")
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML_PATH.write_text(new_txt, encoding="utf-8")
log("[setup] 3 toggles on")

JB_DIR = Path(".huginn/joint_beliefs")
if JB_DIR.exists():
    shutil.rmtree(JB_DIR)
JB_DIR.mkdir(parents=True, exist_ok=True)

try:
    import huginn.autoloop.engine as eng_mod
    import huginn.harness.joint_optimizer as jo_mod
    import huginn.harness.prompt_patch as pp_mod
    import huginn.autoloop.variant_gen as vg_mod
    eng_mod.AutoloopEngine._get_kb = lambda self: None
    eng_mod.AutoloopEngine._build_kb_text = lambda self, query: ""

    # trace H3 select helpers
    _orig_select_blocks = jo_mod.select_block_subset_for_phase
    def _traced_select_blocks(phase, blocks):
        names = [n for n, _ in blocks]
        out = _orig_select_blocks(phase, blocks)
        out_names = [n for n, _ in out]
        log(f"[H3] select_block_subset_for_phase('{phase}', {names}) -> {out_names}")
        return out
    jo_mod.select_block_subset_for_phase = _traced_select_blocks
    # apply_patches 里 from import, 要 patch pp_mod 命名空间
    pp_mod.select_block_subset_for_phase = _traced_select_blocks

    _orig_select_params = jo_mod.select_workflow_params_for_stage
    def _traced_select_params(stage, defaults):
        out = _orig_select_params(stage, defaults)
        log(f"[H3] select_workflow_params_for_stage('{stage}', {defaults}) -> {out}")
        return out
    jo_mod.select_workflow_params_for_stage = _traced_select_params
    vg_mod.select_workflow_params_for_stage = _traced_select_params

    # patch CognitiveLoop.run 记录 action_history
    import huginn.autoloop.cognitive_loop as cl_mod
    _orig_run = cl_mod.CognitiveLoop.run
    async def _traced_run(self, initial_state=None):
        state = await _orig_run(self, initial_state)
        log(f"[loop] action_history = {state.action_history}")
        return state
    cl_mod.CognitiveLoop.run = _traced_run

    from huginn.autoloop.engine import AutoloopEngine
    log("[run] engine ready, max_iterations=8")

    async def run_one():
        t0 = time.time()
        engine = AutoloopEngine(workspace=Path("."))
        result = await engine.run_cognitive(
            objective="Propose a diagnostic checklist for detecting k-point convergence issues in VASP calculations",
            max_iterations=8,
            progressive_budget=False,
        )
        log(f"[run] done {time.time()-t0:.1f}s success={getattr(result,'success',False)}")

    asyncio.run(run_one())

    jb_files = list(JB_DIR.glob("*.json"))
    log(f"\n[verify] joint_beliefs/ has {len(jb_files)} files")

finally:
    shutil.copy(BAK, TOML_PATH)
    BAK.unlink()
    log("[teardown] toml restored")
    LOG.close()
