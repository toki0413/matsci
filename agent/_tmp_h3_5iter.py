"""H3 1 轮 autoloop max_iterations=5, KB disabled, trace _learn."""
from __future__ import annotations
import asyncio
import faulthandler
import json
import os
import shutil
import time
from pathlib import Path

faulthandler.enable()

# 让 JointBandit 写到 workspace 的 .huginn, 跟测试脚本查询路径一致
os.environ["HUGINN_CACHE_DIR"] = str(Path(".huginn").resolve())

LOG = open("_tmp_h3_5iter.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

TOML_PATH = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_5iter")
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
    # monkey-patch KB
    eng_mod.AutoloopEngine._get_kb = lambda self: None
    eng_mod.AutoloopEngine._build_kb_text = lambda self, query: ""
    # trace _learn
    _orig_learn = eng_mod.AutoloopEngine._learn
    async def _traced_learn(self, hypothesis, plan, validation):
        log(f"[trace] _learn called, validation keys={list(validation.keys()) if isinstance(validation, dict) else '?'}")
        await _orig_learn(self, hypothesis, plan, validation)
        log(f"[trace] _learn done, joint_beliefs/ has {len(list(JB_DIR.glob('*.json')))} files")
    eng_mod.AutoloopEngine._learn = _traced_learn

    from huginn.autoloop.engine import AutoloopEngine
    log("[run] engine ready, max_iterations=5")

    async def run_one():
        t0 = time.time()
        engine = AutoloopEngine(workspace=Path("."))
        result = await engine.run_cognitive(
            objective="Propose a diagnostic checklist for detecting k-point convergence issues in VASP calculations",
            max_iterations=5,
            progressive_budget=False,
        )
        log(f"[run] done {time.time()-t0:.1f}s success={getattr(result,'success',False)}")

    asyncio.run(run_one())

    jb_files = list(JB_DIR.glob("*.json"))
    log(f"\n[verify] joint_beliefs/ has {len(jb_files)} files")
    for f in jb_files[:5]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            log(f"  {f.name}: phase={d.get('phase','?')} "
                f"α={d.get('successes',0)} β={d.get('failures',0)} "
                f"mean={d.get('posterior_mean',0):.2f}")
        except Exception:
            log(f"  {f.name}: parse fail")
finally:
    shutil.copy(BAK, TOML_PATH)
    BAK.unlink()
    log("[teardown] toml restored")
    LOG.close()
