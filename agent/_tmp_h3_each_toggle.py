"""Test each toggle individually to find which one crashes."""
from __future__ import annotations
import asyncio
import faulthandler
import shutil
import sys
import time
from pathlib import Path

faulthandler.enable()

LOG = open("_tmp_h3_each_detail.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

TOML_PATH = Path("huginn.toml")

def setup_toml(toggle_name):
    BAK = Path(f"huginn.toml.bak.{toggle_name}")
    if BAK.exists():
        shutil.copy(BAK, TOML_PATH)
    else:
        shutil.copy(TOML_PATH, BAK)
    if toggle_name == "off":
        return
    txt = TOML_PATH.read_text(encoding="utf-8")
    new_txt = txt.replace(
        "[feature_flags]",
        f"[feature_flags]\n{toggle_name} = true\n",
    )
    TOML_PATH.write_text(new_txt, encoding="utf-8")

def restore_toml(toggle_name):
    BAK = Path(f"huginn.toml.bak.{toggle_name}")
    if BAK.exists():
        shutil.copy(BAK, TOML_PATH)
        BAK.unlink()

async def run_one(toggle_name):
    setup_toml(toggle_name)
    log(f"\n=== toggle={toggle_name} ===")
    try:
        from huginn.autoloop.engine import AutoloopEngine
        t0 = time.time()
        engine = AutoloopEngine(workspace=Path("."))
        log(f"[{toggle_name}] engine ready, run_cognitive...")
        result = await engine.run_cognitive(
            objective="Compare LDA vs GGA functionals for Si band gap: literature consensus, no DFT",
            max_iterations=1,
            progressive_budget=False,
        )
        log(f"[{toggle_name}] OK {time.time()-t0:.1f}s success={getattr(result,'success',False)}")
    finally:
        restore_toml(toggle_name)

# 跑 4 次: off / H1 / H2 / H3
for tog in ["off", "harness_prompt_patch", "harness_workflow_evolution", "harness_joint_optimizer"]:
    try:
        asyncio.run(run_one(tog))
    except Exception as e:
        log(f"[{tog}] EXCEPTION: {type(e).__name__}: {e}")

LOG.close()
