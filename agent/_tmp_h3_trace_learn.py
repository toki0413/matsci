"""Trace _learn 内部每一步, 找 H3 record 没触发的原因."""
from __future__ import annotations
import asyncio
import faulthandler
import json
import shutil
from pathlib import Path

faulthandler.enable()

LOG = open("_tmp_h3_trace_learn.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()

TOML_PATH = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_trace")
shutil.copy(TOML_PATH, BAK)
txt = TOML_PATH.read_text(encoding="utf-8")
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML_PATH.write_text(new_txt, encoding="utf-8")
log("[setup] 3 toggles on")

try:
    import huginn.autoloop.engine as eng_mod
    eng_mod.AutoloopEngine._get_kb = lambda self: None
    eng_mod.AutoloopEngine._build_kb_text = lambda self, query: ""

    # 验证 toggle 真生效
    from huginn.config import get_config
    cfg = get_config()
    ff = getattr(cfg, "feature_flags", None) or {}
    log(f"[verify] toggle harness_joint_optimizer = {ff.get('harness_joint_optimizer')}")
    log(f"[verify] toggle harness_prompt_patch = {ff.get('harness_prompt_patch')}")

    from huginn.harness.joint_optimizer import _harness_enabled
    log(f"[verify] _harness_enabled('harness_joint_optimizer') = {_harness_enabled('harness_joint_optimizer')}")

    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine(workspace=Path("."))

    # 先清掉 home 的 joint_beliefs
    home_jb = Path.home() / ".huginn" / "joint_beliefs"
    if home_jb.exists():
        shutil.rmtree(home_jb)
    log(f"[setup] cleared home joint_beliefs: {home_jb}")

    eng._last_hypothesis_blocks = [
        ("body", "test body"),
        ("context", "test ctx"),
        ("fail", "test fail"),
    ]
    log(f"[setup] eng._last_hypothesis_blocks = {[n for n,_ in eng._last_hypothesis_blocks]}")

    hyp = "test hyp"
    plan = {"mode": "test", "description": "manual", "subtasks": []}
    validation = {"tests_passed": True, "reviewer_critique": "manual"}

    # 直接调 JointBandit.record_joint_outcome 看 home 是否有文件
    from huginn.harness.joint_optimizer import JointBandit
    jb = JointBandit.get_instance()
    log(f"[verify] JointBandit store_dir = {jb._store_dir}")
    jb.record_joint_outcome("manual_test", ["body"], {}, True)
    log(f"[verify] after direct record_joint_outcome, home_jb files = {list(home_jb.glob('*.json')) if home_jb.exists() else 'NO DIR'}")

    # 现在调 _learn
    log("\n[run] calling eng._learn(...)")
    try:
        asyncio.run(eng._learn(hyp, plan, validation))
        log("[run] _learn returned without exception")
    except Exception as e:
        log(f"[run] _learn raised: {type(e).__name__}: {e}")

    # 检查 home joint_beliefs
    if home_jb.exists():
        files = list(home_jb.glob("*.json"))
        log(f"\n[verify] home_jb has {len(files)} files")
        for f in files:
            log(f"  {f.name}")
    else:
        log("\n[verify] home_jb does not exist")

finally:
    shutil.copy(BAK, TOML_PATH)
    BAK.unlink()
    log("[teardown] toml restored")
    LOG.close()
