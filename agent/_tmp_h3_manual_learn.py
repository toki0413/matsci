"""H3 最小验证: 手动调 _learn, 验证 record_joint_outcome 真能写文件.

绕过 cognitive_loop 调度, 直接构造 hypothesis/plan/validation, 调 _learn.
如果 joint_beliefs/ 有文件 → H3 record OK, 端到端 0 files 是 cognitive_loop 没跑 learn.
如果仍 0 files → H3 record 代码有 bug.
"""
from __future__ import annotations
import asyncio
import faulthandler
import json
import shutil
from pathlib import Path

faulthandler.enable()

LOG = open("_tmp_h3_manual_learn.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()

TOML_PATH = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_manual")
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
    # 绕开 chromadb crash
    eng_mod.AutoloopEngine._get_kb = lambda self: None
    eng_mod.AutoloopEngine._build_kb_text = lambda self, query: ""

    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine(workspace=Path("."))

    # 构造最小 hypothesis/plan/validation
    hyp = "Test hypothesis for H3 manual learn verification"
    plan = {"mode": "test", "description": "manual", "subtasks": []}
    validation = {"tests_passed": True, "reviewer_critique": "manual"}

    # 模拟 _apply_block_patches 设置 _last_hypothesis_blocks
    # 格式: [(block_name, block_content), ...]
    eng._last_hypothesis_blocks = [
        ("body", "test body content"),
        ("context", "test context"),
        ("fail", "test failure mode"),
    ]
    log(f"[setup] _last_hypothesis_blocks = {[n for n,_ in eng._last_hypothesis_blocks]}")

    asyncio.run(eng._learn(hyp, plan, validation))

    jb_files = list(JB_DIR.glob("*.json"))
    log(f"\n[verify] joint_beliefs/ has {len(jb_files)} files")
    for f in jb_files[:5]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            log(f"  {f.name}: phase={d.get('phase','?')} "
                f"successes={d.get('successes',0)} failures={d.get('failures',0)} "
                f"mean={d.get('posterior_mean',0):.2f} "
                f"subset={d.get('block_subset','?')}")
        except Exception as e:
            log(f"  {f.name}: parse fail {e}")

    # 再调一次失败 outcome, 看 Beta 分化
    eng._last_hypothesis_blocks = [
        ("body", "test body content"),
        ("context", "test context"),
    ]
    log(f"\n[setup2] _last_hypothesis_blocks = {[n for n,_ in eng._last_hypothesis_blocks]}")
    validation2 = {"tests_passed": False, "reviewer_critique": "manual fail"}
    asyncio.run(eng._learn(hyp, plan, validation2))
    jb_files2 = list(JB_DIR.glob("*.json"))
    log(f"\n[verify2] joint_beliefs/ has {len(jb_files2)} files")
    for f in jb_files2[:5]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            log(f"  {f.name}: phase={d.get('phase','?')} "
                f"successes={d.get('successes',0)} failures={d.get('failures',0)} "
                f"mean={d.get('posterior_mean',0):.2f} "
                f"subset={d.get('block_subset','?')}")
        except Exception as e:
            log(f"  {f.name}: parse fail {e}")

finally:
    shutil.copy(BAK, TOML_PATH)
    BAK.unlink()
    log("[teardown] toml restored")
    LOG.close()
