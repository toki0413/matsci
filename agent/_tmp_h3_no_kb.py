"""H3 真实 autoloop 3 轮, monkey-patch 禁用 KB 绕过 chromadb crash."""
from __future__ import annotations
import asyncio
import faulthandler
import json
import shutil
import time
from pathlib import Path

faulthandler.enable()

LOG = open("_tmp_h3_autoloop_no_kb.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

TOML_PATH = Path("huginn.toml")

def setup_toml(toggles_on: bool):
    BAK = Path("huginn.toml.bak.h3_nokb")
    if BAK.exists():
        shutil.copy(BAK, TOML_PATH)
    else:
        shutil.copy(TOML_PATH, BAK)
    if not toggles_on:
        return
    txt = TOML_PATH.read_text(encoding="utf-8")
    new_txt = txt.replace(
        "[feature_flags]",
        "[feature_flags]\nharness_prompt_patch = true\n"
        "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
    )
    TOML_PATH.write_text(new_txt, encoding="utf-8")

def restore_toml():
    BAK = Path("huginn.toml.bak.h3_nokb")
    if BAK.exists():
        shutil.copy(BAK, TOML_PATH)
        BAK.unlink()

# 清空 joint_beliefs
JB_DIR = Path(".huginn/joint_beliefs")
if JB_DIR.exists():
    shutil.rmtree(JB_DIR)
JB_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVES = [
    "Compare LDA vs GGA exchange-correlation functionals for silicon band gap: summarize literature consensus without running DFT",
    "Propose a diagnostic checklist for detecting k-point convergence issues in VASP calculations",
    "Identify three key methodological tradeoffs between pseudopotential and all-electron calculations for transition metal oxides",
]

async def run_one(idx: int, objective: str, toggles_on: bool) -> dict:
    setup_toml(toggles_on)
    t0 = time.time()
    try:
        # monkey-patch 禁用 KB, 绕过 chromadb native crash
        import huginn.autoloop.engine as eng_mod
        eng_mod.AutoloopEngine._get_kb = lambda self: None
        eng_mod.AutoloopEngine._build_kb_text = lambda self, query: ""
        log(f"[loop {idx}] toggles={'on' if toggles_on else 'off'}, KB disabled")

        from huginn.autoloop.engine import AutoloopEngine
        engine = AutoloopEngine(workspace=Path("."))
        result = await engine.run_cognitive(
            objective=objective,
            max_iterations=2,
            progressive_budget=False,
        )
        elapsed = time.time() - t0
        summary = {
            "idx": idx,
            "toggles": "on" if toggles_on else "off",
            "success": getattr(result, "success", False),
            "elapsed_s": round(elapsed, 1),
        }
        log(f"[loop {idx}] OK {summary}")
        return summary
    except Exception as e:
        elapsed = time.time() - t0
        summary = {
            "idx": idx,
            "toggles": "on" if toggles_on else "off",
            "success": False,
            "elapsed_s": round(elapsed, 1),
            "error": f"{type(e).__name__}: {e}",
        }
        log(f"[loop {idx}] FAIL {summary}")
        return summary
    finally:
        restore_toml()

async def main():
    results = []
    # 跑 3 轮 toggle on
    for i, obj in enumerate(OBJECTIVES, 1):
        r = await run_one(i, obj, toggles_on=True)
        results.append(r)

    # 验证 joint_beliefs 持久化
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

    n_success = sum(1 for r in results if r.get("success"))
    n_error = sum(1 for r in results if "error" in r)
    log(f"\n=== H3 端到端验证汇总 ===")
    log(f"3 轮 autoloop (toggle on, KB disabled): {n_success}/3 success, {n_error}/3 error")
    log(f"joint_beliefs/ 持久化: {len(jb_files)} files")
    if n_error == 0:
        log(f"3 轮不崩 ✅")
    else:
        log(f"⚠️  {n_error} 轮崩了")

try:
    asyncio.run(main())
finally:
    LOG.close()
