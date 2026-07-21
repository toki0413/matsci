"""Minimal: setup toml + import engine, no loop."""
from __future__ import annotations
import shutil
from pathlib import Path

TOML = Path("huginn.toml")
BAK = Path("huginn.toml.bak.h3_test")
shutil.copy(TOML, BAK)
print("[1] backup done")

txt = TOML.read_text(encoding="utf-8")
new_txt = txt.replace(
    "[feature_flags]",
    "[feature_flags]\nharness_prompt_patch = true\n"
    "harness_workflow_evolution = true\nharness_joint_optimizer = true\n",
)
TOML.write_text(new_txt, encoding="utf-8")
print("[2] toml patched with 3 toggles")

try:
    print("[3] importing AutoloopEngine...")
    from huginn.autoloop.engine import AutoloopEngine
    print("[4] import OK")
    engine = AutoloopEngine(workspace=Path("."))
    print("[5] engine instance OK")
finally:
    shutil.copy(BAK, TOML)
    BAK.unlink()
    print("[6] toml restored")
