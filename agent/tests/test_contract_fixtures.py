"""T4: 同一道机器验收门, 两个真实任务夹具 (同门不同题).

ws35: 真实产物齐全且符合契约 → PASS;
ws34: 产物齐全, 但 loop_log 行 schema 与契约不符 → FAIL (不是存在性问题).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huginn.autoloop.deliverables import check_manifest, views_from_task_contract
from huginn.task_contract import load_contract

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _gate(name: str):
    root = FIXTURES / name
    task_contract = load_contract(root / "contract.json")
    manifest, contract, _static = views_from_task_contract(task_contract)
    return root, task_contract, manifest, contract


def test_ws35_real_artifacts_pass_the_gate() -> None:
    root, _tc, manifest, contract = _gate("ws35")
    report = check_manifest(root, manifest, contract)
    assert report["passed"], report
    assert report["contract"]["passed"], report["contract"]


def test_ws34_real_artifacts_fail_on_loop_log_schema() -> None:
    root, _tc, manifest, contract = _gate("ws34")
    report = check_manifest(root, manifest, contract)
    assert not report["passed"]
    # 文件都在: 失败只来自内容契约, 不是存在性/空壳.
    assert report["missing"] == [] and report["empty"] == [] and report["invalid"] == []
    assert report["contract"]["missing_fields"], report["contract"]
    assert all("loop_log.jsonl" in item for item in report["contract"]["missing_fields"])


def test_one_byte_change_in_contract_breaks_the_binding(tmp_path: Path) -> None:
    dst = tmp_path / "ws35"
    shutil.copytree(FIXTURES / "ws35", dst)
    path = dst / "contract.json"
    # 追加一个换行: 语义不变, 但契约字节指纹变了 → results.json 里旧 sha 对不上.
    path.write_bytes(path.read_bytes() + b"\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha != load_contract(FIXTURES / "ws35" / "contract.json").sha256

    manifest, contract, _static = views_from_task_contract(load_contract(path))
    report = check_manifest(dst, manifest, contract)
    assert not report["passed"]
    assert report["contract"]["bad_binding"], report["contract"]
