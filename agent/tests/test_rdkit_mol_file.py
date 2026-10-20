"""RDKitTool mol_file_to_mol (MOL/SDF/PDB 结构文件输入) 单元测试 — G-A 能力锚点.

输入方向 (结构文件 → 分子信息), 与 smiles_to_sdf 输出方向互补. 诚实边界:

- 不认识的后缀显式报错并列支持集, 不静默猜格式;
- SDF 多记录逐条解析, 单条坏记录不拖垮整批 (errors 带 record_index 定位),
  部分成功仍 success=True + errors 可见;
- 全部失败才 success=False (metadata.errors 保留定位信息).

fixture: ``tests/fixtures/structures/`` (ethanol.mol / two_records.sdf / mini.pdb).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pydantic")

from huginn.core_types import ToolContext
from huginn.tools.sci.rdkit_tool import RDKitInput, RDKitTool

FIXTURES = Path(__file__).parent / "fixtures" / "structures"

# 坏记录: counts 行声称 5 原子但只给 1 行 → MolFromMolBlock 判非法返回 None.
_BAD_RECORD = (
    "broken record\n"
    "  HuginnTest\n"
    "\n"
    "  5  4  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "M  END\n"
)


def _call(tmp_path: Path, **kwargs) -> object:
    tool = RDKitTool()
    return asyncio.run(
        tool.call(
            RDKitInput(**kwargs),
            ToolContext(session_id="test", workspace=str(tmp_path)),
        )
    )


def test_mol_file_parses_ethanol(tmp_path):
    result = _call(
        tmp_path, action="mol_file_to_mol", mol_file=str(FIXTURES / "ethanol.mol")
    )
    assert result.success is True
    data = result.data
    assert data["format"] == "mol"
    assert data["n_records"] == 1
    assert data["n_parsed"] == 1
    assert data["errors"] == []
    mol = data["molecules"][0]
    assert mol["formula"] == "C2H6O"
    assert mol["num_atoms"] == 9  # removeHs=False: 9 个显式原子全保留
    assert mol["num_heavy_atoms"] == 3
    assert mol["num_rings"] == 0
    assert mol["name"] == "ethanol"  # molblock 首行进 _Name


def test_mdl_alias_maps_to_mol(tmp_path):
    alias = tmp_path / "ethanol.mdl"
    shutil.copyfile(FIXTURES / "ethanol.mol", alias)
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=str(alias))
    assert result.success is True
    assert result.data["format"] == "mol"
    assert result.data["n_parsed"] == 1


def test_sdf_multi_record(tmp_path):
    result = _call(
        tmp_path, action="mol_file_to_mol", mol_file=str(FIXTURES / "two_records.sdf")
    )
    assert result.success is True
    data = result.data
    assert data["format"] == "sdf"
    assert data["n_records"] == 2
    assert data["n_parsed"] == 2
    assert data["errors"] == []
    assert [m["name"] for m in data["molecules"]] == ["ethanol", "methanol"]
    assert [m["formula"] for m in data["molecules"]] == ["C2H6O", "CH4O"]
    assert [m["record_index"] for m in data["molecules"]] == [0, 1]


def test_pdb_backbone_fragment(tmp_path):
    result = _call(
        tmp_path, action="mol_file_to_mol", mol_file=str(FIXTURES / "mini.pdb")
    )
    assert result.success is True
    data = result.data
    assert data["format"] == "pdb"
    assert data["n_parsed"] == 1
    assert data["molecules"][0]["num_atoms"] == 2  # N + CA 两原子


def test_pdb_ent_alias(tmp_path):
    alias = tmp_path / "mini.ent"
    shutil.copyfile(FIXTURES / "mini.pdb", alias)
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=str(alias))
    assert result.success is True
    assert result.data["format"] == "pdb"


def test_unsupported_suffix_names_supported_set(tmp_path):
    weird = tmp_path / "structure.xyz"
    weird.write_text("some coordinates\n", encoding="utf-8")
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=str(weird))
    assert result.success is False
    err = result.error or ""
    assert "Unsupported structure file format" in err
    assert ".xyz" in err
    assert ".sdf" in err  # 支持集必须列全, 不让用户猜


def test_missing_file_is_explicit(tmp_path):
    result = _call(
        tmp_path, action="mol_file_to_mol", mol_file=str(tmp_path / "nope.mol")
    )
    assert result.success is False
    assert "not found" in (result.error or "")


def test_mol_file_arg_is_required(tmp_path):
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=None)
    assert result.success is False
    assert "mol_file is required" in (result.error or "")


def test_sdf_partial_bad_record_keeps_good_ones(tmp_path):
    """单条坏记录不拖垮整批: 好记录照常解析, 坏记录进 errors 带 record_index."""
    good = (FIXTURES / "ethanol.mol").read_text(encoding="utf-8")
    sdf = tmp_path / "mixed.sdf"
    sdf.write_text(f"{good}$$$$\n{_BAD_RECORD}$$$$\n", encoding="utf-8")
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=str(sdf))
    assert result.success is True  # 部分成功不是失败
    data = result.data
    assert data["format"] == "sdf"
    assert data["n_records"] == 2
    assert data["n_parsed"] == 1
    assert data["molecules"][0]["formula"] == "C2H6O"
    assert [e["record_index"] for e in data["errors"]] == [1]


def test_sdf_all_records_bad_fails_explicitly(tmp_path):
    sdf = tmp_path / "all_bad.sdf"
    sdf.write_text(f"{_BAD_RECORD}$$$$\n", encoding="utf-8")
    result = _call(tmp_path, action="mol_file_to_mol", mol_file=str(sdf))
    assert result.success is False
    assert "No parseable molecule records" in (result.error or "")
    assert result.metadata["format"] == "sdf"
    assert len(result.metadata["errors"]) == 1
