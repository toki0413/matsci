"""Extended tests for structure_tool.py — 覆盖 validate_input / call /
_handle_batch_validate / _call_cached / _local_to_output 全部分支.

structure_tool 的 pymatgen 主路径需要真实 numpy (coverage 下 fake numpy
无法 import pymatgen), 测试通过注入 fake pymatgen 模块到 sys.modules 来
覆盖 import 成功 / SpacegroupAnalyzer 成功/异常 / ImportError fallback.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from huginn.core_types import ToolContext
from huginn.tools.structure_tool import (
    StructureTool,
    StructureToolInput,
    _file_mtime,
    _local_to_output,
)


def _ctx(workspace="/tmp"):
    return ToolContext(session_id="test", workspace=workspace)


# ── Input schema model_validator ─────────────────────────────────────────────

class TestStructureToolInput:
    def test_batch_validate_requires_files(self):
        with pytest.raises(ValueError):
            StructureToolInput(action="batch_validate")

    def test_other_action_requires_file_path(self):
        with pytest.raises(ValueError):
            StructureToolInput(action="read")

    def test_batch_validate_with_files_ok(self):
        inp = StructureToolInput(action="batch_validate", files=["a", "b"])
        assert inp.files == ["a", "b"]

    def test_read_with_file_path_ok(self):
        inp = StructureToolInput(action="read", file_path="/tmp/x")
        assert inp.file_path == "/tmp/x"


# ── is_read_only ─────────────────────────────────────────────────────────────

class TestReadOnly:
    def test_is_read_only_true(self):
        tool = StructureTool()
        assert tool.is_read_only(StructureToolInput(action="read", file_path="/tmp/x")) is True


# ── _file_mtime / _local_to_output ───────────────────────────────────────────

class TestHelpers:
    def test_file_mtime_exists(self, tmp_path):
        f = tmp_path / "a"
        f.write_text("x")
        assert _file_mtime(str(f)) > 0

    def test_file_mtime_missing(self, tmp_path):
        assert _file_mtime(str(tmp_path / "nope")) == 0.0

    def test_local_to_output_full(self):
        struct = {
            "lattice_params": {"a": 5.4, "b": 5.4, "c": 5.4,
                               "alpha": 90, "beta": 90, "gamma": 90},
            "atomic_positions": {"num_sites": 8},
            "formula_pretty": "Si",
            "space_group": "Fd-3m",
            "volume": 157.5,
            "density": 2.33,
            "mp_id": "mp-149",
        }
        out = _local_to_output(struct)
        assert out["formula"] == "Si"
        assert out["num_atoms"] == 8
        assert out["warnings"] == ["from local structure db (mp-149)"]

    def test_local_to_output_fallback_fields(self):
        # lattice_params 必须 6 个全给 (pydantic 字段是 dict[str, float],
        # 缺 key 会塞 None 触发校验失败); 这里测的是 formula/num_atoms 缺省回退.
        struct = {
            "lattice_params": {"a": 5.4, "b": 5.4, "c": 5.4,
                               "alpha": 90, "beta": 90, "gamma": 90},
            "atomic_positions": {},
            "space_group": "Fm-3m",
        }
        out = _local_to_output(struct)
        assert out["formula"] is None
        assert out["num_atoms"] is None
        assert out["lattice_params"]["a"] == 5.4


# ── validate_input ───────────────────────────────────────────────────────────

class TestValidateInput:
    @pytest.mark.asyncio
    async def test_batch_validate_passes(self):
        tool = StructureTool()
        vr = await tool.validate_input(
            StructureToolInput(action="batch_validate", files=["a"]), _ctx()
        )
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_missing_file_hits_local_db(self, tmp_path):
        tool = StructureTool()
        # 用本地库里的真实条目 (mp-149) 验证 "文件不存在但库命中" 分支
        vr = await tool.validate_input(
            StructureToolInput(action="read", file_path="mp-149"), _ctx()
        )
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_missing_file_and_not_in_db(self):
        tool = StructureTool()
        vr = await tool.validate_input(
            StructureToolInput(action="read", file_path="/nonexistent/xyz.cif"),
            _ctx(),
        )
        assert vr.result is False
        assert vr.error_code == 404

    @pytest.mark.asyncio
    async def test_existing_file_passes(self, tmp_path):
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si\n")
        vr = await tool.validate_input(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_bad_reference_path(self, tmp_path):
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si\n")
        vr = await tool.validate_input(
            StructureToolInput(
                action="compare",
                file_path=str(f),
                reference_path="/nonexistent/ref.cif",
            ),
            _ctx(str(tmp_path)),
        )
        assert vr.result is False
        assert vr.error_code == 404

    @pytest.mark.asyncio
    async def test_valid_reference_path(self, tmp_path):
        # vr2.result True 分支: reference 文件存在 → 直接放行
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si\n")
        ref = tmp_path / "ref.cif"
        ref.write_text("x\n")
        vr = await tool.validate_input(
            StructureToolInput(
                action="compare",
                file_path=str(f),
                reference_path=str(ref),
            ),
            _ctx(str(tmp_path)),
        )
        assert vr.result is True


# ── call ─────────────────────────────────────────────────────────────────────

class TestCall:
    @pytest.mark.asyncio
    async def test_call_local_db_hit(self):
        tool = StructureTool()
        result = await tool.call(
            StructureToolInput(action="read", file_path="mp-149"), _ctx()
        )
        assert result.success
        assert result.data["formula"] == "Si"

    @pytest.mark.asyncio
    async def test_call_path_dispatch_to_cached(self, tmp_path):
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("a\nb\nc\nd\ne\nf\ng\nh\n")

        class _R:
            success = True
            data = {"ok": 1}

        async def fake_call(args, ctx):
            return _R()

        with patch.object(tool, "_call_cached", new_callable=MagicMock) as m:
            m.side_effect = fake_call
            result = await tool.call(
                StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
            )
        m.assert_called_once()
        assert result.success

    @pytest.mark.asyncio
    async def test_call_batch_validate_dispatch(self, tmp_path):
        # call() 里 batch_validate 走单独路径 (line 134), 不查本地库
        tool = StructureTool()
        with patch.object(tool, "_handle_batch_validate", new_callable=MagicMock) as m:
            async def fake(args, ctx):
                return type("R", (), {"success": True,
                                      "data": {"action": "batch_validate"}})()
            m.side_effect = fake
            result = await tool.call(
                StructureToolInput(action="batch_validate", files=["a"]),
                _ctx(str(tmp_path)),
            )
        m.assert_called_once()
        assert result.success
        assert result.data["action"] == "batch_validate"


# ── batch_validate ───────────────────────────────────────────────────────────

class TestBatchValidate:
    @pytest.mark.asyncio
    async def test_empty_files_list_schema_blocked(self):
        # model_validator 拦截空 files (已在 TestStructureToolInput 覆盖), 空分支死代码
        with pytest.raises(ValueError):
            StructureToolInput(action="batch_validate", files=[])

    @pytest.mark.asyncio
    async def test_empty_files_early_return(self, tmp_path):
        # model_construct 绕过 model_validator, 直击 _handle_batch_validate
        # 内部 "files 为空" 的 early return (line 167); 生产路径被 Literal/校验守护.
        tool = StructureTool()
        args = StructureToolInput.model_construct(action="batch_validate")
        result = await tool._handle_batch_validate(args, _ctx(str(tmp_path)))
        assert not result.success
        assert "requires non-empty" in result.error

    @pytest.mark.asyncio
    async def test_dedup_and_mixed(self, tmp_path):
        tool = StructureTool()
        good = tmp_path / "good.cif"
        good.write_text("a\nb\nc\nd\ne\nf\n")
        missing = tmp_path / "missing.cif"

        with patch.object(tool, "_call_cached", new_callable=MagicMock) as m:
            async def fake_call(args, ctx):
                if args.file_path == str(good):
                    return type("R", (), {"success": True, "data": {"formula": "Si"}})()
                return type("R", (), {"success": False, "error": "boom"})()
            m.side_effect = fake_call

            result = await tool._handle_batch_validate(
                StructureToolInput(
                    action="batch_validate",
                    files=[str(good), str(missing), str(good)],  # 重复
                ),
                _ctx(str(tmp_path)),
            )

        assert result.success
        assert result.data["total"] == 2  # 去重后
        assert result.data["valid"] == 1
        assert result.data["invalid"] == 1
        by_file = {r["file"]: r for r in result.data["results"]}
        assert by_file[str(good)]["valid"] is True
        assert by_file[str(missing)]["valid"] is False
        assert by_file[str(missing)]["error"] == "boom"

    @pytest.mark.asyncio
    async def test_exception_in_validate_one(self, tmp_path):
        tool = StructureTool()
        bad = tmp_path / "bad.cif"
        bad.write_text("x")

        with patch.object(tool, "_call_cached", new_callable=MagicMock) as m:
            m.side_effect = RuntimeError("parse exploded")
            result = await tool._handle_batch_validate(
                StructureToolInput(
                    action="batch_validate", files=[str(bad)]
                ),
                _ctx(str(tmp_path)),
            )
        assert result.success
        assert result.data["invalid"] == 1
        assert "parse exploded" in result.data["results"][0]["error"]


# ── _call_cached ─────────────────────────────────────────────────────────────

def _fake_pymatgen(monkeypatch, **overrides):
    """构造一个 fake pymatgen 模块, 用于覆盖主路径 import 成功分支.

    所有 sys.modules 注入都经由 monkeypatch.setitem, 测试结束后自动恢复,
    避免污染全局导入状态 (否则会泄漏到其他测试文件的真实 pymatgen 导入).
    """
    class FakeLattice:
        a = b = c = 5.4
        alpha = beta = gamma = 90.0

    class FakeStructure:
        formula = "Si8"
        volume = 157.5
        density = 2.33
        lattice = FakeLattice()

        def __len__(self):
            return 8

        @staticmethod
        def from_file(path):
            return FakeStructure()

    monkeypatch.setitem(sys.modules, "pymatgen", types.ModuleType("pymatgen"))
    core_mod = types.ModuleType("pymatgen.core")
    core_mod.Structure = overrides.get("Structure", FakeStructure)
    monkeypatch.setitem(sys.modules, "pymatgen.core", core_mod)

    if overrides.get("analyzer_success"):
        class FakeAnalyzer:
            def __init__(self, structure):
                pass

            def get_space_group_symbol(self):
                return "Fd-3m"

        sga_mod = types.ModuleType("pymatgen.symmetry")
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry", sga_mod)
        analyzer_mod = types.ModuleType("pymatgen.symmetry.analyzer")
        analyzer_mod.SpacegroupAnalyzer = FakeAnalyzer
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry.analyzer", analyzer_mod)
    elif overrides.get("analyzer_error"):
        sga_mod = types.ModuleType("pymatgen.symmetry")
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry", sga_mod)
        analyzer_mod = types.ModuleType("pymatgen.symmetry.analyzer")

        class BadAnalyzer:
            def __init__(self, structure):
                raise RuntimeError("analyzer failed")

        analyzer_mod.SpacegroupAnalyzer = BadAnalyzer
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry.analyzer", analyzer_mod)
    elif overrides.get("analyzer_import_error"):
        # SpacegroupAnalyzer import 失败 → 走 get_space_group_info fallback
        sga_mod = types.ModuleType("pymatgen.symmetry")
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry", sga_mod)
        # 让 analyzer 子模块 import 抛 ImportError (key 存在但值为 None)
        monkeypatch.setitem(sys.modules, "pymatgen.symmetry.analyzer", None)


class TestCallCached:
    def _unload_pymatgen(self, monkeypatch):
        # 把所有 pymatgen 条目替换成空 ModuleType (无 Structure/子模块属性),
        # 强制 `from pymatgen.core import Structure` 抛 ImportError——即使
        # pymatgen 已安装 (CI test job 会 pip install pymatgen), 空模块也会让
        # import 从 sys.modules 拿到无 Structure 的模块而失败, 稳定覆盖
        # fallback 分支. 显式占位核心子模块, 因为若 pymatgen 尚未被导入,
        # sys.modules 里根本没有这些 key, 只遍历已存在条目会漏, import 时
        # 会从磁盘重新加载真实 pymatgen. monkeypatch.setitem 保证测试结束后
        # 恢复, 不泄漏全局导入状态.
        blank = types.ModuleType("pymatgen")
        keys = [
            "pymatgen",
            "pymatgen.core",
            "pymatgen.symmetry",
            "pymatgen.symmetry.analyzer",
            "pymatgen.core.composition",
            "pymatgen.core.periodic_table",
            "pymatgen.core.lattice",
            "pymatgen.core.sites",
            "pymatgen.core.structure",
        ]
        for k in keys:
            monkeypatch.setitem(sys.modules, k, blank)
        # 顺带清掉已加载的其他 pymatgen.* 子模块
        for k in list(sys.modules):
            if (k == "pymatgen" or k.startswith("pymatgen.")) and k not in keys:
                monkeypatch.setitem(sys.modules, k, blank)

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        tool = StructureTool()
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(tmp_path / "nope")),
            _ctx(str(tmp_path)),
        )
        assert not result.success
        assert "File not found" in result.error

    @pytest.mark.asyncio
    async def test_pymatgen_success_with_analyzer(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        _fake_pymatgen(monkeypatch, analyzer_success=True)
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si8\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n8\n0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert result.success
        assert result.data["formula"] == "Si8"
        assert result.data["spacegroup"] == "Fd-3m"
        assert result.data["num_atoms"] == 8

    @pytest.mark.asyncio
    async def test_pymatgen_analyzer_error_fallback(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        _fake_pymatgen(monkeypatch, analyzer_error=True)
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si8\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n8\n0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        # analyzer 抛异常, 但 structure 无 get_space_group_info → spacegroup None
        assert result.success
        assert result.data["spacegroup"] is None

    @pytest.mark.asyncio
    async def test_pymatgen_analyzer_error_uses_sg_info(self, tmp_path, monkeypatch):
        # analyzer 抛异常, 但 structure 有 get_space_group_info → fallback 用它
        self._unload_pymatgen(monkeypatch)

        class FakeLattice:
            a = b = c = 5.4
            alpha = beta = gamma = 90.0

        class FakeStructureWithSG:
            formula = "Si8"
            volume = 157.5
            density = 2.33
            lattice = FakeLattice()

            def __len__(self):
                return 8

            def get_space_group_info(self):
                return ("Fm-3m", 225)

            @staticmethod
            def from_file(path):
                return FakeStructureWithSG()

        _fake_pymatgen(monkeypatch, analyzer_error=True, Structure=FakeStructureWithSG)
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\n8\n0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert result.success
        assert result.data["spacegroup"] == "Fm-3m"

    @pytest.mark.asyncio
    async def test_pymatgen_import_error_basic_info(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        tool = StructureTool()
        # 基本的 POSCAR: 第 6 行 (idx 5) 是原子数, 好让 fallback 解析出 8
        f = tmp_path / "POSCAR"
        f.write_text("Si\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\n8\n0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert result.success
        assert "pymatgen not installed" in result.data["warnings"][0]
        assert result.data["num_atoms"] == 8

    @pytest.mark.asyncio
    async def test_import_error_invalid_num_atoms(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("Si8\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\nnotanumber\n0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert result.success
        assert result.data["num_atoms"] is None  # ValueError 被吞

    @pytest.mark.asyncio
    async def test_fallback_not_poscar(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        tool = StructureTool()
        f = tmp_path / "data.xyz"
        f.write_text("3\ncomment\nH 0 0 0\n")
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert result.success
        assert result.data["num_atoms"] is None  # 非 POSCAR 不解析

    @pytest.mark.asyncio
    async def test_parse_exception_wrapped(self, tmp_path, monkeypatch):
        self._unload_pymatgen(monkeypatch)
        tool = StructureTool()
        f = tmp_path / "POSCAR"
        f.write_text("x")
        # 让 Structure.from_file 抛异常 → 走 except 包装
        core_mod = types.ModuleType("pymatgen.core")
        class Boom:
            @staticmethod
            def from_file(path):
                raise ValueError("bad file")
        core_mod.Structure = Boom
        monkeypatch.setitem(sys.modules, "pymatgen", types.ModuleType("pymatgen"))
        monkeypatch.setitem(sys.modules, "pymatgen.core", core_mod)
        result = await tool._call_cached(
            StructureToolInput(action="read", file_path=str(f)), _ctx(str(tmp_path))
        )
        assert not result.success
        assert "Failed to parse structure" in result.error
