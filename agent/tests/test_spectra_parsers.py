"""谱图解析器 (JCAMP-DX / mzML) 单元测试 — G-B「NMR/质谱谱图解析」能力锚点.

单一判据来源: ``huginn.tools.sci.spectra_parsers`` 是唯一解析实现,
``characterization_tool`` 的 jdx_parse / mzml_parse 只是薄适配层. 覆盖:

- JCAMP-DX 峰表 (真实 fixture 方言) + 压缩轨迹 (SQZ/DIF/DUP 字母表) +
  XFACTOR/YFACTOR 缩放 + 不支持的 DATA FORM **显式 ValueError 点名形态**
  (静默返回空 = 假绿, 是本能力最忌讳的失败方式).
- mzML: 64/32-bit float + zlib + 截断/多谱上限 + 非法 XML 显式报错 +
  裸 ``referenceableParamGroupRef`` 数组显式入 ``skipped_arrays`` (不静默丢).
- 工具接线面 (缺 numpy/pydantic 则 skip): CharacterizationTool 的
  jdx_parse/mzml_parse 只做「读文件 + 调判据」, 解析器错误原样上抛为
  success=False, 不吞不降级.

fixture: ``tests/fixtures/spectra/*.jdx`` (真实方言样本, 与判据同源校验).
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import pytest

from huginn.tools.sci.spectra_parsers import parse_jdx, parse_mzml

FIXTURES = Path(__file__).parent / "fixtures" / "spectra"

# ── mzML 测试用 PSI-MS accession (与判据实现同一标准术语表) ──────────────────
_F64 = "MS:1000523"
_F32 = "MS:1000521"
_NOCOMP = "MS:1000576"
_ZLIB = "MS:1000574"
_MZ = "MS:1000514"
_INT = "MS:1000515"
_MSLEVEL = "MS:1000511"


def _b64(values: list[float], *, dtype: str = "d", compress: bool = False) -> str:
    raw = struct.pack(f"<{len(values)}{dtype}", *values)
    if compress:
        raw = zlib.compress(raw)
    return base64.b64encode(raw).decode()


def _bda(
    payload: str,
    arr_acc: str,
    *,
    precision: str = _F64,
    compression: str = _NOCOMP,
) -> str:
    return (
        "<binaryDataArray>"
        f'<cvParam accession="{precision}" name="precision"/>'
        f'<cvParam accession="{compression}" name="compression"/>'
        f'<cvParam accession="{arr_acc}" name="array"/>'
        f"<binary>{payload}</binary>"
        "</binaryDataArray>"
    )


def _ref_only_bda(payload: str) -> str:
    """裸 referenceableParamGroupRef (无内联 accession) —— 判据须显式跳过它."""
    return (
        "<binaryDataArray>"
        '<referenceableParamGroupRef ref="commonInstrumentParams"/>'
        f"<binary>{payload}</binary>"
        "</binaryDataArray>"
    )


def _spectrum(index: int, mz_bda: str, inten_bda: str, ms_level: str = "1") -> str:
    return (
        f'<spectrum index="{index}" id="scans={index}">'
        f'<cvParam accession="{_MSLEVEL}" name="ms level" value="{ms_level}"/>'
        f"<binaryDataArrayList count=\"2\">{mz_bda}{inten_bda}</binaryDataArrayList>"
        "</spectrum>"
    )


def _mzml(*spectra: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">'
        "<mzML><run>" + "".join(spectra) + "</run></mzML></indexedmzML>"
    )


# ── JCAMP-DX: 峰表 ─────────────────────────────────────────────────────────


def test_peak_table_fixture():
    data = parse_jdx((FIXTURES / "nmr_peaktable.jdx").read_text(encoding="utf-8"))
    assert data["title"] == "Huginn test 1H NMR peak table"
    assert data["data_type"] == "NMR SPECTRUM"
    assert data["xunits"] == "HZ"
    assert data["yunits"] == "ARBITRARY UNITS"
    assert data["observer_frequency"] == "400.13"
    assert data["data_form"] == "PEAK TABLE"
    assert data["n_points"] == 3
    assert data["peaks"] == [
        {"x": 100.5, "y": 10.0},
        {"x": 205.0, "y": 25.5},
        {"x": 310.25, "y": 5.0},
    ]


# ── JCAMP-DX: 压缩轨迹 (SQZ/DIF/DUP) ──────────────────────────────────────


def test_xydata_compressed_fixture():
    """fixture 覆盖全部三类压缩字母 (规范字母表: DIF=J..R, DUP=S..Z):
    SQZ(B0=+20) DIF(K=+2, j=-1) DUP(U=再重复 3 次)."""
    data = parse_jdx((FIXTURES / "nmr_xydata.jdx").read_text(encoding="utf-8"))
    assert data["data_form"] == "XYDATA"
    assert data["n_points"] == 7
    assert data["x"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # 行1: B0→20, K(DIF=+2)→22, j(DIF=-1)→21; 行2: E(SQZ)→5, U(DUP=3)→5,5,5,5
    assert data["y"] == [20.0, 22.0, 21.0, 5.0, 5.0, 5.0, 5.0]


def test_xydata_plain_affn_numbers():
    data = parse_jdx(
        "##XYDATA= (X++(Y..Y))\n1++1 2 3\n##END=\n"
    )
    assert data["y"] == [1.0, 2.0, 3.0]


def test_xydata_factors_scale_axes():
    data = parse_jdx(
        "##XFACTOR= 0.5\n##YFACTOR= 0.1\n"
        "##XYDATA= (X++(Y..Y))\n10++B0 K\n11++B\n##END=\n"
    )
    assert data["x"] == [10.0, 10.5, 11.0]
    assert data["y"] == [2.0, 2.2, 0.2]


def test_xydata_dif_letter_at_line_start_is_absolute():
    """宽容边界: 行首 DIF 字母 (规范应首用 SQZ) 按绝对值处理, 只此一次."""
    data = parse_jdx("##XYDATA= (X++(Y..Y))\n1++K\n##END=\n")
    assert data["y"] == [2.0]


def test_xydata_dup_letter_at_line_start_raises():
    with pytest.raises(ValueError, match="DUP"):
        parse_jdx("##XYDATA= (X++(Y..Y))\n0++S\n##END=\n")


# ── JCAMP-DX: 诚实边界 (不支持的形态显式报错) ──────────────────────────────


def test_unsupported_data_form_names_the_form():
    for form in ("DATATABLE", "XYPOINTS"):
        with pytest.raises(ValueError, match=form):
            parse_jdx(f"##TITLE=t\n##{form}= (X..Y)\n1 2\n##END=\n")


def test_missing_data_section_raises():
    with pytest.raises(ValueError, match="未找到数据段"):
        parse_jdx("##TITLE=empty\n##JCAMP-DX= 4.24\n##END=\n")
    with pytest.raises(ValueError, match="未找到数据段"):
        parse_jdx("")


# ── mzML ───────────────────────────────────────────────────────────────────


def test_mzml_two_peaks():
    doc = _mzml(
        _spectrum(0, _bda(_b64([100.0, 200.5]), _MZ), _bda(_b64([10.0, 42.0]), _INT))
    )
    data = parse_mzml(doc)
    assert data["n_spectra_total"] == 1
    assert data["n_spectra"] == 1
    assert data["skipped_arrays"] == []
    spec = data["spectra"][0]
    assert spec["index"] == "0"
    assert spec["ms_level"] == "1"
    assert spec["n_peaks"] == 2
    assert spec["truncated"] is False
    assert spec["peaks"] == [
        {"mz": 100.0, "intensity": 10.0},
        {"mz": 200.5, "intensity": 42.0},
    ]


def test_mzml_zlib_compression():
    doc = _mzml(
        _spectrum(
            0,
            _bda(_b64([100.0, 200.5], compress=True), _MZ, compression=_ZLIB),
            _bda(_b64([10.0, 42.0], compress=True), _INT, compression=_ZLIB),
        )
    )
    data = parse_mzml(doc)
    assert data["spectra"][0]["peaks"] == [
        {"mz": 100.0, "intensity": 10.0},
        {"mz": 200.5, "intensity": 42.0},
    ]


def test_mzml_float32_precision():
    # 选 f32 可精确表示的值, 避免 32-bit 截断噪声干扰判据
    doc = _mzml(
        _spectrum(
            0,
            _bda(_b64([100.0, 200.5], dtype="f"), _MZ, precision=_F32),
            _bda(_b64([10.0, 42.0], dtype="f"), _INT, precision=_F32),
        )
    )
    data = parse_mzml(doc)
    assert data["spectra"][0]["peaks"] == [
        {"mz": 100.0, "intensity": 10.0},
        {"mz": 200.5, "intensity": 42.0},
    ]


def test_mzml_ref_only_array_is_explicitly_skipped():
    """裸 ref 数组不静默丢: 进 skipped_arrays, 缺 mz → 0 峰也不假装解析成功."""
    doc = _mzml(
        _spectrum(0, _ref_only_bda(_b64([100.0])), _bda(_b64([10.0]), _INT))
    )
    data = parse_mzml(doc)
    assert data["skipped_arrays"] == ["spectrum[0]"]
    assert data["spectra"][0]["n_peaks"] == 0
    assert data["spectra"][0]["peaks"] == []


def test_mzml_invalid_xml_raises():
    with pytest.raises(ValueError, match="XML 不合法"):
        parse_mzml("<indexedmzML><not-closed")


def test_mzml_peak_truncation_is_explicit():
    doc = _mzml(
        _spectrum(0, _bda(_b64([1.0, 2.0, 3.0]), _MZ), _bda(_b64([5.0, 6.0, 7.0]), _INT))
    )
    data = parse_mzml(doc, max_peaks=2)
    spec = data["spectra"][0]
    assert spec["n_peaks"] == 3
    assert spec["n_peaks_returned"] == 2
    assert spec["truncated"] is True


def test_mzml_spectrum_cap_counts_total():
    spectra = [
        _spectrum(i, _bda(_b64([float(i)]), _MZ), _bda(_b64([9.0]), _INT))
        for i in range(3)
    ]
    data = parse_mzml(_mzml(*spectra), max_spectra=2)
    assert data["n_spectra_total"] == 3
    assert data["n_spectra"] == 2


# ── CharacterizationTool 接线 (薄适配层; 缺 numpy/pydantic 则 skip) ──────────


def _tool_layer():
    try:
        from huginn.core_types import ToolContext
        from huginn.tools.characterization_tool import CharacterizationTool
    except ModuleNotFoundError as exc:  # 裸环境降级: CI 全量套件必装, 不掩真故障
        pytest.skip(f"characterization_tool 栈不可导入: {exc}")
    return CharacterizationTool, ToolContext


def _run_tool(tool, ctx, **kwargs):
    import asyncio

    return asyncio.run(tool.call(tool.input_schema(**kwargs), ctx))


def test_tool_jdx_parse_peak_table(tmp_path):
    CharacterizationTool, ToolContext = _tool_layer()
    tool = CharacterizationTool()
    result = _run_tool(
        tool,
        ToolContext(session_id="t", workspace=str(tmp_path)),
        action="jdx_parse",
        data_path=str(FIXTURES / "nmr_peaktable.jdx"),
    )
    assert result.success is True
    assert result.data["data_form"] == "PEAK TABLE"
    assert result.data["n_points"] == 3
    assert result.data["source_file"].endswith("nmr_peaktable.jdx")


def test_tool_jdx_parse_unsupported_form_is_explicit_error(tmp_path):
    CharacterizationTool, ToolContext = _tool_layer()
    tool = CharacterizationTool()
    bad = tmp_path / "bad.jdx"
    bad.write_text("##DATATABLE= (X..Y)\n1 2\n##END=\n", encoding="utf-8")
    result = _run_tool(
        tool,
        ToolContext(session_id="t", workspace=str(tmp_path)),
        action="jdx_parse",
        data_path=str(bad),
    )
    assert result.success is False
    assert "DATATABLE" in (result.error or "")


def test_tool_mzml_parse_with_parameter_passthrough(tmp_path):
    CharacterizationTool, ToolContext = _tool_layer()
    tool = CharacterizationTool()
    mzml_file = tmp_path / "mini.mzml"
    mzml_file.write_text(
        _mzml(
            _spectrum(
                0, _bda(_b64([1.0, 2.0, 3.0]), _MZ), _bda(_b64([5.0, 6.0, 7.0]), _INT)
            )
        ),
        encoding="utf-8",
    )
    ctx = ToolContext(session_id="t", workspace=str(tmp_path))
    result = _run_tool(
        tool, ctx, action="mzml_parse", data_path=str(mzml_file)
    )
    assert result.success is True
    assert result.data["n_spectra"] == 1

    capped = _run_tool(
        tool,
        ctx,
        action="mzml_parse",
        data_path=str(mzml_file),
        parameters={"max_peaks": 1},
    )
    assert capped.success is True
    assert capped.data["spectra"][0]["n_peaks_returned"] == 1
    assert capped.data["spectra"][0]["truncated"] is True


def test_tool_mzml_parse_invalid_xml_is_explicit_error(tmp_path):
    CharacterizationTool, ToolContext = _tool_layer()
    tool = CharacterizationTool()
    bad = tmp_path / "bad.mzml"
    bad.write_text("<indexedmzML><broken", encoding="utf-8")
    result = _run_tool(
        tool,
        ToolContext(session_id="t", workspace=str(tmp_path)),
        action="mzml_parse",
        data_path=str(bad),
    )
    assert result.success is False
    assert "XML" in (result.error or "")
