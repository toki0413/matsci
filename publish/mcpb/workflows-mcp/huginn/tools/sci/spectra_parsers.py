"""科学谱图文件解析器 (纯 stdlib): JCAMP-DX (.jdx) + mzML.

单一判据来源: 本模块是 JCAMP-DX / mzML 解析的**唯一**实现,
``characterization_tool`` 的 ``jdx_parse`` / ``mzml_parse`` action 只是薄适配层
(读文件 + 调这里), 杜绝两处实现各漂各的.

设计约束:
- 纯 stdlib (re / base64 / zlib / struct / xml.etree), 不 import huginn.*,
  不 import 任何第三方 —— L0 探针零新增依赖, 测试 weightless 可跑.
- 不认识的格式**显式报错** (报出具体形态), 不静默返回空 (静默 = 假绿).

诚实边界 (JCAMP-DX 方言覆盖):
- 支持: ``##PEAK TABLE`` (NMR 峰表, Bruker/Varian/ JEOL 常见导出) 与
  ``##XYDATA= (X++(Y..Y))`` 压缩轨迹 (SQZ/DIF/DUP 编码).
- 不支持: 其余 DATA FORM (##DATATABLE / ##XYPOINTS / 6 版新头部等) 显式
  ValueError 点名形态; 厂商私有方言不承诺兼容.
- DUP 语义按 JCAMP 规范: S..Z 表示把**前一个数**再重复 1..8 次.

诚实边界 (mzML):
- cvParam 按 accession 识别 (64/32-bit float, zlib/no compression, m/z /
  intensity); ``referenceableParamGroup`` **不展开** —— 依赖 ref 而无内联
  accession 的数组会显式记入 ``skipped_arrays``, 不静默丢弃.
"""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
import zlib

__all__ = ["parse_jdx", "parse_mzml", "JDX_SUPPORTED_FORMS"]

# JCAMP 压缩编码字符表 (规范字母表):
#   SQZ: A..I = +1..+9, a..i = -1..-9   (行内第一个 Y = 绝对值)
#   DIF: J..R = +1..+9, j..r = -1..-9   (后续 Y = 与前值的差)
#   DUP: S..Z = 前一个数再重复 1..8 次
_SQZ = {c: i + 1 for i, c in enumerate("ABCDEFGHI")}
_SQZ.update({c: -(i + 1) for i, c in enumerate("abcdefghi")})
_DIF = {c: i + 1 for i, c in enumerate("JKLMNOPQR")}
_DIF.update({c: -(i + 1) for i, c in enumerate("jklmnopqr")})
_DUP = {c: i + 1 for i, c in enumerate("STUVWXYZ")}

# 一个"数"的 token: (可选符号字母)(数字串, 可含小数点); 或裸数字
_NUM_RE = re.compile(
    r"(?P<sqz>[A-Ia-i])(?P<sdigit>[0-9.]*)"
    r"|(?P<dif>[J-Rj-r])(?P<ddigit>[0-9.]*)"
    r"|(?P<dup>[S-Z])"
    r"|(?P<plain>[-+]?[0-9]+(?:\.[0-9]*)?)"
)

JDX_SUPPORTED_FORMS = ("PEAK TABLE", "XYDATA")


def _meta_value(line: str) -> tuple[str, str]:
    """##KEY=VALUE 行 → (KEY 大写, VALUE 原样去首尾空白)."""
    body = line[2:].strip()
    key, _, value = body.partition("=")
    return key.strip().upper(), value.strip()


def _decode_y_stream(chunk: str) -> list[float]:
    """解一行 XYDATA 的 Y 串 (SQZ/DIF/DUP 压缩) → 数值列表 (未乘 YFACTOR)."""
    vals: list[float] = []
    for m in _NUM_RE.finditer(chunk):
        if m.group("dup"):
            if not vals:
                raise ValueError("DUP 字母出现在行首, 前面没有可重复的数")
            vals.extend([vals[-1]] * _DUP[m.group("dup")])
        elif m.group("sqz"):
            sign = _SQZ[m.group("sqz")]
            digits = m.group("sdigit")
            vals.append(float(f"{sign}{digits}"))
        elif m.group("dif"):
            sign = _DIF[m.group("dif")]
            digits = m.group("ddigit")
            if not vals:
                # 行内第一个 token 出现 DIF 字母: 规范里第一项应是 SQZ.
                # 宽容处理为绝对值 (有些导出器首项也用 DIF 字母), 但只此一次.
                vals.append(float(f"{sign}{digits}"))
            else:
                vals.append(vals[-1] + float(f"{sign}{digits}"))
        else:
            vals.append(float(m.group("plain")))
    return vals


def _scan_jdx_header(
    lines: list[str],
) -> tuple[dict[str, str], str | None, int | None]:
    """扫描 ## 头部; 返回 (meta, data_form, data_start)."""
    meta: dict[str, str] = {}
    data_start: int | None = None
    data_form: str | None = None
    for i, line in enumerate(lines):
        if line.startswith("##"):
            key, value = _meta_value(line)
            if key in ("END",):
                break
            if key in ("PEAK TABLE", "XYDATA"):
                data_form = key
                data_start = i + 1
                meta[key] = value  # 保留 (X++(Y..Y)) 之类的形态描述
                break
            meta[key] = value
    if data_form is None or data_start is None:
        # 再扫一遍: 有些文件把数据形态写在别处 (如 ##DATA TABLE=)
        other = [
            _meta_value(line)[0]
            for line in lines
            if line.startswith("##")
            and _meta_value(line)[0] in ("DATATABLE", "XYPOINTS")
        ]
        if other:
            raise ValueError(
                f"不支持的 JCAMP DATA FORM: {other[0]} "
                f"(支持: {', '.join(JDX_SUPPORTED_FORMS)})"
            )
        raise ValueError("未找到数据段 (##PEAK TABLE / ##XYDATA)")
    return meta, data_form, data_start


def _jdx_body(lines: list[str], data_start: int) -> list[str]:
    body = []
    for line in lines[data_start:]:
        if line.startswith("##"):
            break
        body.append(line)
    return body


def _parse_peak_table(body: list[str]) -> list[dict]:
    peaks = []
    for line in body:
        parts = re.split(r"[,\s]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            peaks.append({"x": float(parts[0]), "y": float(parts[1])})
        except ValueError:
            continue  # 表头/杂行
    return peaks


def _parse_xydata(
    body: list[str], meta: dict[str, str]
) -> tuple[list[float], list[float]]:
    """XYDATA= (X++(Y..Y)): 每行 "x首值++Y压缩串", x 以 XFACTOR 步进."""
    xfactor = float(meta.get("XFACTOR", "1") or 1)
    yfactor = float(meta.get("YFACTOR", "1") or 1)
    xs: list[float] = []
    ys: list[float] = []
    for line in body:
        if "++" not in line:
            continue
        left, _, ychunk = line.partition("++")
        ychunk = ychunk.strip()
        if not ychunk:
            continue
        try:
            x_first = float(left.strip())
        except ValueError as exc:
            raise ValueError(f"XYDATA 行首 x 值不合法: {left.strip()!r}") from exc
        row = [v * yfactor for v in _decode_y_stream(ychunk)]
        for j, y in enumerate(row):
            xs.append(x_first + j * xfactor)
            ys.append(y)
    return xs, ys


def parse_jdx(text: str) -> dict:
    """解析 JCAMP-DX 文本 (常见 NMR 变体) → 结构化 dict.

    返回: {"title", "data_type", "xunits", "yunits", "observer_frequency",
           "data_form", "x"/"y" (XYDATA) 或 "peaks" (PEAK TABLE), "n_points"}.
    不认识的 DATA FORM → ValueError 点名形态.
    """
    lines = text.splitlines()
    meta, data_form, data_start = _scan_jdx_header(lines)

    body = _jdx_body(lines, data_start)

    out = {
        "title": meta.get("TITLE", ""),
        "data_type": meta.get("DATA TYPE", ""),
        "xunits": meta.get("XUNITS", ""),
        "yunits": meta.get("YUNITS", ""),
        "observer_frequency": meta.get(".OBSERVER FREQUENCY", ""),
        "data_form": data_form,
    }

    if data_form == "PEAK TABLE":
        peaks = _parse_peak_table(body)
        out["peaks"] = peaks
        out["n_points"] = len(peaks)
        return out

    xs, ys = _parse_xydata(body, meta)
    out["x"] = xs
    out["y"] = ys
    out["n_points"] = len(ys)
    return out


# mzML cvParam accession (PSI-MS 标准术语)
_MS_LEVEL = "MS:1000511"
_MZ_ARRAY = "MS:1000514"
_INTENSITY_ARRAY = "MS:1000515"
_FLOAT64 = "MS:1000523"
_FLOAT32 = "MS:1000521"
_ZLIB = "MS:1000574"
_NO_COMPRESSION = "MS:1000576"


def _local(tag: str) -> str:
    """去命名空间: '{ns}spectrum' → 'spectrum'."""
    return tag.rsplit("}", 1)[-1]


def _cv_accessions(elem: ET.Element) -> set[str]:
    return {
        c.get("accession")
        for c in elem
        if _local(c.tag) == "cvParam" and c.get("accession")
    }


def _decode_binary(bd: ET.Element) -> tuple[str | None, list[float]]:
    """解一个 <binaryDataArray> → (数组种类名, 数值列表)."""
    accessions = _cv_accessions(bd)
    if not accessions:
        return None, []
    binary = None
    for child in bd:
        if _local(child.tag) == "binary":
            binary = (child.text or "").strip()
    if not binary:
        return next(iter(accessions)), []
    raw = base64.b64decode(binary)
    if _ZLIB in accessions:
        raw = zlib.decompress(raw)
    dtype = "d" if _FLOAT64 in accessions else ("f" if _FLOAT32 in accessions else "")
    if not dtype:
        return next(iter(accessions)), []
    n = len(raw) // struct.calcsize(f"<{dtype}")
    values = list(
        struct.unpack(f"<{n}{dtype}", raw[: n * struct.calcsize(f"<{dtype}")])
    )
    if _MZ_ARRAY in accessions:
        return "mz", values
    if _INTENSITY_ARRAY in accessions:
        return "intensity", values
    return None, values


def _parse_spectrum(
    elem: ET.Element, max_peaks: int, skipped_arrays: list[str]
) -> dict:
    """解一个 <spectrum> 元素 → 谱 dict."""
    ms_level = ""
    for child in elem:
        if _local(child.tag) == "cvParam" and child.get("accession") == _MS_LEVEL:
            ms_level = child.get("value") or ""
    arrays: dict[str, list[float]] = {}
    for arr in elem.iter():
        if _local(arr.tag) != "binaryDataArray":
            continue
        kind, values = _decode_binary(arr)
        if kind is None:
            skipped_arrays.append(f"spectrum[{elem.get('index')}]")
            continue
        arrays[kind] = values
    mz = arrays.get("mz", [])
    inten = arrays.get("intensity", [])
    n = min(len(mz), len(inten))
    peaks = [{"mz": mz[i], "intensity": inten[i]} for i in range(min(n, max_peaks))]
    return {
        "index": elem.get("index"),
        "ms_level": ms_level,
        "n_peaks": n,
        "n_peaks_returned": len(peaks),
        "peaks": peaks,
        "truncated": n > max_peaks,
    }


def parse_mzml(
    data: bytes | str,
    *,
    max_spectra: int = 5,
    max_peaks: int = 100,
) -> dict:
    """解析 mzML (质谱) → {"n_spectra_total", "n_spectra", "spectra", "skipped_arrays"}.

    - cvParam 按 PSI-MS accession 识别; 数组解码 base64 (+可选 zlib) → float.
    - ``skipped_arrays`` 显式列出因无内联 accession (仅 referenceableParamGroup
      ref) 而跳过的数组 —— 不静默丢.
    - XML 不合法 → ValueError.
    - 最多回 ``max_spectra`` 个谱, 每谱最多 ``max_peaks`` 个峰 (防巨文件爆输出).
    """
    try:
        root = ET.fromstring(data if isinstance(data, bytes) else str(data))
    except ET.ParseError as exc:
        raise ValueError(f"mzML XML 不合法: {exc}") from exc

    spectra_total = 0
    skipped_arrays: list[str] = []
    spectra: list[dict] = []
    for elem in root.iter():
        if _local(elem.tag) != "spectrum":
            continue
        spectra_total += 1
        if len(spectra) >= max_spectra:
            continue
        spectra.append(_parse_spectrum(elem, max_peaks, skipped_arrays))
    return {
        "n_spectra_total": spectra_total,
        "n_spectra": len(spectra),
        "spectra": spectra,
        "skipped_arrays": sorted(set(skipped_arrays)),
    }
