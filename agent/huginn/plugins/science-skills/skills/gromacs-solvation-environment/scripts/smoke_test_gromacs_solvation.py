#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gromacs_solvation_environment import (
    DISTRIBUTION_FILENAME,
    DISTRIBUTION_PLOT_FILENAME,
    PLOT_FILENAME,
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    analyze_universe,
    build_inspection_report,
    choose_frame_indices,
    write_analysis_outputs,
)


def build_synthetic_universe() -> object:
    """构造含跨 PBC 分子和多帧坐标的最小测试体系。

    功能目的：在不依赖外部 GROMACS fixture 的情况下验证核心 molnum/moltype/bonds 语义。
    输入参数：无。
    返回值：内存中的 MDAnalysis Universe。
    关键流程：两个 Li 单原子分子、一个跨边界 SOL 双原子分子和一个远处 NA 分子。
    可能报错或边界情况：需要当前 base 环境已有 MDAnalysis；不写用户文件。
    """

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mda.Universe.empty(
        5,
        n_residues=4,
        atom_resindex=[0, 1, 1, 2, 3],
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["LI", "O1", "C1", "LI", "NA"])
    universe.add_TopologyAttr("resnames", ["LI", "SOL", "LI", "NA"])
    universe.add_TopologyAttr("resids", [1, 2, 3, 4])
    # MDAnalysis 的 Molnums/Moltypes 是 residue-level 属性，AtomGroup 访问时会广播到原子。
    universe.add_TopologyAttr("molnums", [0, 1, 2, 3])
    universe.add_TopologyAttr("moltypes", ["LI", "SOL", "LI", "NA"])
    universe.add_TopologyAttr("bonds", [(1, 2)])

    coordinates = np.asarray(
        [
            [[0.1, 5.0, 5.0], [9.8, 5.0, 5.0], [0.2, 5.0, 5.0], [0.3, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[0.1, 5.0, 5.0], [7.0, 5.0, 5.0], [7.4, 5.0, 5.0], [3.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[0.1, 5.0, 5.0], [9.8, 5.0, 5.0], [0.2, 5.0, 5.0], [0.3, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[0.1, 5.0, 5.0], [7.0, 5.0, 5.0], [7.4, 5.0, 5.0], [3.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            [[0.1, 5.0, 5.0], [9.8, 5.0, 5.0], [0.2, 5.0, 5.0], [0.3, 5.0, 5.0], [5.0, 5.0, 5.0]],
        ],
        dtype=np.float32,
    )
    dimensions = np.tile(np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], dtype=np.float32), (5, 1))
    universe.load_new(coordinates, format=MemoryReader, dimensions=dimensions)
    return universe


def assert_rfc4180_csv(path: Path) -> None:
    """检查 CSV 的 BOM、CRLF 和标准库可解析性。"""

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), f"CSV 缺少 UTF-8 BOM: {path}"
    payload = raw[3:]
    assert b"\r\n" in payload, f"CSV 缺少 CRLF: {path}"
    assert b"\n" not in payload.replace(b"\r\n", b""), f"CSV 含非 CRLF 换行: {path}"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"CSV 没有数据行: {path}"


def test_inspect(universe: object) -> None:
    """检查 inspect 能稳定报告 moltype/resname/atomname 和拓扑能力。"""

    report = build_inspection_report(universe, Path("synthetic.tpr"))
    assert report["capabilities"]["molnum"] is True
    assert report["capabilities"]["moltype"] is True
    assert report["capabilities"]["bonds"] is True
    assert report["capabilities"]["box"] is True
    assert [item["moltype"] for item in report["moltypes"]] == ["LI", "NA", "SOL"]
    assert any(item["atomname"] == "LI" and item["count"] == 2 for item in report["atomnames"])


def test_sampling() -> None:
    """检查默认逐帧、stride、均匀 n_frames 及冲突参数。"""

    assert choose_frame_indices(5, stride=None, n_frames=None) == [0, 1, 2, 3, 4]
    assert choose_frame_indices(5, stride=2, n_frames=None) == [0, 2, 4]
    assert choose_frame_indices(5, stride=None, n_frames=3) == [0, 2, 4]
    try:
        choose_frame_indices(5, stride=2, n_frames=3)
    except ValueError:
        pass
    else:
        raise AssertionError("stride 与 n_frames 冲突时应抛出 ValueError。")


def test_atom_and_molecule_centers(universe: object) -> dict[str, object]:
    """检查 atom/molecule 中心、PBC COG、自身排除和同 species 保留。"""

    atom_result = analyze_universe(
        universe,
        rdf_radius_nm=0.05,
        center_mode="atom",
        center_selection="name LI",
        n_frames=3,
    )
    assert atom_result["sampling"]["frame_indices"] == [0, 2, 4]
    assert atom_result["validation"]["actual_total_events"] == 6
    assert atom_result["validation"]["expected_total_events"] == 6
    assert atom_result["validation"]["fraction_sum_is_one"] is True
    for record in atom_result["records"]:
        # 每个 Li 中心都排除自身，但保留另一个同 moltype 的 Li 和跨 PBC 的 SOL。
        assert record["composition"] == {"LI": 1, "SOL": 1, "NA": 0}

    molecule_result = analyze_universe(
        universe,
        rdf_radius_nm=0.05,
        center_mode="molecule",
        # 只选 SOL 的一个原子，解析后必须扩展为 molnum=1 的完整双原子 molecule。
        center_selection="name O1",
        stride=2,
    )
    assert molecule_result["center_resolution"]["selected_atom_count"] == 1
    assert molecule_result["center_resolution"]["center_count"] == 1
    assert molecule_result["validation"]["actual_total_events"] == 3
    for record in molecule_result["records"]:
        assert record["composition"] == {"LI": 2, "SOL": 0, "NA": 0}
        assert record["total_neighbor_molecules"] == 2
    return atom_result


def test_outputs(result: dict[str, object], tmpdir: Path) -> None:
    """检查 CSV/JSON schema、编码、600 DPI PNG 和不生成 PDF。"""

    settings = {
        "tpr": "/tmp/synthetic.tpr",
        "xtc": None,
        "gro": None,
        "source_mode": "snapshot",
        "coordinate_source": "TPR current frame",
        "rdf_radius_nm": 0.05,
        "center_mode": "atom",
        "center_selection": "name LI",
        "stride": None,
        "n_frames": 3,
        "output_dir": str(tmpdir),
        "plot": True,
    }
    paths = write_analysis_outputs(result, tmpdir, settings=settings, plot=True)
    assert paths["records_csv"].name == RECORDS_FILENAME
    assert paths["distribution_csv"].name == DISTRIBUTION_FILENAME
    assert paths["summary_json"].name == SUMMARY_FILENAME
    assert paths["plot_png"].name == PLOT_FILENAME
    assert paths["distribution_plot_png"].name == DISTRIBUTION_PLOT_FILENAME
    assert_rfc4180_csv(paths["records_csv"])
    assert_rfc4180_csv(paths["distribution_csv"])

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["confirmed_settings"]["center_selection"] == "name LI"
    assert summary["summary"]["total_entries"] == 6
    assert math.isclose(sum(item["fraction"] for item in summary["environments"]), 1.0, abs_tol=1e-12)

    from PIL import Image

    with Image.open(paths["plot_png"]) as image:
        dpi = image.info.get("dpi")
    assert dpi is not None and math.isclose(float(dpi[0]), 600.0, rel_tol=0.01), dpi
    with Image.open(paths["distribution_plot_png"]) as image:
        distribution_dpi = image.info.get("dpi")
    assert distribution_dpi is not None and math.isclose(float(distribution_dpi[0]), 600.0, rel_tol=0.01), distribution_dpi
    assert not list(tmpdir.rglob("*.pdf")), "固定半径 smoke test 不应生成 PDF。"


def main() -> None:
    """执行不依赖真实 TPR/XTC 的确定性 smoke test。"""

    universe = build_synthetic_universe()
    test_inspect(universe)
    test_sampling()
    result = test_atom_and_molecule_centers(universe)
    with tempfile.TemporaryDirectory(prefix="gromacs_solvation_smoke_") as tmp:
        tmpdir = Path(tmp)
        test_outputs(result, tmpdir)
        print(f"GROMACS solvation smoke test OK: {tmpdir}")


if __name__ == "__main__":
    main()
