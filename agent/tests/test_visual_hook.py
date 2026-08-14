"""visual_hook.py 覆盖率补测 — 覆盖 should_visualize / render_tool_output /
extract_visual_primitives / 2D primitives / extract_box_primitives /
<point3d> 原语 / comparative / confidence 估算等全部分支.

配合既有测试把覆盖率从 6% 提升到 90%+.
"""

from __future__ import annotations

import io as _io

import numpy as np
from PIL import Image

from huginn.tools import visual_hook as vh


def _img_b64(arr) -> str:
    import base64 as b64

    buf = _io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return b64.b64encode(buf.getvalue()).decode()


# ── should_visualize ──────────────────────────────────────────────────────


def test_should_visualize_false_empty():
    assert vh.should_visualize("band_structure", {}) is False
    assert vh.should_visualize("band_structure", {"result": None}) is False
    assert vh.should_visualize("band_structure", {"result": "not dict"}) is False


def test_should_visualize_tool_name_pattern():
    assert vh.should_visualize("band_structure", {"result": {"foo": 1}}) is True
    assert vh.should_visualize("thermo_tool", {"result": {"foo": 1}}) is True


def test_should_visualize_result_keys():
    assert vh.should_visualize("foo", {"result": {"energies": [1, 2, 3]}}) is True
    assert vh.should_visualize("foo", {"result": {"scores": {"a": 1}}}) is True
    # 数值但不是 list/dict → False
    assert vh.should_visualize("foo", {"result": {"energies": 5}}) is False


def test_should_visualize_code_tool_numeric_stdout():
    out = {"result": {"stdout": "MAE: 0.05\nRMSE: 0.12\nR2 0.89\nloss 1.23e-4"}}
    assert vh.should_visualize("code_tool", out) is True
    # 少于 3 个数字 → False
    assert vh.should_visualize("bash_tool", {"result": {"output": "done"}}) is False


# ── _extract_metric_pairs ─────────────────────────────────────────────────


def test_extract_metric_pairs_basic():
    pairs = vh._extract_metric_pairs("MAE: 0.05\nRMSE = 0.12\nR2: 0.89")
    keys = [k for k, _ in pairs]
    assert "MAE" in keys and "RMSE" in keys and "R2" in keys


def test_extract_metric_pairs_dedup_and_filter():
    # 同名只保留第一次; file:line 模式排除; 非指标 key 排除
    pairs = vh._extract_metric_pairs("MAE: 0.05\nMAE: 9.9\nfoo.py:10\nline: 3")
    keys = [k for k, _ in pairs]
    assert keys.count("MAE") == 1
    assert "foo" not in keys
    assert "line" not in keys


def test_extract_metric_pairs_non_metric_key_filtered():
    # key 命中 _NON_METRIC_KEYS → 跳过
    pairs = vh._extract_metric_pairs("count: 5\nversion: 2")
    assert pairs == []


# ── render_tool_output ────────────────────────────────────────────────────


def test_render_line_plot():
    out = {"result": {"bands": [[0.0, 0.1, 0.2], [0.0, -0.1, -0.2]]}}
    b = vh.render_tool_output("band_structure", out)
    assert isinstance(b, str) and b


def test_render_single_list_line():
    out = {"result": {"dos": [0.1, 0.2, 0.3]}}
    assert vh.render_tool_output("dos", out)


def test_render_energies_bar():
    out = {"result": {"energies": [1.0, 2.0, 3.0], "labels": ["a", "b", "c"]}}
    assert vh.render_tool_output("thermo_tool", out)


def test_render_energy_scalar():
    out = {"result": {"energy": 1.234}}
    assert vh.render_tool_output("thermo_tool", out)


def test_render_stress_strain():
    out = {"result": {"stress": [1.0, 2.0], "strain": [0.0, 0.1]}}
    assert vh.render_tool_output("mechanical_tool", out)


def test_render_scores_bar():
    out = {"result": {"scores": {"a": 1.0, "b": 0.5}}}
    assert vh.render_tool_output("benchmark", out)


def test_render_code_tool_metrics(monkeypatch):
    out = {"result": {"stdout": "MAE: 0.05\nRMSE: 0.12"}}
    assert vh.render_tool_output("code_tool", out)


def test_render_not_plotted_returns_none():
    out = {"result": {"foo": 1}}
    assert vh.render_tool_output("band_structure", out) is None


def test_render_result_not_dict():
    assert vh.render_tool_output("band_structure", {"result": "x"}) is None


def test_render_import_matplotlib_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert vh.render_tool_output("band_structure", {"result": {"bands": [[1, 2]]}}) is None


def test_render_base64_too_large_returns_none(monkeypatch):
    # 图编码超过 _MAX_IMAGE_BYTES → 返回 None (防上下文膨胀)
    monkeypatch.setattr(vh, "_MAX_IMAGE_BYTES", 10)
    out = {"result": {"bands": [[0.0, 0.1, 0.2]]}}
    assert vh.render_tool_output("band_structure", out) is None


# ── extract_visual_primitives ─────────────────────────────────────────────


def test_extract_primitives_1d():
    out = {"result": {"energies": [1.0, 2.0, 3.0, 4.0, 5.0]}}
    text = vh.extract_visual_primitives("energies", out)
    assert "[energies]" in text
    assert "peak=<point>" in text
    assert "trend=increasing" in text


def test_extract_primitives_nested_bands():
    out = {"result": {"bands": [[0.0, 0.1, 0.2], [0.0, -0.1, -0.2]]}}
    text = vh.extract_visual_primitives("band_structure", out)
    assert "[bands]" in text
    assert "band0:" in text


def test_extract_primitives_flat_constant():
    # 全常数值 → std=0, 无 anomalies
    out = {"result": {"dos": [1.0, 1.0, 1.0, 1.0, 1.0]}}
    text = vh.extract_visual_primitives("dos", out)
    assert "anomalies=none" in text


def test_extract_primitives_scores():
    out = {"result": {"scores": {"a": 3.0, "b": 2.0, "c": 1.0}}}
    text = vh.extract_visual_primitives("benchmark", out)
    assert "[scores]" in text
    assert "top3" in text


def test_extract_primitives_code_tool_metrics():
    out = {"result": {"stdout": "MAE: 0.05\nRMSE: 0.12\nR2: 0.89"}}
    text = vh.extract_visual_primitives("code_tool", out)
    assert "[metrics]" in text


def test_extract_primitives_empty():
    assert vh.extract_visual_primitives("foo", {"result": {}}) == ""
    assert vh.extract_visual_primitives("foo", {"result": "x"}) == ""


def test_extract_primitives_derivative():
    # 有 anomalies(坐标化) + 拐点 + FWHM (n>=5, 非单调)
    data = [1.0, 1.2, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    out = {"result": {"dos": data}}
    text = vh.extract_visual_primitives("dos", out)
    assert "anomalies=" in text
    assert ("inflections" in text) or ("FWHM" in text)


def test_extract_primitives_single_point_trend_unknown():
    # n==1 → trend=unknown, 无导数分析
    out = {"result": {"dos": [3.0]}}
    text = vh.extract_visual_primitives("dos", out)
    assert "trend=unknown" in text


def test_extract_primitives_short_list_no_derivative():
    # n<5 → 不触发导数分析分支
    out = {"result": {"dos": [1.0, 2.0, 3.0]}}
    text = vh.extract_visual_primitives("dos", out)
    assert "FWHM" not in text


def test_extract_primitives_scores_bad_values():
    # scores 有非数值 → 触发 except 分支, 不崩溃
    out = {"result": {"scores": {"a": "x", "b": "y"}}}
    text = vh.extract_visual_primitives("benchmark", out)
    assert "[scores]" not in text


def test_extract_primitives_nested_band_no_numeric():
    # 嵌套 band 内无非数值 → nums 空 → continue
    out = {"result": {"bands": [["a", "b"]]}}
    assert vh.extract_visual_primitives("band_structure", out) == ""


def test_extract_primitives_flat_no_numeric():
    # 扁平 list 内无非数值 → nums 空 → continue
    out = {"result": {"dos": ["x", "y"]}}
    assert vh.extract_visual_primitives("dos", out) == ""


# ── _extract_2d_primitives ────────────────────────────────────────────────


def test_extract_2d_eds():
    result = {
        "elements": {
            "Fe": {"centroid_px": [50, 60], "coverage_fraction": 0.3,
                   "hotspots": [{"centroid_px": [10, 20], "area_px2": 100}]},
        },
        "image_shape": [100, 100],
        "overlaps": {"Fe-O": {"iou": 0.5}},
    }
    lines = vh._extract_2d_primitives(result)
    assert any("centroid=<point>" in l for l in lines)
    assert any("hotspot1" in l for l in lines)
    assert any("IoU" in l for l in lines)


def test_extract_2d_phase_field():
    result = {
        "volume_fractions": {"martensite": 0.4, "austenite": 0.6},
        "image_shape": [200, 200],
        "interface_pixel_fraction": 0.05,
        "morphology": {
            "martensite": {"n_domains": 3, "mean_domain_area_px2": 500,
                           "top_domain_centroids_px": [[10, 10], [50, 50]]},
        },
    }
    lines = vh._extract_2d_primitives(result)
    assert any("[phase_field]" in l for l in lines)
    assert any("martensite=" in l for l in lines)
    assert any("interface_fraction" in l for l in lines)
    assert any("morphology:" in l for l in lines)


def test_extract_2d_empty():
    assert vh._extract_2d_primitives({}) == []


# ── _normalize_coord / _to_coord ──────────────────────────────────────────


def test_normalize_coord():
    assert vh._normalize_coord(0, 2, 0.0, [0.0, 10.0]) == "0,0"
    assert vh._normalize_coord(1, 2, 10.0, [0.0, 10.0]) == "999,999"
    # v_range == 0 → 用 1.0 避免除零
    assert vh._normalize_coord(0, 1, 5.0, [5.0]) == "0,0"


def test_to_coord():
    idx, xy = vh._to_coord([1.0, 2.0, 3.0], 2)
    assert idx == 2
    assert xy == "999,999"


# ── extract_comparative_primitives ────────────────────────────────────────


def test_extract_comparative_1d():
    bl = {"result": {"dos": [1.0, 2.0, 3.0]}}
    cr = {"result": {"dos": [1.5, 2.5, 4.0]}}
    text = vh.extract_comparative_primitives(bl, cr)
    assert "peak_shift" in text
    assert "valley_shift" in text


def test_extract_comparative_new_anomalies():
    # baseline 展布大, current 收敛且带一个偏向 baseline 均值的离群点
    bl = {"result": {"dos": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0]}}
    cr = {"result": {"dos": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0]}}
    text = vh.extract_comparative_primitives(bl, cr)
    assert "new_anomalies" in text


def test_extract_comparative_nested_skipped():
    bl = {"result": {"bands": [[1.0, 2.0]]}}
    cr = {"result": {"bands": [[1.0, 3.0]]}}
    assert vh.extract_comparative_primitives(bl, cr) == ""


def test_extract_comparative_empty():
    assert vh.extract_comparative_primitives({"result": {}}, {"result": {}}) == ""
    assert vh.extract_comparative_primitives({"result": "x"}, {"result": "y"}) == ""


def test_extract_comparative_2d_eds():
    bl = {"result": {
        "elements": {"Fe": {"centroid_px": [10, 10], "coverage_fraction": 0.2}},
        "image_shape": [100, 100],
    }}
    cr = {"result": {
        "elements": {"Fe": {"centroid_px": [40, 50], "coverage_fraction": 0.4}},
        "image_shape": [100, 100],
    }}
    text = vh.extract_comparative_primitives(bl, cr)
    assert "centroid_shift" in text
    assert "coverage" in text


def test_extract_comparative_2d_lost_new_elements():
    bl = {"result": {"elements": {"Fe": {"centroid_px": [0, 0]}}, "image_shape": [10, 10]}}
    cr = {"result": {"elements": {"O": {"centroid_px": [0, 0]}}, "image_shape": [10, 10]}}
    text = vh.extract_comparative_primitives(bl, cr)
    assert "lost_elements" in text
    assert "new_elements" in text


def test_extract_comparative_2d_phase_field():
    bl = {"result": {"volume_fractions": {"martensite": 0.3}, "interface_pixel_fraction": 0.1}}
    cr = {"result": {"volume_fractions": {"martensite": 0.5}, "interface_pixel_fraction": 0.2}}
    text = vh.extract_comparative_primitives(bl, cr)
    assert "martensite:" in text
    assert "interface:" in text


def test_extract_comparative_2d_no_change():
    bl = {"result": {"elements": {"Fe": {"centroid_px": [10, 10], "coverage_fraction": 0.2}},
                     "image_shape": [100, 100]}}
    cr = {"result": {"elements": {"Fe": {"centroid_px": [10, 10], "coverage_fraction": 0.2}},
                     "image_shape": [100, 100]}}
    # 无位移无覆盖率变化 → 比较函数返回空或仅整体
    assert "centroid_shift" not in vh.extract_comparative_primitives(bl, cr)


# ── _estimate_data_confidence ─────────────────────────────────────────────


def test_confidence_1d():
    out = {"result": {"dos": [float(i) for i in range(20)]}}
    sc = vh._estimate_data_confidence(out)
    assert sc["confidence"] > 0
    assert "caveats" in sc


def test_confidence_few_points():
    out = {"result": {"dos": [1.0, 2.0]}}
    sc = vh._estimate_data_confidence(out)
    assert any("too_few_points" in c for c in sc["caveats"])


def test_confidence_2d_eds():
    out = {"result": {
        "elements": {"Fe": {"coverage_fraction": 0.6, "hotspots": [1]}},
    }}
    sc = vh._estimate_data_confidence(out)
    assert sc["confidence"] > 0


def test_confidence_2d_phase_field():
    out = {"result": {
        "volume_fractions": {"a": 0.5},
        "morphology": {"a": {"n_domains": 5, "top_domain_centroids_px": [[1, 2]]}},
    }}
    sc = vh._estimate_data_confidence(out)
    assert sc["confidence"] > 0


def test_confidence_result_not_dict():
    assert vh._estimate_data_confidence({"result": "x"}) == {}


def test_confidence_nested_bands_skipped():
    # 嵌套 bands → 1D 循环跳过, 不估
    out = {"result": {"bands": [[1.0, 2.0]]}}
    sc = vh._estimate_data_confidence(out)
    assert "confidence" in sc


def test_confidence_low_snr_and_high_anomaly():
    # 噪声大 → low_snr caveat; 高异常率 → high_anomaly caveat
    rng = np.random.default_rng(0)
    nums = rng.normal(0, 5, size=30).tolist()
    nums[0] = 100.0  # 制造离群
    out = {"result": {"dos": nums}}
    sc = vh._estimate_data_confidence(out)
    caveats = sc["caveats"]
    assert any("low_snr" in c for c in caveats) or any("high_anomaly" in c for c in caveats)


def test_confidence_eds_no_hotspots_low_coverage():
    out = {"result": {"elements": {"Fe": {"coverage_fraction": 0.01}}}}
    sc = vh._estimate_data_confidence(out)
    assert any("low_coverage_eds" in c for c in sc["caveats"])


def test_confidence_phase_field_no_domains():
    out = {"result": {"volume_fractions": {"a": 0.5}, "morphology": {"a": {"n_domains": 0}}}}
    sc = vh._estimate_data_confidence(out)
    assert any("no_domains_phase_field" in c for c in sc["caveats"])


# ── enrich_with_visual ────────────────────────────────────────────────────


def test_enrich_not_visualizable_returns_same():
    out = {"result": {"foo": 1}}
    assert vh.enrich_with_visual("foo", out) is out


def test_enrich_adds_hint_and_selfcheck():
    out = {"result": {"energies": [1.0, 2.0, 3.0]}}
    res = vh.enrich_with_visual("thermo_tool", out)
    assert "_visual_hint" in res
    assert "_visual_self_check" in res
    assert "_visual_base64" in res


# ── extract_box_primitives ────────────────────────────────────────────────


def test_extract_box_primitives_regions():
    img = Image.new("L", (100, 100), 255)
    for cx, cy in [(10, 10), (50, 50)]:
        for x in range(cx, cx + 10):
            for y in range(cy, cy + 10):
                img.putpixel((x, y), 0)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    out = vh.extract_box_primitives(buf.getvalue(), threshold=128)
    assert "[boxes]" in out
    assert "<box>" in out


def test_extract_box_primitives_white_returns_empty():
    img = Image.new("L", (50, 50), 255)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    assert vh.extract_box_primitives(buf.getvalue(), threshold=128) == ""


def test_extract_box_primitives_import_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "scipy":
            raise ImportError("no scipy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert vh.extract_box_primitives(b"abc") == ""


def test_extract_box_primitives_no_regions_after_filter():
    img = Image.new("L", (100, 100), 255)
    img.putpixel((50, 50), 0)  # 1px → 小于 min_area
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    assert vh.extract_box_primitives(buf.getvalue(), threshold=128, min_area_px2=10) == ""


# ── parse_box_primitive ───────────────────────────────────────────────────


def test_parse_box_primitive():
    text = (
        "<box>[100,200,300,400]</box>(overall:5000px²), "
        "<box>[150,250,280,380]</box>(region1:1200px²), <box>[1,2,3,4]</box>"
    )
    parsed = vh.parse_box_primitive(text)
    assert len(parsed) == 3
    assert parsed[0]["coordinates"] == [100, 200, 300, 400]
    assert parsed[0]["label"] == "overall"
    assert parsed[0]["area_px2"] == 5000
    assert parsed[2]["label"] == ""
    assert parsed[2]["area_px2"] is None


def test_parse_box_primitive_empty():
    assert vh.parse_box_primitive("") == []
    assert vh.parse_box_primitive("no box here") == []


# ── extract_point3d_primitives ────────────────────────────────────────────


def test_extract_point3d_normalized():
    coords = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
    out = vh.extract_point3d_primitives(coords, species=["Fe", "O"])
    lines = out.split("\n")
    assert "(Fe)" in lines[0]
    assert "999" in lines[1]  # x 归一化到 999


def test_extract_point3d_no_normalize():
    coords = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    out = vh.extract_point3d_primitives(coords, normalize_to=None)
    assert "[1,2,3]" in out


def test_extract_point3d_labels_fallback():
    out = vh.extract_point3d_primitives([[0.0, 0.0, 0.0]], labels=["atomX"])
    assert "(atomX)" in out
    out2 = vh.extract_point3d_primitives([[0.0, 0.0, 0.0]])
    assert "(atom0)" in out2


def test_extract_point3d_invalid():
    assert vh.extract_point3d_primitives([]) == ""
    assert vh.extract_point3d_primitives([[1, 2]]) == ""
    assert vh.extract_point3d_primitives("not valid") == ""


# ── parse_point3d_primitive ───────────────────────────────────────────────


def test_parse_point3d():
    parsed = vh.parse_point3d_primitive("atom at <point3d>[-500,200,0]</point3d>(C)")
    assert len(parsed) == 1
    assert parsed[0]["coordinates"] == [-500, 200, 0]
    assert parsed[0]["label"] == "C"


def test_parse_point3d_no_label():
    parsed = vh.parse_point3d_primitive("<point3d>[100,200,300]</point3d>")
    assert len(parsed) == 1 and parsed[0]["label"] == ""


def test_parse_point3d_empty():
    assert vh.parse_point3d_primitive("") == []


# ── _selfcheck ────────────────────────────────────────────────────────────


def test_selfcheck_runs():
    vh._selfcheck()


# ── 覆盖率 93%→更高 补测 ─────────────────────────────────────────────────


def test_extract_2d_elements_non_dict():
    # elements 值非 dict → 跳过该元素
    result = {"elements": {"Fe": "not-a-dict", "O": {"centroid_px": [0, 0], "coverage_fraction": 0.1}}}
    lines = vh._extract_2d_primitives(result)
    assert any("O:" in l for l in lines)


def test_extract_2d_hotspot_non_dict():
    # hotspots 列表里混入非 dict 元素 → 跳过
    result = {
        "elements": {"Fe": {"centroid_px": [10, 10], "coverage_fraction": 0.3,
                            "hotspots": [5, {"centroid_px": [20, 20], "area_px2": 9}]}},
        "image_shape": [100, 100],
    }
    lines = vh._extract_2d_primitives(result)
    assert any("hotspot" in l for l in lines)


def test_extract_2d_centroid_exception():
    # centroid_px 非数值 → except 分支跳过该元素
    result = {"elements": {"Fe": {"centroid_px": "bad", "coverage_fraction": 0.3}}}
    lines = vh._extract_2d_primitives(result)
    assert "Fe:" not in "\n".join(lines)


def test_extract_2d_vol_frac_exception():
    # volume_fractions 值无法转 float → except
    result = {"volume_fractions": {"a": "bad", "b": 0.5}}
    lines = vh._extract_2d_primitives(result)
    assert any("b=<point>" in l for l in lines)


def test_extract_2d_morphology_non_dict():
    # morphology 值非 dict → 跳过该相
    result = {"volume_fractions": {"a": 0.5}, "morphology": {"a": "not-dict"}}
    lines = vh._extract_2d_primitives(result)
    assert not any("domains=" in l for l in lines)


def test_extract_2d_morphology_centroid_exception():
    # top_domain_centroids 非数值 → except
    result = {
        "volume_fractions": {"a": 0.5},
        "image_shape": [100, 100],
        "morphology": {"a": {"n_domains": 1, "top_domain_centroids_px": [["x", "y"]]}},
    }
    lines = vh._extract_2d_primitives(result)
    assert any("domains=1" in l for l in lines)


def test_extract_comparative_non_numeric():
    # bl/cr data 全非数值 → nums 空 → continue, 无输出
    bl = {"result": {"dos": ["a", "b"]}}
    cr = {"result": {"dos": ["c", "d"]}}
    assert vh.extract_comparative_primitives(bl, cr) == ""


def test_extract_comparative_2d_element_non_dict():
    # 元素 stats 非 dict → 跳过
    bl = {"result": {"elements": {"Fe": "x"}, "image_shape": [10, 10]}}
    cr = {"result": {"elements": {"Fe": "y"}, "image_shape": [10, 10]}}
    assert not any("centroid_shift" in l for l in vh.extract_comparative_primitives(bl, cr).split("\n"))


def test_extract_comparative_2d_centroid_exception():
    # centroid_px 非数值 → except
    bl = {"result": {"elements": {"Fe": {"centroid_px": "x"}}, "image_shape": [10, 10]}}
    cr = {"result": {"elements": {"Fe": {"centroid_px": "y"}}, "image_shape": [10, 10]}}
    assert not any("centroid_shift" in l for l in vh.extract_comparative_primitives(bl, cr).split("\n"))


def test_extract_comparative_2d_pf_exception():
    # phase 值无法转 float → except
    bl = {"result": {"volume_fractions": {"a": "bad"}}}
    cr = {"result": {"volume_fractions": {"a": 0.5}}}
    assert not any("a:" in l for l in vh.extract_comparative_primitives(bl, cr).split("\n"))


def test_confidence_all_non_numeric():
    # data 全非数值 → nums 空 → continue (786 分支)
    out = {"result": {"dos": ["x", "y"]}}
    sc = vh._estimate_data_confidence(out)
    assert "confidence" in sc


def test_confidence_few_points_mid():
    # n in [5,19] → few_points caveat (799-800)
    out = {"result": {"dos": [float(i) for i in range(10)]}}
    sc = vh._estimate_data_confidence(out)
    assert any("few_points_dos" in c for c in sc["caveats"])


def test_confidence_high_anomaly():
    # 离群点 >10% → high_anomaly caveat (810)
    out = {"result": {"dos": [-100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    sc = vh._estimate_data_confidence(out)
    assert any("high_anomaly_rate" in c for c in sc["caveats"])


def test_confidence_no_elements_eds():
    # elements 存在但值全非 dict → n_elems=0 → no_elements caveat (832)
    out = {"result": {"elements": {"Fe": "x", "O": "y"}}}
    sc = vh._estimate_data_confidence(out)
    assert any("no_elements_eds" in c for c in sc["caveats"])


def test_box_rgb_image_converts():
    # RGB 图 → convert("L") 分支 (900)
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    for x in range(10, 20):
        for y in range(10, 20):
            img.putpixel((x, y), (0, 0, 0))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    out = vh.extract_box_primitives(buf.getvalue(), threshold=128)
    assert "[boxes]" in out


def test_box_max_boxes_break():
    # 超过 max_boxes 个连通域 → break (931)
    img = Image.new("L", (200, 200), 255)
    for i in range(6):
        cx = 10 + i * 30
        for x in range(cx, cx + 10):
            for y in range(10, 20):
                img.putpixel((x, y), 0)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    out = vh.extract_box_primitives(buf.getvalue(), threshold=128, max_boxes=2)
    boxes = vh.parse_box_primitive(out)
    assert len(boxes) <= 3  # 1 overall + up to max_boxes regions


def test_box_invalid_bytes_exception():
    # 非图片字节 → 整体 except → 返回空 (949-951)
    assert vh.extract_box_primitives(b"not an image at all") == ""


def test_point3d_asarray_error():
    # 不规则嵌套 → np.asarray(dtype=float) 抛 ValueError → 返回空 (1018-1020)
    assert vh.extract_point3d_primitives([[1, 2], [3]]) == ""


def test_derivative_short_direct():
    # 直接调用 n<5 → 提前返回 (508)
    assert vh._extract_derivative_primitives([1.0, 2.0, 3.0], "dos") == []


def test_box_image_zero_dim(monkeypatch):
    # img 解码后 shape 为 (0, w) → h==0 → 返回空 (904)
    import numpy as _np
    monkeypatch.setattr(_np, "asarray", lambda img: _np.zeros((0, 10)))
    img = Image.new("L", (100, 100), 255)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    assert vh.extract_box_primitives(buf.getvalue(), threshold=128) == ""


def test_box_find_objects_none_slice(monkeypatch):
    # find_objects 返回的 tuple 含 None → sl is None → continue (915)
    import scipy.ndimage as ndimage
    real = ndimage.find_objects

    def _fake(labeled):
        return (None, real(labeled)[0])

    monkeypatch.setattr(ndimage, "find_objects", _fake)
    img = Image.new("L", (100, 100), 255)
    for x in range(10, 20):
        for y in range(10, 20):
            img.putpixel((x, y), 0)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    out = vh.extract_box_primitives(buf.getvalue(), threshold=128)
    assert "[boxes]" in out


def test_point3d_import_numpy_missing(monkeypatch):
    # import numpy 抛 ImportError → 返回空 (1018-1020)
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert vh.extract_point3d_primitives([[0.0, 0.0, 0.0]]) == ""


def test_point3d_empty_label_else():
    # species 提供空字符串 label → else 分支, 无 (label) 后缀 (1050)
    out = vh.extract_point3d_primitives([[0.0, 0.0, 0.0]], species=[""])
    assert "<point3d>[0,0,0]</point3d>" in out
    assert "(atom0)" not in out
