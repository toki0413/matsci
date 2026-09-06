"""结构层 condition 归一测试 — 纯规则, 零 LLM, 断言缺自由度诊断.

覆盖:
- normalize_method_family: DFT / 实验 / MD / unknown 分类, HSE06 数字后缀不漏判
- normalize_functional: PBE/HSE06 提取
- normalize_temperature: 0K / 室温 / 数值 / unknown; 不许把方法数字当温度
- condition_key: 组合 + 缺自由度 (missing_dims) 暴露
- group_reported: 替代纯 unit 分组, 结构层稳定
- 幂等: 同输入两次输出完全一致
"""

from __future__ import annotations

from huginn.tools.literature.condition_normalize import (
    condition_key,
    feature_vector,
    group_reported,
    normalize_functional,
    normalize_method_family,
    normalize_temperature,
)


class TestMethodFamily:
    def test_dft_pbe(self):
        assert normalize_method_family("DFT-PBE") == "dft"

    def test_hse06_digit_suffix_not_missed(self):
        # 不允许 \b 断词把 HSE06 漏判成 unknown
        assert normalize_method_family("HSE06") == "dft"

    def test_vasp_software_name(self):
        assert normalize_method_family("VASP") == "dft"

    def test_experiment(self):
        assert normalize_method_family("experiment") == "experiment"
        assert normalize_method_family("experimental measurement") == "experiment"

    def test_md(self):
        assert normalize_method_family("basin-hopping") == "md"
        assert normalize_method_family("MD") == "md"

    def test_empty_unknown(self):
        assert normalize_method_family("") == "unknown"
        assert normalize_method_family("some random method") == "unknown"


class TestFunctional:
    def test_pbe(self):
        assert normalize_functional("DFT-PBE", "dft") == "pbe"

    def test_hse06(self):
        assert normalize_functional("HSE06", "dft") == "hse06"

    def test_non_dft_returns_none(self):
        assert normalize_functional("experiment", "experiment") is None


class TestTemperature:
    def test_zero_k(self):
        assert normalize_temperature("T=0K", "DFT-PBE") == "t_zero"

    def test_room_temperature_full_phrase(self):
        assert normalize_temperature("room temperature", "experiment") == "t_room"
        assert normalize_temperature("RT", "experiment") == "t_room"
        assert normalize_temperature("室温", "experiment") == "t_room"

    def test_numeric_band(self):
        assert normalize_temperature("T=300K", "") == "t_room"

    def test_method_digit_is_not_temperature(self):
        # HSE06 / PBE+U 里的数字不得被当成温度
        assert normalize_temperature("", "HSE06") == "unknown"

    def test_unknown(self):
        assert normalize_temperature("", "") == "unknown"


class TestConditionKey:
    def test_combines_dims(self):
        key, missing = condition_key("DFT-PBE", "T=0K", "eV")
        assert key == "dft/pbe/eV"
        assert missing == ""

    def test_room_temp_adds_dim(self):
        key, _ = condition_key("experiment", "room temperature", "eV")
        assert key == "experiment/t_room/eV"

    def test_missing_dims_reported_not_faked(self):
        # 纯 unit, 无 method 无温度: 自由度缺失显性暴露, 不塞进 key 当假同质
        _, missing = condition_key("", "", "eV")
        assert "method_family" in missing
        assert "temperature" in missing

    def test_hse_vs_pbe_distinct_groups(self):
        k_hse, _ = condition_key("HSE06", "", "eV")
        k_pbe, _ = condition_key("PBE", "", "eV")
        assert k_hse != k_pbe  # 不同泛函各自成组, 不伪装一致


class TestFeatureAndGroup:
    def test_feature_vector_fields(self):
        fv = feature_vector("DFT-PBE", "T=0K", "eV")
        assert fv["method_family"] == "dft"
        assert fv["functional"] == "pbe"
        assert fv["temperature"] == "t_zero"
        assert fv["condition_key"] == "dft/pbe/eV"
        assert fv["missing_dims"] == []

    def test_group_reported_splits_by_condition(self):
        reported = [
            {"value": 5.3, "unit": "eV", "method": "PBE", "note": ""},
            {"value": 6.1, "unit": "eV", "method": "HSE06", "note": ""},
            {"value": 5.2, "unit": "eV", "method": "PBE", "note": "T=0K"},
        ]
        groups = group_reported(reported)
        # PBE@0K 与 PBE@(default) 归同组 (0K 是 DFT 默认不单列); HSE06 独立
        pbe_keys = [k for k in groups if "hse" not in k]
        assert len(groups) == 2  # pbe 组 + hse 组
        assert len(pbe_keys) == 1
        assert len(groups[pbe_keys[0]]) == 2

    def test_idempotent(self):
        reported = [{"value": 5.3, "unit": "eV", "method": "PBE", "note": ""}]
        g1 = group_reported(reported)
        g2 = group_reported(reported)
        assert g1 == g2
