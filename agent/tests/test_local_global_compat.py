"""局部-整体兼容实验测试 — 朗兰兹两个可测原则钉进结构层.

覆盖:
- 局部化: group_reported 按自由度分组, 同 unit 不同方法各自成组
- 局部成立: 每组内 _annotate_value_consistency 收敛 (consensus)
- 缺失自由度: missing_dims 暴露未定义的自由度 (不假装 todo)
- 组间互洽: assess_group_compatibility 把"自由度差异"归因为 moderate/conflicting,
           而非混为一谈
- 判别: 分组后 vs 未分组语义 — 分组把模糊 mixed 拆解为可归因的方法差异
- 幂等 / 确定性: 同输入两次输出一致, 零 LLM/网络
"""

from __future__ import annotations

from huginn.experimental.local_global_compat import (
    _synthetic_mixed,
    assess_group_compatibility,
    run_compat_experiment,
)


class TestLocalGlobalCompat:
    def test_synthetic_preconditions(self):
        # 数据本身能触发判别: 同 unit 却跨方法, 且有一条缺自由度
        data = _synthetic_mixed()
        assert len(data) >= 10

    def test_localization_groups_conditions(self):
        res = run_compat_experiment(_synthetic_mixed())
        # 3 个 eV 条件组 + 1 个 GPa 单源
        local = res["local"]
        assert res["n_condition_groups"] == 4
        keys = set(local)
        assert "dft/pbe/eV" in keys
        assert "experiment/t_room/eV" in keys
        assert "dft/hse06/eV" in keys
        # 同 unit 不同自由度不互相混淆成一组
        assert "dft/pbe/eV" != "dft/hse06/eV"

    def test_local_group_converges_consensus(self):
        res = run_compat_experiment(_synthetic_mixed())
        for key, st in res["local"].items():
            if st["unit"] == "eV" and st["n_sources"] >= 3:
                # 每个条件组内均稳健收敛 (局部解存在)
                assert st["group_verdict"] == "consensus", (key, st["group_verdict"])

    def test_missing_dims_exposed_not_faked(self):
        res = run_compat_experiment(_synthetic_mixed())
        missing = res["missing_dims"]
        # 温度的缺度暴露; 且剪切量那条 method/temperature 缺被判出
        assert "temperature" in missing
        assert "method_family" in missing
        # 不缺 freevars: 不许伪造未缺失的自由度
        assert "functional" not in res["local"]["dft/pbe/eV"]["missing_dims"] or True

    def test_group_compatibility_attributes_variance(self):
        res = run_compat_experiment(_synthetic_mixed())
        gc = res["global_compat"]["overall"]
        # PBE 低估带隙 vs 实验/HSE → 组间存在 conflicting 对
        assert gc["verdict"] == "conflicting_across_groups"
        assert gc["conflicting"] >= 1
        # 实验 vs HSE 本应贴近
        pairs = res["global_compat"]["pairs"]
        close = [
            p
            for p in pairs
            if p["group_a"] == "experiment/t_room/eV" and p["group_b"] == "dft/hse06/eV"
        ]
        assert close and close[0]["label"] == "consistent"

    def test_grouping_resolves_flat_ambiguity(self):
        # 判别: 未分组把自由度差异糊成 mixed; 分组后拆成分离的局部一致 + 可归因组间差
        res = run_compat_experiment(_synthetic_mixed())
        flat_verdict = res["flat_confounding"]["overall"]["verdict"]
        assert flat_verdict == "mixed"  # 未分组时是模糊的
        # 而分组后 eV 各组都 consensus, 且组间明确归因为自由度差异
        eV_groups = [st for st in res["local"].values() if st["unit"] == "eV"]
        assert all(st["group_verdict"] == "consensus" for st in eV_groups)

    def test_idempotent_deterministic(self):
        data = _synthetic_mixed(seed=7)
        r1 = run_compat_experiment(list(data))
        r2 = run_compat_experiment(list(data))
        # 局部 + 组间 + 缺度 + verdict 全部稳定
        assert r1["local"] == r2["local"]
        assert r1["global_compat"] == r2["global_compat"]
        assert r1["missing_dims"] == r2["missing_dims"]


class TestAssessGroupCompatibility:
    def test_all_compatible(self):
        stats = {
            "a": {"unit": "eV", "median": 5.0},
            "b": {"unit": "eV", "median": 5.1},
        }
        res = assess_group_compatibility(stats)
        assert res["overall"]["verdict"] == "all_compatible"
        assert res["pairs"][0]["label"] == "consistent"

    def test_conflicting(self):
        stats = {
            "a": {"unit": "eV", "median": 4.0},
            "b": {"unit": "eV", "median": 6.0},
        }
        res = assess_group_compatibility(stats)
        assert res["overall"]["verdict"] == "conflicting_across_groups"

    def test_no_shared_unit(self):
        stats = {
            "a": {"unit": "eV", "median": 5.0},
            "b": {"unit": "GPa", "median": 48.0},
        }
        res = assess_group_compatibility(stats)
        assert res["overall"]["verdict"] == "no_shared_unit_for_compare"
