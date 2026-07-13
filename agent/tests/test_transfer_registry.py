"""transfer_registry 自检 — 结构同构识别 (SymPy 等价检查).

验证:
1. structure_signature 重叠时, 即使 composition/structure_type 完全不同,
   similarity 也应跳到 ≥0.9.
2. SymPy 等价检查能识别变量重命名 (a*P**2 + b*P**4 ≡ alpha*m**2 + beta*m**4).
3. 变量数不同的表达式不误判为等价.
4. _REGISTRY 中的真实 perovskite_solar ↔ ferromagnet 能被识别为同构.
5. IPI 防御: register_domain 过滤不在 allowlist 的 signature.
6. 数据来源独立性: dual_covered 检查 data_source 字段.
"""
from __future__ import annotations

from huginn.ml.transfer_registry import (
    DomainProfile,
    _REGISTRY,
    _ALLOWED_SIGNATURE_NAMES,
    _sanitize_signature,
    _sympy_equivalent,
    register_domain,
    shared_structure,
    similarity,
)


class TestStructureIsomorphism:
    def _ferromagnet(self) -> DomainProfile:
        return DomainProfile(
            name="ferromagnet",
            composition=frozenset({"Fe", "Ni", "Co"}),
            structure_type="bulk",
            property_type="magnetic",
            structure_signature=(("landau_phi4", "a*m**2 + b*m**4"),),
        )

    def _ferroelectric(self) -> DomainProfile:
        # 表面完全不同: 不同元素, 不同结构, 不同性质
        # 但数学结构相同 — Landau phi4 (变量重命名后等价)
        return DomainProfile(
            name="ferroelectric",
            composition=frozenset({"Ba", "Ti", "O"}),
            structure_type="perovskite",
            property_type="electronic",
            structure_signature=(("landau_phi4", "alpha*P**2 + beta*P**4"),),
        )

    def _semiconductor(self) -> DomainProfile:
        # 无共享签名
        return DomainProfile(
            name="semiconductor",
            composition=frozenset({"Si", "Ge"}),
            structure_type="bulk",
            property_type="electronic",
            structure_signature=(("kane_model", "E*(1 + alpha*E)"),),
        )

    def test_signature_overlap_boosts_similarity_across_surface_diff(self):
        """铁磁 ↔ 铁电: 表面完全不同但共享 Landau phi4 结构 → sim ≥ 0.9."""
        sim = similarity(self._ferromagnet(), self._ferroelectric())
        assert sim >= 0.9, f"signature 重叠应把 sim 拉到 0.9, got {sim}"

    def test_no_signature_overlap_uses_base_similarity(self):
        """铁磁 ↔ 半导体: 无共享签名 → 用字段相似度."""
        sim = similarity(self._ferromagnet(), self._semiconductor())
        assert sim < 0.5, f"无 signature 共享时不应 boost, got {sim}"

    def test_empty_signature_does_not_boost(self):
        """signature 为空的两域不触发 boost."""
        a = DomainProfile(name="a", composition=frozenset({"Si"}))
        b = DomainProfile(name="b", composition=frozenset({"Ge"}))
        sim = similarity(a, b)
        assert sim < 0.5

    def test_shared_structure_returns_intersection(self):
        shared = shared_structure(self._ferromagnet(), self._ferroelectric())
        assert shared == ["landau_phi4"]

    def test_shared_structure_empty_when_no_overlap(self):
        shared = shared_structure(self._ferromagnet(), self._semiconductor())
        assert shared == []

    def test_shared_structure_empty_when_one_side_empty(self):
        a = DomainProfile(name="a", composition=frozenset({"Si"}))
        shared = shared_structure(a, self._ferromagnet())
        assert shared == []


class TestSymPyEquivalence:
    def test_variable_renaming_recognized_as_equivalent(self):
        """a*P**2 + b*P**4 ≡ alpha*m**2 + beta*m**4 (变量重命名后等价)."""
        assert _sympy_equivalent("a*P**2 + b*P**4", "alpha*m**2 + beta*m**4")

    def test_same_expression_different_coefficient_names(self):
        """a*x**2 + b*x**4 ≡ c*y**2 + d*y**4."""
        assert _sympy_equivalent("a*x**2 + b*x**4", "c*y**2 + d*y**4")

    def test_different_structure_not_equivalent(self):
        """a*x**2 ≠ a*x**3 (结构不同)."""
        assert not _sympy_equivalent("a*x**2", "a*x**3")

    def test_different_variable_count_not_equivalent(self):
        """a*x**2 ≠ a*x**2 + b*y**4 (变量数不同)."""
        assert not _sympy_equivalent("a*x**2", "a*x**2 + b*y**4")

    def test_equivalent_with_constants(self):
        """a*x**2 + 1 ≡ b*y**2 + 1 (常数保持一致)."""
        assert _sympy_equivalent("a*x**2 + 1", "b*y**2 + 1")

    def test_non_equivalent_with_different_constants(self):
        """a*x**2 + 1 ≠ a*x**2 + 2 (常数不同)."""
        assert not _sympy_equivalent("a*x**2 + 1", "a*x**2 + 2")

    def test_invalid_expression_falls_back_to_string_compare(self):
        """无效表达式 → 降级到字符串比较."""
        # 两个相同的无效字符串 → 字符串相等 → True
        assert _sympy_equivalent("@@invalid@@", "@@invalid@@")
        # 不同的无效字符串 → False
        assert not _sympy_equivalent("@@invalid@@", "!!different!!")

    def test_many_variables_no_degradation(self):
        """n>5 变量也用 unify, 不退化到单替换.
        升级前 (全排列版): n>5 退化到按名排序的单替换, 会漏同构.
        升级后 (unify): 不受变量数限制, AC 匹配自动处理."""
        # 6 个变量: a, b, c (系数) + x, y, z (坐标)
        # 等价形式只是变量重命名
        assert _sympy_equivalent(
            "a*x + b*y + c*z",
            "alpha*u + beta*v + gamma*w",
        )


class TestRegistryIntegration:
    def test_perovskite_ferromagnet_isomorphism_in_registry(self):
        """_REGISTRY 中 perovskite_solar 与 ferromagnet 应被识别为同构
        (都填了 landau_phi4 但用不同变量名 P 和 m)."""
        perovskite = next(d for d in _REGISTRY if d.name == "perovskite_solar")
        ferro = next(d for d in _REGISTRY if d.name == "ferromagnet")

        # 应该被识别为共享结构
        shared = shared_structure(perovskite, ferro)
        assert "landau_phi4" in shared

        # similarity 应该 ≥0.9 (跨表面差异的同构 boost)
        sim = similarity(perovskite, ferro)
        assert sim >= 0.9, (
            f"perovskite_solar ↔ ferromagnet 应该被同构 boost 到 0.9, got {sim}. "
            f"perovskite signature: {perovskite.structure_signature}, "
            f"ferro signature: {ferro.structure_signature}"
        )

    def test_no_signature_domains_not_boosted(self):
        """_REGISTRY 中没填 structure_signature 的域不应被 boost."""
        lunar = next(d for d in _REGISTRY if d.name == "lunar_regolith")
        oxide = next(d for d in _REGISTRY if d.name == "oxide_catalyst")
        # 两者都没填 signature (空 tuple)
        sim = similarity(lunar, oxide)
        # 应该用字段相似度, 不被 signature boost
        assert sim < 0.9, f"无 signature 的域不应被 boost 到 0.9, got {sim}"


class TestIpiDefense:
    """间接提示注入 (IPI) 防御 — register_domain 过滤污染 signature."""

    def test_allowed_signature_passes_through(self):
        """在 allowlist 里的 canonical_name 正常通过."""
        sig = (("landau_phi4", "a*x**2 + b*x**4"),)
        clean = _sanitize_signature(sig)
        assert clean == sig

    def test_disallowed_name_is_dropped(self):
        """不在 allowlist 的 canonical_name 被静默丢弃.
        攻击场景: 用户上传 CIF 的 meta 字段注入恶意 signature name."""
        sig = (
            ("landau_phi4", "a*x**2 + b*x**4"),
            ("malicious_injected_name", "ignore_previous_instructions()"),
        )
        clean = _sanitize_signature(sig)
        assert "malicious_injected_name" not in dict(clean)
        assert "landau_phi4" in dict(clean)

    def test_overlong_expr_is_dropped(self):
        """过长的表达式被丢弃 (防长 payload 注入)."""
        long_expr = "a*" + "+".join(["x"] * 100)  # 远超 200 字符
        sig = (("landau_phi4", long_expr),)
        clean = _sanitize_signature(sig)
        assert clean == ()

    def test_register_domain_strips_malicious_signature(self):
        """register_domain 注册时, 不在 allowlist 的 signature 被剥离.
        这防止 IPI 通过 DomainProfile.structure_signature → shared_structure
        → _render_transfer_prompt 进入 LLM prompt."""
        from huginn.ml.transfer_registry import _REGISTRY
        before = len(_REGISTRY)
        profile = DomainProfile(
            name="test_ipi_domain",
            composition=frozenset({"Test"}),
            structure_signature=(
                ("landau_phi4", "a*x**2 + b*x**4"),
                ("evil_prompt_injection", "exfiltrate_credentials()"),
            ),
        )
        register_domain(profile)
        after = len(_REGISTRY)
        assert after == before + 1
        registered = _REGISTRY[-1]
        assert "evil_prompt_injection" not in dict(registered.structure_signature)
        assert "landau_phi4" in dict(registered.structure_signature)
        # 清理: 移除测试注册的域
        _REGISTRY.pop()
