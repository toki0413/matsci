"""Born 稳定性判据测试.

审计 14号报告指出 _check_hexagonal 用了错误的旧判据 `C11*C33 > C13²`,
应为 Mouhat & Coudert PRB 90, 224104 (2014) Eq. (60):
    (C11+C12)*C33 > 2*C13²

差异在临界案例: 旧判据说稳定但新判据说不稳定的材料, 旧代码会漏判.
本测试用 Mg hcp 实验张量验证稳定案例, 用构造的临界张量抓漏判 bug,
并验证未实现晶系 (tetragonal 等) 返回 None 而非默认 False.
"""

from __future__ import annotations

import numpy as np

from huginn.mechanics import BornStabilityChecker


def _hex_tensor(C11, C12, C13, C33, C44):
    """构造 6x6 Voigt 张量 (hexagonal symmetry).

    Hexagonal 独立常数: C11=C22, C12, C13=C23, C33, C44=C55, C66=(C11-C12)/2.
    其余分量为 0.
    """
    C66 = (C11 - C12) / 2.0
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = C66
    return C


class TestHexagonalBornCriteria:
    """六方 Born 判据: Mouhat & Coudert PRB 2014 Eq. (60)."""

    def test_mg_hcp_stable(self):
        """Mg hcp 实验弹性常数 (Slutsky & Brofman 1964) → 稳定.

        C11=59.7, C12=26.2, C13=21.7, C33=61.8, C44=16.4 GPa.
        所有 Mouhat-Coudert 判据通过.
        """
        C = _hex_tensor(59.7, 26.2, 21.7, 61.8, 16.4)
        result = BornStabilityChecker.check(C, crystal_system="hexagonal")
        assert result["stable"] is True, (
            f"Mg hcp 应稳定, 判据结果: {result['criteria']}"
        )
        # 六方应有 6 个判据 (旧版只有 4 个)
        assert len(result["criteria"]) == 6

    def test_old_criteria_would_miss_critical_instability(self):
        """临界案例: 旧判据 C11*C33 > C13² 说稳定, 新判据说不稳定.

        构造 C13=52 GPa (其余同 Mg):
          旧判据: C11*C33 = 59.7*61.8 = 3689.46 > C13² = 2704 → 稳定 (错!)
          新判据: (C11+C12)*C33 = 85.9*61.8 = 5308.62
                  2*C13² = 2*2704 = 5408
                  5308.62 < 5408 → 不稳定 (对!)
        这个测试会 fail 如果有人把判据改回旧形式.
        """
        C = _hex_tensor(59.7, 26.2, 52.0, 61.8, 16.4)
        result = BornStabilityChecker.check(C, crystal_system="hexagonal")
        assert result["stable"] is False, (
            f"C13=52 应不稳定 (新判据), 但判为 stable={result['stable']}. "
            f"旧判据 C11*C33 > C13² 会漏判. "
            f"判据详情: {result['criteria']}"
        )
        # 定位是哪个判据 fail
        critical = next(
            (c for c in result["criteria"] if "(C11+C12)*C33" in c["name"]),
            None,
        )
        assert critical is not None, "缺少 (C11+C12)*C33 > 2*C13² 判据"
        # numpy.bool_ 不是 Python bool, 用 bool() 转换
        assert bool(critical["passed"]) is False

    def test_c11_plus_c12_negative_fails(self):
        """C11+C12 < 0 时应不稳定 (新加判据, 旧版没有)."""
        C = _hex_tensor(-10.0, 5.0, 1.0, 61.8, 16.4)
        result = BornStabilityChecker.check(C, crystal_system="hexagonal")
        assert result["stable"] is False

    def test_cubic_still_works(self):
        """cubic 判据未改, 回归测试."""
        # Iron: C11=230, C12=135, C44=117 GPa → 稳定
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = 230.0
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = 135.0
        C[3, 3] = C[4, 4] = C[5, 5] = 117.0
        result = BornStabilityChecker.check(C, crystal_system="cubic")
        assert result["stable"] is True


class TestUnimplementedCrystalSystem:
    """未实现晶系返回 None + error, 不默认判 unstable."""

    def test_tetragonal_returns_none_not_false(self):
        """tetragonal 未实现, 应返回 stable=None + error, 不是 stable=False.

        旧版返回 False 会让真实 tetragonal 材料被错误拒绝为"不稳定".
        None 让调用方知道"无法判断"而非"不稳定".
        """
        C = np.eye(6) * 100.0
        result = BornStabilityChecker.check(C, crystal_system="tetragonal")
        assert result["stable"] is None, (
            f"未实现晶系应返回 stable=None, got {result['stable']}"
        )
        assert "not implemented" in result["error"]
        assert "Mouhat" in result["error"] or "2014" in result["error"]

    def test_monoclinic_returns_none(self):
        C = np.eye(6) * 100.0
        result = BornStabilityChecker.check(C, crystal_system="monoclinic")
        assert result["stable"] is None

    def test_triclinic_implemented(self):
        """triclinic 已实现 (主子式判据), 应返回 bool 不是 None."""
        C = np.eye(6) * 100.0  # 对角占优 → 稳定
        result = BornStabilityChecker.check(C, crystal_system="triclinic")
        assert result["stable"] is True
