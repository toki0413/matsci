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
        assert (
            result["stable"] is True
        ), f"Mg hcp 应稳定, 判据结果: {result['criteria']}"
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

    def test_triclinic_implemented(self):
        """triclinic 已实现 (主子式判据), 应返回 bool 不是 None."""
        C = np.eye(6) * 100.0  # 对角占优 → 稳定
        result = BornStabilityChecker.check(C, crystal_system="triclinic")
        assert result["stable"] is True


# ── 全 7 晶系覆盖扩展 (原 test_mechanics_debt.py) ────────────────────────

def _make_stable_cubic():
    """A clearly stable cubic tensor."""
    return np.array([
        [250, 100, 100, 0, 0, 0],
        [100, 250, 100, 0, 0, 0],
        [100, 100, 250, 0, 0, 0],
        [0, 0, 0, 120, 0, 0],
        [0, 0, 0, 0, 120, 0],
        [0, 0, 0, 0, 0, 120],
    ], dtype=float)


def _make_unstable_cubic():
    """A clearly unstable cubic tensor (negative C44)."""
    C = _make_stable_cubic()
    C[3, 3] = -10
    C[4, 4] = -10
    C[5, 5] = -10
    return C


def _make_stable_tetragonal():
    """Stable tetragonal (C11=C22, C13=C23, C44=C55)."""
    return np.array([
        [300, 150, 100, 0, 0, 0],
        [150, 300, 100, 0, 0, 0],
        [100, 100, 400, 0, 0, 0],
        [0, 0, 0, 100, 0, 0],
        [0, 0, 0, 0, 100, 0],
        [0, 0, 0, 0, 0, 80],
    ], dtype=float)


def _make_unstable_tetragonal():
    """Unstable tetragonal (C11-C12 < 0)."""
    C = _make_stable_tetragonal()
    C[0, 1] = 350  # C12 > C11 → C11-C12 < 0
    C[1, 0] = 350
    return C


def _make_stable_trigonal():
    """Stable trigonal (7 independent constants)."""
    return np.array([
        [300, 100, 80, 5, 0, 0],
        [100, 300, 80, -5, 0, 0],
        [80, 80, 350, 0, 0, 0],
        [5, -5, 0, 80, 0, 0],
        [0, 0, 0, 0, 80, 0],
        [0, 0, 0, 0, 0, 100],
    ], dtype=float)


def _make_unstable_trigonal():
    """Unstable trigonal (C14 too large)."""
    C = _make_stable_trigonal()
    C[0, 3] = 200  # Huge C14 coupling
    C[3, 0] = 200
    C[1, 3] = -200
    C[3, 1] = -200
    return C


def _make_stable_monoclinic():
    """Stable monoclinic (13 independent constants, b-axis unique)."""
    return np.array([
        [250, 80, 60, 0, 10, 0],
        [80, 220, 50, 0, -5, 0],
        [60, 50, 280, 0, 8, 0],
        [0, 0, 0, 90, 0, 0],
        [10, -5, 8, 0, 70, 0],
        [0, 0, 0, 0, 0, 65],
    ], dtype=float)


def _make_unstable_monoclinic():
    """Unstable monoclinic (negative leading minor)."""
    C = _make_stable_monoclinic()
    C[0, 0] = -10  # C11 < 0 → M1 < 0
    return C


class TestCubic:
    def test_stable_cubic(self):
        r = BornStabilityChecker.check(_make_stable_cubic(), crystal_system="cubic")
        assert r["stable"] is True
        assert r["crystal_system"] == "cubic"
        assert len(r["criteria"]) == 3

    def test_unstable_cubic(self):
        r = BornStabilityChecker.check(_make_unstable_cubic(), crystal_system="cubic")
        assert r["stable"] is False


class TestHexagonal:
    def test_stable_hexagonal(self):
        C = np.array([
            [300, 100, 80, 0, 0, 0],
            [100, 300, 80, 0, 0, 0],
            [80, 80, 350, 0, 0, 0],
            [0, 0, 0, 100, 0, 0],
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 100],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="hexagonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "hexagonal"


class TestTetragonal:
    """P2 fix: tetragonal criteria newly implemented."""

    def test_stable_tetragonal(self):
        r = BornStabilityChecker.check(_make_stable_tetragonal(), crystal_system="tetragonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "tetragonal"
        assert len(r["criteria"]) == 7

    def test_unstable_tetragonal(self):
        r = BornStabilityChecker.check(_make_unstable_tetragonal(), crystal_system="tetragonal")
        assert r["stable"] is False
        failing = [c for c in r["criteria"] if not c["passed"]]
        assert any("C11 - C12" in c["name"] for c in failing)


class TestTrigonal:
    """P2 fix: trigonal criteria newly implemented."""

    def test_stable_trigonal(self):
        r = BornStabilityChecker.check(_make_stable_trigonal(), crystal_system="trigonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "trigonal"
        assert len(r["criteria"]) == 8

    def test_unstable_trigonal(self):
        r = BornStabilityChecker.check(_make_unstable_trigonal(), crystal_system="trigonal")
        assert r["stable"] is False


class TestMonoclinic:
    """P2 fix: monoclinic criteria newly implemented."""

    def test_stable_monoclinic(self):
        r = BornStabilityChecker.check(_make_stable_monoclinic(), crystal_system="monoclinic")
        assert r["stable"] is True
        assert r["crystal_system"] == "monoclinic"
        assert len(r["criteria"]) == 6  # 6 leading principal minors

    def test_unstable_monoclinic(self):
        r = BornStabilityChecker.check(_make_unstable_monoclinic(), crystal_system="monoclinic")
        assert r["stable"] is False
        failing = [c for c in r["criteria"] if not c["passed"]]
        assert any("M1" in c["name"] for c in failing)


class TestOrthorhombic:
    def test_stable_orthorhombic(self):
        C = np.array([
            [250, 80, 60, 0, 0, 0],
            [80, 220, 50, 0, 0, 0],
            [60, 50, 280, 0, 0, 0],
            [0, 0, 0, 90, 0, 0],
            [0, 0, 0, 0, 70, 0],
            [0, 0, 0, 0, 0, 65],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="orthorhombic")
        assert r["stable"] is True


class TestTriclinic:
    def test_stable_triclinic(self):
        C = np.array([
            [250, 10, 10, 5, 5, 5],
            [10, 240, 8, 4, 3, 4],
            [10, 8, 260, 3, 6, 2],
            [5, 4, 3, 90, 2, 1],
            [5, 3, 6, 2, 80, 1],
            [5, 4, 2, 1, 1, 70],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="triclinic")
        assert r["stable"] is True
        assert len(r["criteria"]) == 6


class TestUnknownCrystalSystem:
    """Verify unimplemented crystal systems return stable=None, not False."""

    def test_unknown_system_returns_none(self):
        C = np.eye(6) * 100
        r = BornStabilityChecker.check(C, crystal_system="nonexistent")
        assert r["stable"] is None
        assert "not implemented" in r.get("error", "")


class TestAutoDetection:
    def test_auto_detect_cubic(self):
        r = BornStabilityChecker.check(_make_stable_cubic(), crystal_system="auto")
        assert r["crystal_system"] in ("cubic", "hexagonal", "triclinic")
