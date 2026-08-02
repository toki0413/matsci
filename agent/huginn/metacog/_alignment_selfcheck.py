"""AlignmentDataset + AlignmentFunction 自检 — assert 驱动, 失败即报错.

跑法: python -m huginn.metacog._alignment_selfcheck
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from huginn.metacog.alignment import AlignmentFunction
from huginn.metacog.alignment_dataset import AlignmentDataset


class FakeSource:
    name = "fake_source"
    dim = 4

    def encode(self, obj):
        return np.asarray(obj, dtype=float)


class FakeTarget:
    name = "fake_target"
    dim = 3

    def encode(self, obj):
        return np.asarray(obj, dtype=float)


def _check_dataset_roundtrip() -> None:
    ds = AlignmentDataset()
    sv = np.array([1.0, 2.0, 3.0, 4.0])
    tv = np.array([0.1, 0.2, 0.3])
    ds.add(sv, tv, "fake_source", "fake_target", metadata={"src": "dft"})

    X, y = ds.get_pairs("fake_source", "fake_target")
    assert X.shape == (1, 4), X.shape
    assert y.shape == (1, 3), y.shape
    assert np.allclose(X[0], sv)
    assert np.allclose(y[0], tv)
    assert ds.count() == 1
    assert ds.count("fake_source", "fake_target") == 1
    assert ds.count("nope", "nope") == 0

    # 不匹配的空间名返回空
    X2, y2 = ds.get_pairs("x", "y")
    assert X2.shape == (0, 0)
    assert y2.shape == (0, 0)

    # save/load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ds.json"
        ds.save(p)
        ds2 = AlignmentDataset.load(p)
        X3, y3 = ds2.get_pairs("fake_source", "fake_target")
        assert np.allclose(X3, X)
        assert np.allclose(y3, y)
        assert ds2.count() == 1
    print("[ok] AlignmentDataset add/get_pairs/save/load round-trip")


def _check_alignment_function() -> None:
    ds = AlignmentDataset()
    rng = np.random.default_rng(42)
    # 线性关系 + 噪声: target ≈ source[:3] * 2 + 1
    for _ in range(15):
        sv = rng.standard_normal(4)
        tv = sv[:3] * 2 + 1 + rng.standard_normal(3) * 0.1
        ds.add(sv, tv, "fake_source", "fake_target")

    af = AlignmentFunction(FakeSource(), FakeTarget(), min_samples=10)
    assert not af.ready
    af.fit(ds)
    assert af.ready

    test_vec = np.array([1.0, 2.0, 3.0, 4.0])
    mean, std = af.predict(test_vec)
    assert mean.shape == (3,), mean.shape
    assert std.shape == (3,), std.shape
    assert np.all(std > 0), std
    # GP 在 15 样本下外推会回退到 y 均值附近, 不强查精度 — 只验 shape + std>0.
    # 线性关系的学习质量由 surprise 单调性间接覆盖.

    # surprise: 近预测低, 远预测高
    s_low = af.surprise(test_vec, np.array([3.0, 5.0, 7.0]))
    s_high = af.surprise(test_vec, np.array([30.0, 50.0, 70.0]))
    assert s_high > s_low, (s_low, s_high)
    print(f"[ok] AlignmentFunction predict+surprise (low={s_low:.3f}, high={s_high:.3f})")

    # 数据不足 not ready
    ds_small = AlignmentDataset()
    for _ in range(5):
        ds_small.add(rng.standard_normal(4), rng.standard_normal(3),
                     "fake_source", "fake_target")
    af2 = AlignmentFunction(FakeSource(), FakeTarget(), min_samples=10)
    af2.fit(ds_small)
    assert not af2.ready
    # predict 未 ready 抛 RuntimeError
    try:
        af2.predict(test_vec)
        raise AssertionError("predict should raise when not ready")
    except RuntimeError:
        pass
    print("[ok] AlignmentFunction not-ready 退化 + RuntimeError")


def _check_surprise_uncertainty_monotone() -> None:
    """spec 要求: 高不确定性低 surprise, 低不确定性高 surprise.

    构造两个 dataset: 一个噪声大 (GP 不确定性高), 一个噪声小 (不确定性低).
    同样的偏差下, 噪声大的 surprise 应更低.
    """
    rng = np.random.default_rng(7)

    def _build(n: int, noise: float) -> AlignmentDataset:
        ds = AlignmentDataset()
        for _ in range(n):
            sv = rng.standard_normal(4)
            tv = sv[:3] + rng.standard_normal(3) * noise
            ds.add(sv, tv, "fake_source", "fake_target")
        return ds

    af_low_unc = AlignmentFunction(FakeSource(), FakeTarget(), min_samples=10)
    af_low_unc.fit(_build(20, 0.05))
    af_high_unc = AlignmentFunction(FakeSource(), FakeTarget(), min_samples=10)
    af_high_unc.fit(_build(20, 2.0))
    assert af_low_unc.ready and af_high_unc.ready

    test_vec = np.array([0.0, 0.0, 0.0, 0.0])
    # 同一偏差: actual 偏离预测 5 个单位
    _, std_low = af_low_unc.predict(test_vec)
    _, std_high = af_high_unc.predict(test_vec)
    # 高噪声 dataset 的 GP 预测不确定性应更大
    assert np.linalg.norm(std_high) > np.linalg.norm(std_low), (std_low, std_high)

    actual = np.array([5.0, 5.0, 5.0])
    s_low_unc = af_low_unc.surprise(test_vec, actual)
    s_high_unc = af_high_unc.surprise(test_vec, actual)
    assert s_high_unc < s_low_unc, (s_low_unc, s_high_unc)
    print(f"[ok] surprise 不确定性单调 (低unc={s_low_unc:.3f} > 高unc={s_high_unc:.3f})")


if __name__ == "__main__":
    _check_dataset_roundtrip()
    _check_alignment_function()
    _check_surprise_uncertainty_monotone()
    print("ALL TESTS PASSED")
