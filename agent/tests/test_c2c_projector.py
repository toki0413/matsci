"""C2CProjector 参考实现的 torch 测试.

独立成文件, 且顶部 importorskip("torch"): 无 torch 环境下这 11 个测试整体 skip,
不影响无 torch 的 fusion 维度门禁. 有 torch 时才验证算法语义与论文一致.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="C2CProjector 参考实现需要 torch")

from research.c2c_projector import C2CProjector  # noqa: E402


@pytest.fixture()
def small_projector():
    return C2CProjector(
        source_dim=8, target_dim=8,
        source_num_heads=2, target_num_heads=2,
        intermediate_dim=32, hidden_dim=32,
        num_layers=3, dropout=0.0,
    )


def test_shapes_conserved(small_projector):
    torch.manual_seed(0)
    proj = small_projector.eval()
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, out_v = proj(source_kv, target_kv)
    assert out_k.shape == target_kv[0].shape
    assert out_v.shape == target_kv[1].shape


def test_zero_gate_preserves_target(small_projector):
    torch.manual_seed(0)
    proj = small_projector.eval()  # gate logit 初值 0 → 见 test_fusion_c2c.py 中语义
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, out_v = proj(source_kv, target_kv)
    torch.testing.assert_close(out_k, target_kv[0])
    torch.testing.assert_close(out_v, target_kv[1])


def test_negative_gate_preserves_target(small_projector):
    torch.manual_seed(0)
    proj = small_projector.eval()
    with torch.no_grad():
        proj.key_gate_logit.fill_(-1.0)
        proj.value_gate_logit.fill_(-1.0)
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, _ = proj(source_kv, target_kv)
    torch.testing.assert_close(out_k, target_kv[0])


def test_positive_gate_blends_projection(small_projector):
    torch.manual_seed(0)
    proj = small_projector.eval()
    with torch.no_grad():
        proj.key_gate_logit.fill_(2.0)
        proj.value_gate_logit.fill_(2.0)
        proj.key_proj_out.weight.fill_(0.1)
        proj.key_proj_out.bias.zero_()
        proj.value_proj_out.weight.fill_(0.1)
        proj.value_proj_out.bias.zero_()
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, out_v = proj(source_kv, target_kv)
    assert not torch.allclose(out_k, target_kv[0])
    assert not torch.allclose(out_v, target_kv[1])


def test_zero_init_preserves_target_even_gate_on():
    torch.manual_seed(0)
    proj = C2CProjector(
        source_dim=8, target_dim=8,
        source_num_heads=2, target_num_heads=2,
        intermediate_dim=32, hidden_dim=32,
        num_layers=3, dropout=0.0, zero_init=True,
    ).eval()
    with torch.no_grad():
        proj.key_gate_logit.fill_(2.0)
        proj.value_gate_logit.fill_(2.0)
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, out_v = proj(source_kv, target_kv)
    torch.testing.assert_close(out_k, target_kv[0])
    torch.testing.assert_close(out_v, target_kv[1])


def test_temperature_annealing(small_projector):
    proj = small_projector
    proj.update_temperature(0)
    t0 = float(proj.gate_temperature)
    assert t0 == pytest.approx(proj.initial_temperature)
    proj.update_temperature(proj.anneal_steps * 2)
    assert float(proj.gate_temperature) == pytest.approx(proj.final_temperature)


def test_make_fusion_channel_adapter():
    from research.c2c_projector import make_fusion_channel

    torch.manual_seed(0)
    proj = C2CProjector(
        source_dim=8, target_dim=8,
        source_num_heads=2, target_num_heads=2,
        hidden_dim=32, intermediate_dim=32, num_layers=3,
    ).eval()
    fuse = make_fusion_channel(proj)
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, out_v = fuse(source_kv, target_kv)
    assert out_k.shape == target_kv[0].shape
    assert out_v.shape == target_kv[1].shape


def test_backward_populates_projector_grads():
    """训练期只有 projector 参数有梯度 (两端 input 视为冻结)."""
    torch.manual_seed(0)
    proj = C2CProjector(
        source_dim=8, target_dim=8,
        source_num_heads=2, target_num_heads=2,
        intermediate_dim=32, hidden_dim=32,
        num_layers=3, dropout=0.0,
    ).train()
    B, Hs, Ht, N, D = 1, 2, 2, 4, 8
    source_kv = (torch.randn(B, Hs, N, D), torch.randn(B, Hs, N, D))
    target_kv = (torch.randn(B, Ht, N, D), torch.randn(B, Ht, N, D))
    out_k, _ = proj(source_kv, target_kv)
    loss = (out_k**2).mean()
    loss.backward()
    assert proj.key_in.weight.grad is not None
    assert proj.key_gate_logit.grad is not None
    assert proj.key_proj_out.weight.grad is not None
