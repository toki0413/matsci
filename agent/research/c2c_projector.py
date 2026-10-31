"""C2CProjector 参考实现 —— 模型间 KV-Cache 语义融合 (存证, 非主链路交付).

来源: thu-nics/C2C (arxiv:2510.03215, "Cache-to-Cache: Direct Semantic
Communication Between Large Language Models", ICLR'26), Apache-2.0 许可.
本文件是从上游 ``rosetta/model/projector.py`` **精简移植**的纯 torch 子集:
  - 去掉全部 transformers 依赖 (不 import Cache/DynamicCache/registry);
  - 保留算法本体 `C2CProjector.forward` 与门控/加权语义, 忠实还原 shape 约定;
  - 用途: 作为本体 capability ``fusion.fused_kv`` 的**参考实现**存证 —— 当运行环境
    拥有本地模型 + projector 权重时, 可据此挂载真正的 FusionChannel.

契约 (与上游一致):
  - source_kv / target_kv 各为 (key, value): key shape (B, Hs, N, Ds) /
    (B, Ht, N, Dt); value 同理.
  - 输出 (output_key, output_value) shape 对齐 target: (B, Ht, N, Dt).
  - 语义: output = target + gate * sigmoid(scalar) * projected (残差融合).

Apache-2.0 来源声明与重分发条件见仓库 LICENSE; 本文件是作者重写, 非逐字拷贝.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

# ⚠️ 本模块只在确实需要融合参考时被 import; 环境无 torch 时应由调用方先做 features 探测.
#    Huginn 主链路不 import 本文件 (research 存证), 因此 torch 缺失不影响默认运行.


class StandardFFNLayer(nn.Module):
    """Pre-norm RMSNorm + 经典 MLP 子层: y = x + Dropout(W2(Act(W1(RMSNorm(x)))))."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        act = activation.lower()
        if act == "gelu":
            self.act: nn.Module = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.w1(self.norm(x)))
        h = self.w2(h)
        return x + self.drop(h)


class RegularMLP(nn.Module):
    """按 num_layers 堆叠 StandardFFNLayer. 不含输入/输出投影, 由调用方负责."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 1,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.blocks = nn.ModuleList(
            StandardFFNLayer(hidden_dim, intermediate_dim, dropout=dropout, dtype=dtype)
            for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class C2CProjector(nn.Module):
    """C2C 投影/融合器 (参考实现).

    forward 不再需要 transformers; 直接消费对齐形状的 KV 元组. 门控语义与论文一致:
      - gate: 标量 logit + Gumbel-sigmoid (训练) / logit>0 二值 (推断), 逐层选 Layer;
      - scalar: head 级动态加权 (sigmoid);
      - 融合: output = target + gate * scalar * projected (残差, 保留 target 自身).
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 3, "num_layers must be >= 3"

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads

        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) 拼接 source+target 特征 → 投影到 hidden_dim
        _init_c2c_input_proj(self, in_dim, out_dim, hidden_dim, dtype)
        # 2) 一层公共嵌入 MLP
        _init_c2c_embed_mlps(self, hidden_dim, intermediate_dim, dropout, dtype)
        # 3a) 加权标量路径 → head 维
        _init_c2c_scalar_path(self, hidden_dim, target_num_heads, dropout, dtype)
        # 3b) 投影特征路径 → target 头维
        _init_c2c_proj_path(
            self, hidden_dim, intermediate_dim, num_layers, out_dim, dropout, dtype
        )

        if zero_init:
            _init_c2c_zero_out(self)

        # 逐层门控: 标量 logit + 温度淬火
        _init_c2c_gate_params(self, dtype)

    def update_temperature(self, step: int) -> None:
        ratio = min(step / self.anneal_steps, 1.0)
        self.gate_temperature.fill_(
            self.initial_temperature
            * (self.final_temperature / self.initial_temperature) ** ratio
        )

    def forward(
        self,
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        B, _Hs, N, Ds = source_key.shape
        _, Ht, _, Dt = target_key.shape

        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, -1)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, -1)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, -1)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, -1)

        key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
        value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)

        key_hidden = self.key_mlp1(self.key_in(key_cat))
        value_hidden = self.value_mlp1(self.value_in(value_cat))

        # 投影特征路径
        projected_key = _c2c_project_out(
            self.key_proj_mlp2, self.key_proj_out, key_hidden, B, N, Ht, Dt
        )
        projected_value = _c2c_project_out(
            self.value_proj_mlp2, self.value_proj_out, value_hidden, B, N, Ht, Dt
        )

        # 标量加权路径
        key_scalar = _c2c_scalar_out(
            self.key_scalar_mlp2, self.key_scalar_head, key_hidden
        )
        value_scalar = _c2c_scalar_out(
            self.value_scalar_mlp2, self.value_scalar_head, value_hidden
        )

        # 门控 (Gumbel-sigmoid 训练 / 二值推断)
        key_gate, value_gate = _c2c_gates(self, B, Ht, N, source_key)

        output_key = target_key + key_gate * torch.sigmoid(key_scalar) * projected_key
        output_value = (
            target_value + value_gate * torch.sigmoid(value_scalar) * projected_value
        )
        return output_key, output_value


def _init_c2c_input_proj(
    m: nn.Module, in_dim: int, out_dim: int, hidden_dim: int, dtype: torch.dtype
) -> None:
    """1) 拼接 source+target 特征 → 投影到 hidden_dim."""
    m.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
    m.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)


def _init_c2c_embed_mlps(
    m: nn.Module,
    hidden_dim: int,
    intermediate_dim: int,
    dropout: float,
    dtype: torch.dtype,
) -> None:
    """2) 一层公共嵌入 MLP."""
    m.key_mlp1 = RegularMLP(
        hidden_dim, intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype
    )
    m.value_mlp1 = RegularMLP(
        hidden_dim, intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype
    )


def _init_c2c_scalar_path(
    m: nn.Module,
    hidden_dim: int,
    target_num_heads: int,
    dropout: float,
    dtype: torch.dtype,
) -> None:
    """3a) 加权标量路径 → head 维."""
    m.key_scalar_mlp2 = RegularMLP(
        hidden_dim, hidden_dim, num_layers=1, dropout=dropout, dtype=dtype
    )
    m.value_scalar_mlp2 = RegularMLP(
        hidden_dim, hidden_dim, num_layers=1, dropout=dropout, dtype=dtype
    )
    m.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
    m.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)


def _init_c2c_proj_path(
    m: nn.Module,
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    out_dim: int,
    dropout: float,
    dtype: torch.dtype,
) -> None:
    """3b) 投影特征路径 → target 头维."""
    m.key_proj_mlp2 = RegularMLP(
        hidden_dim,
        intermediate_dim,
        num_layers=num_layers - 2,
        dropout=dropout,
        dtype=dtype,
    )
    m.value_proj_mlp2 = RegularMLP(
        hidden_dim,
        intermediate_dim,
        num_layers=num_layers - 2,
        dropout=dropout,
        dtype=dtype,
    )
    m.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
    m.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)


def _init_c2c_zero_out(m: nn.Module) -> None:
    """zero_init: 把两条投影输出层的权重/偏置置零."""
    nn.init.zeros_(m.key_proj_out.weight)
    nn.init.zeros_(m.key_proj_out.bias)
    nn.init.zeros_(m.value_proj_out.weight)
    nn.init.zeros_(m.value_proj_out.bias)


def _init_c2c_gate_params(m: nn.Module, dtype: torch.dtype) -> None:
    """逐层门控: 标量 logit + 温度淬火."""
    m.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
    m.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
    m.use_gumbel = True
    m.initial_temperature = 1.0
    m.final_temperature = 0.001
    m.anneal_steps = 1929
    m.register_buffer(
        "gate_temperature", torch.tensor(m.initial_temperature, dtype=dtype)
    )
    m.scalar_temperature = 1.0


def _c2c_project_out(
    proj_mlp: nn.Module,
    proj_out: nn.Module,
    hidden: Tensor,
    batch: int,
    seq: int,
    heads: int,
    head_dim: int,
) -> Tensor:
    """投影特征路径: hidden → (B, Ht, N, Dt)."""
    return proj_out(proj_mlp(hidden)).view(batch, seq, heads, head_dim).transpose(1, 2)


def _c2c_scalar_out(
    scalar_mlp: nn.Module, scalar_head: nn.Module, hidden: Tensor
) -> Tensor:
    """标量加权路径: hidden → (B, 1, N, 1) 加权标量."""
    return scalar_head(scalar_mlp(hidden)).permute(0, 2, 1).unsqueeze(-1)


def _c2c_gates(
    m: C2CProjector, batch: int, heads: int, seq: int, ref: Tensor
) -> tuple[Tensor, Tensor]:
    """门控 (Gumbel-sigmoid 训练 / 二值推断)."""
    if m.training and m.use_gumbel:
        g1 = _gumbel_noise(
            batch,
            heads,
            seq,
            key_gate_logit=m.key_gate_logit,
            device=ref.device,
            dtype=ref.dtype,
        )
        g2 = _gumbel_noise(
            batch,
            heads,
            seq,
            key_gate_logit=m.value_gate_logit,
            device=ref.device,
            dtype=ref.dtype,
        )
        key_gate = torch.sigmoid((m.key_gate_logit + g1) / m.gate_temperature)
        value_gate = torch.sigmoid((m.value_gate_logit + g2) / m.gate_temperature)
        return key_gate, value_gate
    key_gate = (m.key_gate_logit > 0).to(torch.float32)
    value_gate = (m.value_gate_logit > 0).to(torch.float32)
    return key_gate, value_gate


def _gumbel_noise(
    batch: int,
    heads: int,
    seq: int,
    key_gate_logit: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Gumbel 噪声重参数 (与上游一致)."""
    u = torch.rand(batch, heads, seq, 1, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + 1e-20) + 1e-20)


def make_fusion_channel(projector: C2CProjector) -> Callable[..., Any]:
    """把一个 C2CProjector 适配成本体 ``fusion.fused_kv`` 的 FusionChannel.fuse.

    Args:
        projector: 已加载权重的 C2CProjector 实例.

    Returns:
        一个 ``fuse(source_kv, target_kv, **kwargs) -> (key, value)`` 可调用对象,
        可直接包装成 FusionChannel 挂到 capability 上. 注意: 真实使用需两端模型都
        有 KV-Cache 访问权 (本地模型精调), API 推理链路拿不到内部表示.
    """

    def fuse(
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor],
        **_kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        return projector(source_kv, target_kv)

    return fuse


__all__ = ["C2CProjector", "StandardFFNLayer", "RegularMLP", "make_fusion_channel"]
