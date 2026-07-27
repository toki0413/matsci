"""科学计算类工具子包 — 符号回归 / 自动微分 / 数值 / 单位 / 对称 / TDA / UQ / GP / 描述符 / 证据融合 / 主动学习 / ML势 / 高通量.

模块:
  symbolic_regression_tool, autodiff_tool, numerical_tool, unit_tool,
  symmetry_tool, tda_tool, uq_tool, gp_tool, descriptor_tool,
  evidence_fusion_tool, active_learning_tool, ml_potential_tool,
  high_throughput_tool

注: symbolic_math_tool 的物理实现在 huginn.tools.symbolic_math 子包, 不在此处;
通过 huginn.tools.symbolic_math_tool (shim) 仍可访问.
"""

import os


def get_torch_device() -> str:
    """返回 torch 应使用的 device 字符串.

    rcb_runner / rcb_huginn 入口检测 GPU 后设置 HUGINN_TORCH_DEVICE:
      - "cuda": GPU 可用且 cudnn 验证通过
      - "cpu": GPU 不可用或 cudnn 损坏

    ponytail: 单一源 (env var), 不在每个工具里重复检测. 升级路径:
    按 model_size + batch_size 自动决定 (大 model 强制 GPU, 小 model CPU 够用).
    """
    return os.environ.get("HUGINN_TORCH_DEVICE", "cpu")

