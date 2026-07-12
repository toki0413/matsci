"""通用迁移学习域注册表 — 基于数学相似度而非硬编码关键词。

核心数学直觉:
  迁移学习有效性 ≈ f(domain_similarity, data_sufficiency)
  domain_similarity = cos(φ_source, φ_target)  其中 φ 是域特征向量
  data_sufficiency = min(1.0, n_samples / n_threshold)

  当 similarity > θ_sim 且 sufficiency > θ_data 时触发迁移。
  θ_sim=0.3, θ_data=0.1 (至少 3 个样本/30 阈值)

域特征向量由以下维度构成:
  - composition_space: 化学组成空间 (元素集合的 Jaccard 相似度)
  - structure_type: 结构类型 (perovskite/layered/amorphous/bulk)
  - property_type: 目标性质类型 (electronic/mechanical/thermal/magnetic)
  - method: 计算方法 (DFT-PBE/DFT-HSE/MD/ML)
  - scale: 特征尺度 (atomic/nano/meso/macro)

ponytail: 不引入新依赖。用标准库 math + dict 操作实现余弦相似度。
特征向量是离散的 one-hot 拼接，余弦退化为 Jaccard/Tanimoto。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainProfile:
    """一个材料科学域的特征画像。"""

    name: str
    composition: frozenset[str]  # 涉及元素: {"Si", "O", "Ti"}
    structure_type: str = ""    # perovskite, layered, amorphous, bulk, molecular
    property_type: str = ""     # electronic, mechanical, thermal, magnetic, optical
    method: str = ""             # DFT-PBE, DFT-HSE, MD, ML, experiment
    scale: str = "atomic"        # atomic, nano, meso, macro
    # 预训练数据量 (0-1 归一化, 1.0 = 1000+ 样本)
    pretrain_coverage: float = 0.0
    # 该域可用的预训练模型名 (GP/NN surrogate 名称)
    pretrained_models: tuple[str, ...] = ()


# ── 已注册的域 ──
_REGISTRY: list[DomainProfile] = [
    DomainProfile(
        name="lunar_regolith",
        composition=frozenset({"Si", "O", "Al", "Ca", "Fe", "Mg", "Ti"}),
        structure_type="amorphous",
        property_type="mechanical",
        method="experiment",
        scale="meso",
        pretrain_coverage=0.6,
        pretrained_models=("gp_lunar_v1",),
    ),
    DomainProfile(
        name="perovskite_solar",
        composition=frozenset({"Pb", "I", "Br", "Cs", "MA", "FA"}),
        structure_type="perovskite",
        property_type="electronic",
        method="DFT-PBE",
        scale="atomic",
        pretrain_coverage=0.8,
        pretrained_models=("transolver_perovskite_v2",),
    ),
    DomainProfile(
        name="oxide_catalyst",
        composition=frozenset({"Ti", "O", "Ce", "O", "Fe"}),
        structure_type="bulk",
        property_type="electronic",
        method="DFT-PBE",
        scale="atomic",
        pretrain_coverage=0.5,
        pretrained_models=("gp_oxide_v1",),
    ),
    DomainProfile(
        name="metal_alloy",
        composition=frozenset({"Fe", "Cr", "Ni", "Al", "Cu", "Ti"}),
        structure_type="bulk",
        property_type="mechanical",
        method="DFT-PBE",
        scale="atomic",
        pretrain_coverage=0.7,
        pretrained_models=("mtp_alloy_v3",),
    ),
    DomainProfile(
        name="2d_material",
        composition=frozenset({"Mo", "S", "W", "Se", "Graphene", "BN"}),
        structure_type="layered",
        property_type="electronic",
        method="DFT-HSE",
        scale="atomic",
        pretrain_coverage=0.65,
        pretrained_models=("transolver_2d_v1",),
    ),
]


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard 相似度 — 标准库实现，无新依赖。"""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _cosine_discrete(a: dict[str, float], b: dict[str, float]) -> float:
    """离散特征向量的余弦相似度。"""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def domain_to_vector(d: DomainProfile) -> dict[str, float]:
    """将 DomainProfile 转为离散特征向量。"""
    v: dict[str, float] = {}
    v[f"struct:{d.structure_type}"] = 1.0
    v[f"prop:{d.property_type}"] = 1.0
    v[f"method:{d.method}"] = 1.0
    v[f"scale:{d.scale}"] = 1.0
    return v


def similarity(source: DomainProfile, target: DomainProfile) -> float:
    """计算两个域之间的迁移相似度。

    composition_similarity (Jaccard) × 0.4
    + feature_similarity (cosine) × 0.4
    + pretrain_coverage × 0.2
    """
    comp_sim = _jaccard(source.composition, target.composition)
    feat_sim = _cosine_discrete(domain_to_vector(source), domain_to_vector(target))
    coverage = source.pretrain_coverage
    return comp_sim * 0.4 + feat_sim * 0.4 + coverage * 0.2


def find_transfer_domain(
    target_composition: frozenset[str] | set[str] | list[str],
    target_structure: str = "",
    target_property: str = "",
    target_method: str = "",
    n_samples: int = 0,
    threshold: float = 0.3,
) -> tuple[DomainProfile | None, float, str]:
    """查找最适合迁移的预训练域。

    返回: (best_domain, similarity_score, recommendation)
    recommendation 是给 LLM 的人类可读建议。

    n_samples: 用户当前域已有的数据量。
    迁移有效性 = similarity × min(1.0, n_samples / 30)
    （迁移学习需要至少 ~30 个样本做 fine-tune）
    """
    target = DomainProfile(
        name="target",
        composition=frozenset(target_composition),
        structure_type=target_structure,
        property_type=target_property,
        method=target_method,
    )

    best: DomainProfile | None = None
    best_sim = 0.0

    for src in _REGISTRY:
        sim = similarity(src, target)
        if sim > best_sim:
            best_sim = sim
            best = src

    if best is None or best_sim < threshold:
        return None, best_sim, (
            f"No pretrained domain exceeds similarity threshold ({threshold}). "
            f"Best was {best.name if best else 'none'} at {best_sim:.2f}. "
            f"Consider collecting more data or using a general surrogate."
        )

    data_sufficiency = min(1.0, n_samples / 30.0) if n_samples > 0 else 0.0
    transfer_effectiveness = best_sim * data_sufficiency

    if data_sufficiency < 0.1:
        return best, best_sim, (
            f"Domain '{best.name}' matched (similarity={best_sim:.2f}) but "
            f"insufficient target data ({n_samples}/30 samples). "
            f"Collect at least {30 - n_samples} more samples for effective transfer. "
            f"Available pretrained models: {', '.join(best.pretrained_models) or 'none'}."
        )

    return best, best_sim, (
        f"Transfer learning viable: '{best.name}' → target "
        f"(similarity={best_sim:.2f}, effectiveness={transfer_effectiveness:.2f}). "
        f"Pretrained models: {', '.join(best.pretrained_models) or 'none'}. "
        f"Method: load pretrained weights → fine-tune on {n_samples} target samples."
    )


def register_domain(profile: DomainProfile) -> None:
    """运行时注册新域（用户发现新材料体系时）。"""
    _REGISTRY.append(profile)
