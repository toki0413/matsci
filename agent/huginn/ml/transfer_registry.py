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
    # 数学结构签名 — (canonical_name, sympy_expr) 元组.
    # canonical_name 做初筛 (必要非充分), sympy_expr 做 SymPy 等价检查.
    # 能识别 "a*phi_m**2 + b*phi_m**4" ≡ "c*phi_e**2 + d*phi_e**4"
    # (变量名不同但表达式结构同构).
    # ponytail: SymPy simplify 是半形式化等价检查, 不是形式证明.
    # 升级路径: sympy.simplify → Lean unifier (需 Lean 环境, 暂不做).
    structure_signature: tuple[tuple[str, str], ...] = ()
    # 例: (("landau_phi4", "a*phi**2 + b*phi**4"),
    #      ("group_O3", "g_so3(phi)"))


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
        # 铁电钙钛矿的 Landau 自由能: 标准形式 a*P**2 + b*P**4
        # P = 极化序参量
        structure_signature=(("landau_phi4", "a*P**2 + b*P**4"),),
    ),
    DomainProfile(
        name="ferromagnet",
        composition=frozenset({"Fe", "Ni", "Co"}),
        structure_type="bulk",
        property_type="magnetic",
        method="DFT-PBE",
        scale="atomic",
        pretrain_coverage=0.75,
        pretrained_models=("mtp_ferro_v1",),
        # 铁磁体的 Landau 自由能: 标准形式 a*m**2 + b*m**4
        # m = 磁化序参量. 与铁电共享同一表达式结构 (变量重命名后等价)
        structure_signature=(("landau_phi4", "alpha*m**2 + beta*m**4"),),
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

    结构同构覆盖: 若两域 structure_signature 在 SymPy 等价检查下有交集,
    直接返回 max(base, 0.9).
    原理: 共享数学结构的两域即使表面字段完全不同也通常可迁移
    (类比: 铁磁 ↔ 铁电共享 Landau phi4 泛函, 扩散方程 ↔ 热传导共享抛物 PDE 半群).
    """
    comp_sim = _jaccard(source.composition, target.composition)
    feat_sim = _cosine_discrete(domain_to_vector(source), domain_to_vector(target))
    coverage = source.pretrain_coverage
    base = comp_sim * 0.4 + feat_sim * 0.4 + coverage * 0.2

    # 结构同构硬覆盖: shared_structure 用 SymPy 等价检查 (含变量重命名)
    # ponytail: 0.9 是经验值, 避免完全 1.0 误把不真同构的域绑死.
    # 升级: signature 替换为 Lean 项后, 改用 unifier 输出置信度.
    if shared_structure(source, target):
        return max(base, 0.9)
    return base


def shared_structure(source: DomainProfile, target: DomainProfile) -> list[str]:
    """两域共享的数学结构标签, 用于 LLM 提示增强.
    返回 list 而非 set 保持稳定顺序.

    升级: canonical_name 匹配后, 用 SymPy simplify 检查表达式等价
    (含变量重命名规范化). 不只是字符串比较.
    """
    if not source.structure_signature or not target.structure_signature:
        return []
    # target 的 name → expr 字典, 用于查表
    target_by_name = dict(target.structure_signature)
    shared: list[str] = []
    for name, expr_src in source.structure_signature:
        if name in target_by_name:
            expr_tgt = target_by_name[name]
            if _sympy_equivalent(expr_src, expr_tgt):
                shared.append(name)
    return shared


def _sympy_equivalent(expr_a: str, expr_b: str) -> bool:
    """两 SymPy 表达式是否在变量重命名下等价.

    用 sympy.unify.usympy.unify 做结构 unification:
    - A 的所有 Symbol 设为 variables, 让 A 作为 pattern 匹配 B
    - 反向再 unify 一次 (B 作为 pattern 匹配 A)
    - 两个方向都有非空 binding → 变量重命名下等价

    能识别 "a*P**2 + b*P**4" ≡ "alpha*m**2 + beta*m**4".
    Add/Mul 的交换性由 unify 自动处理, 不需要全排列.

    ponytail: 双向 unify 是充分非必要 (能匹配但语义可能不同).
    常数 (1, 2 等) 不在 variables 列表里, 自动按字面值比较 —
    "a*x**2 + 1" ≠ "b*y**2 + 2".
    升级路径: sympy.unify → Lean unifier (形式证明).
    """
    try:
        import re
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
        from sympy.unify import usympy

        def _parse(expr_str: str):
            # 强制所有标识符为普通 Symbol, 避免 SymPy 把 alpha/beta 等希腊字母
            # 名解析成预定义 FunctionClass (algebraic_field 等)
            names = set(re.findall(r"[a-zA-Z_]\w*", expr_str))
            local_dict = {n: sp.Symbol(n) for n in names}
            return parse_expr(expr_str, local_dict=local_dict)

        ea = _parse(expr_a)
        eb = _parse(expr_b)

        if len(ea.free_symbols) != len(eb.free_symbols):
            return False

        # 双向 unify: A→B 和 B→A 都要有非空 binding
        vars_a = tuple(ea.free_symbols)
        vars_b = tuple(eb.free_symbols)
        ab = next(usympy.unify(ea, eb, {}, variables=vars_a), None)
        ba = next(usympy.unify(eb, ea, {}, variables=vars_b), None)
        return ab is not None and ba is not None
    except Exception:
        return expr_a == expr_b


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

    # 共享结构提示 (若两域 structure_signature 有交集)
    shared = shared_structure(best, target)
    shared_hint = (
        f" Shared mathematical structure: {', '.join(shared)}. "
        f"Cross-domain transfer by structural isomorphism, "
        f"surface features may differ."
        if shared else ""
    )

    if data_sufficiency < 0.1:
        return best, best_sim, (
            f"Domain '{best.name}' matched (similarity={best_sim:.2f}) but "
            f"insufficient target data ({n_samples}/30 samples). "
            f"Collect at least {30 - n_samples} more samples for effective transfer. "
            f"Available pretrained models: {', '.join(best.pretrained_models) or 'none'}."
            f"{shared_hint}"
        )

    return best, best_sim, (
        f"Transfer learning viable: '{best.name}' → target "
        f"(similarity={best_sim:.2f}, effectiveness={transfer_effectiveness:.2f}). "
        f"Pretrained models: {', '.join(best.pretrained_models) or 'none'}. "
        f"Method: load pretrained weights → fine-tune on {n_samples} target samples."
        f"{shared_hint}"
    )


# IPI 防御: signature canonical_name allowlist + 表达式长度限制
# 防止用户上传的 CIF meta 字段污染 structure_signature, 进而通过
# _sympy_equivalent → shared_structure → _render_transfer_prompt 注入 LLM
# ponytail: 静默丢弃不在 allowlist 的 signature, 不阻断主流程
# 升级: 从用户材料自动抽取 signature 时, 改为从已知数据库查表而非信任输入
_ALLOWED_SIGNATURE_NAMES: frozenset[str] = frozenset({
    "landau_phi4",
    "group_O3",
    "parabolic_pde",
    "semigroup",
    "conserved_current",
    "order_discrete",
    "kane_model",
})
_MAX_EXPR_LEN = 200  # sympy_expr 最大字符数, 防长 payload 注入


def _sanitize_signature(sig: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """过滤掉不在 allowlist 的 signature, 截断过长的表达式."""
    return tuple(
        (name, expr[:_MAX_EXPR_LEN])
        for name, expr in sig
        if name in _ALLOWED_SIGNATURE_NAMES and len(expr) <= _MAX_EXPR_LEN
    )


def register_domain(profile: DomainProfile) -> None:
    """运行时注册新域（用户发现新材料体系时）.

    IPI 防御: structure_signature 会被 sanitize, 不在 allowlist 的
    canonical_name 静默丢弃. 这防止用户上传的 CIF meta 字段污染
    LLM prompt (通过 shared_structure → _render_transfer_prompt).
    """
    if profile.structure_signature:
        clean = _sanitize_signature(profile.structure_signature)
        from dataclasses import replace
        profile = replace(profile, structure_signature=clean)
    _REGISTRY.append(profile)
