"""物理结构形式化层 — v6 G46.

用户思想: 数学结构主义. 材料功能源于结构关系 (空间构型 / 相互作用模式 /
拓扑关系) 而非原子本体身份. AI 设计应解耦 "对象-结构" 捆绑: 锁定关键功能
位置的结构关系不变, 允许完全不同的实现者 (原子种类 / 合金成分 / 电场参数 /
晶格类型) 填充这些位置. 同构即等价.

5 类预定义结构关系 (覆盖大部分材料功能场景):
- catalytic_geometry:     催化几何 (活性位点空间构型)
- interface_binding:      界面结合模式 (异质界面键合)
- percolation_topology:   逾渗网络拓扑 (传导网络连通性)
- band_symmetry:          能带对称性 (电子结构对称性)
- defect_chemistry:       缺陷化学 (点缺陷形成与电荷补偿)

同构验证 (validate_structure_preservation): 给定 mapping (实现者替换),
验证结构关系是否保持. 用 sympy 符号化 + 物理约束库.
不调 vasp_tool (太重), 升级路径接 vasp_tool 做 DFT 验证.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhysicalStructure:
    """抽象物理结构 — 结构主义的形式化载体.

    relation_type: 5 类预定义之一 (catalytic_geometry / interface_binding /
        percolation_topology / band_symmetry / defect_chemistry) 或自定义
    relation_expr: sympy 表达式字符串, 形式化描述结构关系
        e.g. "Eq(d_M_M, 2.88*angstrom) & Eq(theta, 109.47*degree)"
    implementor_slots: 实现者槽位 dict, key 是槽位名, value 是当前实现者
        e.g. {"active_site": "Pt", "support": "TiO2"}
        结构主义核心: 槽位是结构位置, 实现者可替换
    constraints: 物理约束列表 (量纲 / 守恒 / 对称性), sympy 表达式
    provenance_id: 来源 provenance 记录 id (可追溯)
    relative_anchors: v6 G58 — 槽位 → anchor 实现者名列表. 空表示不做 relative
        验证 (回到 G46 sympy 验证). 论文 Moschella ICLR 2023: anchor 用于
        构造相对表示 (cosine similarity 向量), 对角度保持变换不变.
        e.g. {"active_site": ["Pt", "Pd", "Ni", "Cu"]}
    """
    relation_type: str
    relation_expr: str
    implementor_slots: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    provenance_id: str | None = None
    relative_anchors: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class StructureMapping:
    """结构映射 — 一次实现者替换 + 同构保持验证结果.

    source: 原始 PhysicalStructure
    target: 替换后的 PhysicalStructure (同 relation_type, 不同 implementor_slots)
    slot_replacements: 槽位替换映射 e.g. {"active_site": "Pt -> Pd"}
    is_isomorphic: validate_structure_preservation 的结果
    violation_detail: 若不保持, 违反的约束列表
    """
    source: PhysicalStructure
    target: PhysicalStructure
    slot_replacements: dict[str, str]
    is_isomorphic: bool = False
    violation_detail: list[str] = field(default_factory=list)


# ── 5 类预定义结构关系常量 ──────────────────────────────────────────
# 每类给一个典型 relation_expr + 常见槽位 + 物理约束, 当模板用.
# ponytail: 这些是骨架, 实际用时 LLM 或物理 precheck 会填充具体数值.

CATALYTIC_GEOMETRY = PhysicalStructure(
    relation_type="catalytic_geometry",
    # d-band center + 吸附几何 — Hammer-Nørskov d-band model 的结构关系
    # ponytail: relation_expr 用 sympy And() 语法, & 在 sympify 里优先级错
    relation_expr=(
        "And(Eq(epsilon_d, Symbol('epsilon_d')), "
        "Eq(theta_ads, Symbol('theta_ads')), "
        "Eq(d_M_X, Symbol('d_M_X')))"
    ),
    implementor_slots={
        "active_site": "M",  # 金属活性位点
        "adsorbate": "X",    # 吸附物
    },
    constraints=[
        # 吸附距离物理上下界
        "d_M_X > 1.0*angstrom",
        "d_M_X < 3.5*angstrom",
        # d-band center 物理上下界
        "epsilon_d > -5*eV",
        "epsilon_d < 2*eV",
    ],
    # G58: anchor 实现者 — 过渡金属 d-band model 典型参考
    relative_anchors={
        "active_site": ["Pt", "Pd", "Ni", "Cu"],
        "adsorbate": ["O", "H", "CO", "OH"],
    },
)

INTERFACE_BINDING = PhysicalStructure(
    relation_type="interface_binding",
    # 界面结合能 + 功函数差 — 异质界面键合的结构关系
    relation_expr=(
        "And(Eq(W_adh, Symbol('W_adh')), "
        "Eq(Delta_Phi, Symbol('Delta_Phi')), "
        "Eq(rho_interface, Symbol('rho_interface')))"
    ),
    implementor_slots={
        "matrix": "A",
        "precipitate": "B",
    },
    constraints=[
        # 界面粘附功物理上下界
        "W_adh > 0*joule/meter**2",
        # 界面电荷密度物理上下界 (sympy 用 Abs)
        "Abs(rho_interface) < 1e3*coulomb/meter**2",
    ],
    # G58: anchor 实现者 — 典型金属/陶瓷界面体系
    relative_anchors={
        "matrix": ["Al", "Cu", "Fe", "Ti"],
        "precipitate": ["Al2O3", "SiC", "TiB2", "MgO"],
    },
)

PERCOLATION_TOPOLOGY = PhysicalStructure(
    relation_type="percolation_topology",
    # 逾渗阈值 + 配位数 + 连通性 — 导电/导热网络的拓扑关系
    relation_expr=(
        "And(Eq(phi_c, Symbol('phi_c')), "
        "Eq(z, Symbol('z')), "
        "Eq(P_inf, Symbol('P_inf')))"
    ),
    implementor_slots={
        "conductor": "C",
        "matrix": "M",
    },
    constraints=[
        # 逾渗阈值物理上下界
        "phi_c > 0",
        "phi_c < 1",
        # 配位数物理上下界
        "z >= 2",
        "z <= 12",
        # 连通概率物理上下界
        "P_inf >= 0",
        "P_inf <= 1",
    ],
    # G58: anchor 实现者 — 典型导电填料/绝缘基体组合
    relative_anchors={
        "conductor": ["CNT", "Graphene", "Ag", "CB"],
        "matrix": ["PVDF", "Epoxy", "PDMS", "PE"],
    },
)

BAND_SYMMETRY = PhysicalStructure(
    relation_type="band_symmetry",
    # 能带对称性 + 时间反演 + 空间群 — 拓扑电子结构的对称关系
    relation_expr=(
        "And(Eq(E_gap, Symbol('E_gap')), "
        "Eq(SG, Symbol('SG')), "
        "Eq(TR_sym, Symbol('TR_sym')))"
    ),
    implementor_slots={
        "lattice": "L",
        "basis": "B",
    },
    constraints=[
        # 带隙物理上下界
        "E_gap >= 0*eV",
        # 空间群编号合法范围
        "SG >= 1",
        "SG <= 230",
        # 时间反演对称性二元 (sympy: Or(Eq(TR_sym,0), Eq(TR_sym,1)))
        "Or(Eq(TR_sym, 0), Eq(TR_sym, 1))",
    ],
    # G58: anchor 实现者 — 典型晶系/基元组合
    relative_anchors={
        "lattice": ["cubic", "tetragonal", "hexagonal", "orthorhombic"],
        "basis": ["Si", "GaAs", "MoS2", "Bi2Se3"],
    },
)

DEFECT_CHEMISTRY = PhysicalStructure(
    relation_type="defect_chemistry",
    # 缺陷形成能 + 电荷态 + Fermi 能级 — 点缺陷的结构关系
    relation_expr=(
        "And(Eq(E_f, Symbol('E_f')), "
        "Eq(q, Symbol('q')), "
        "Eq(E_Fermi, Symbol('E_Fermi')))"
    ),
    implementor_slots={
        "host": "H",
        "defect": "D",
    },
    constraints=[
        # 形成能物理上下界 (负值 = 自发形成)
        "E_f > -10*eV",
        "E_f < 20*eV",
        # 电荷态整数 (sympy: Or(Eq(q,k) for k in -4..4))
        "Or(Eq(q, -4), Eq(q, -3), Eq(q, -2), Eq(q, -1), Eq(q, 0), Eq(q, 1), Eq(q, 2), Eq(q, 3), Eq(q, 4))",
        # Fermi 能级在带隙内
        "E_Fermi >= E_vbm",
        "E_Fermi <= E_cbm",
    ],
    # G58: anchor 实现者 — 典型宿主/缺陷组合
    relative_anchors={
        "host": ["Si", "GaAs", "ZnO", "TiO2"],
        "defect": ["V_O", "V_M", "H_i", "M_sub"],
    },
)

PREDEFINED_STRUCTURES: dict[str, PhysicalStructure] = {
    "catalytic_geometry": CATALYTIC_GEOMETRY,
    "interface_binding": INTERFACE_BINDING,
    "percolation_topology": PERCOLATION_TOPOLOGY,
    "band_symmetry": BAND_SYMMETRY,
    "defect_chemistry": DEFECT_CHEMISTRY,
}


# ── G46 同构验证 — sympy + 物理约束库 ──────────────────────────────

def validate_structure_preservation(mapping: StructureMapping) -> bool:
    """验证结构关系在实现者替换后是否保持 (同构即等价).

    三层检查:
    1. relation_type 一致 (结构类型不变)
    2. relation_expr 可解析 (sympy 形式合法)
    3. constraints 全部满足 (物理约束不违反)

    返回 True = 同构保持, False = 结构破坏 (进 violation_detail).

    ponytail: 不调 vasp_tool (太重), 用 sympy 符号化 + 物理约束库.
    升级路径: 接 vasp_tool 做 DFT 级验证, 接 dimensional validator 做量纲验证.
    当前只验证约束可解析 + 不违反, 不做数值代入 (实现者值可能未定).
    """
    mapping.violation_detail = []

    # 1. relation_type 必须一致
    if mapping.source.relation_type != mapping.target.relation_type:
        mapping.violation_detail.append(
            f"relation_type mismatch: {mapping.source.relation_type} "
            f"-> {mapping.target.relation_type}"
        )
        mapping.is_isomorphic = False
        return False

    # 2. relation_expr 可解析 (sympy 形式合法)
    try:
        from sympy import sympify
        sympify(mapping.target.relation_expr)
    except Exception as e:
        mapping.violation_detail.append(
            f"target relation_expr sympify failed: {e}"
        )
        mapping.is_isomorphic = False
        return False

    # 3. constraints 全部可解析 (不违反 — 当前实现者值未定时只检查可解析)
    #    ponytail: 真正的数值验证需要实现者值代入, 留给 v7 DFT 升级.
    #    v6 只验证约束表达式本身合法, 防止 LLM 生成垃圾约束.
    for c in mapping.target.constraints:
        try:
            sympify(c)
        except Exception as e:
            mapping.violation_detail.append(
                f"constraint sympify failed: {c} ({e})"
            )
            mapping.is_isomorphic = False
            return False

    # 4. 槽位一致性 — 源和目标的槽位名集合必须相同 (结构位置不变)
    src_slots = set(mapping.source.implementor_slots.keys())
    tgt_slots = set(mapping.target.implementor_slots.keys())
    if src_slots != tgt_slots:
        missing = src_slots - tgt_slots
        extra = tgt_slots - src_slots
        if missing:
            mapping.violation_detail.append(f"missing slots in target: {missing}")
        if extra:
            mapping.violation_detail.append(f"extra slots in target: {extra}")
        mapping.is_isomorphic = False
        return False

    # 5. 至少一个槽位的实现者不同 (否则不算替换, 是平凡映射)
    #    ponytail: 平凡映射技术上同构, 但研究上无意义 — 标记为 violation
    #    让上游知道. 升级路径是返回 (is_isomorphic, is_trivial) 元组.
    diff_count = sum(
        1 for k in src_slots
        if mapping.source.implementor_slots[k] != mapping.target.implementor_slots[k]
    )
    if diff_count == 0:
        mapping.violation_detail.append("trivial mapping: no slot replacement")
        mapping.is_isomorphic = False
        return False

    mapping.is_isomorphic = True
    return True


# ── G58 相对结构同构 — Moschella et al. ICLR 2023 ─────────────────
# 论文: "Relative representations enable zero-shot latent space communication"
# 核心: cosine similarity 对角度保持变换 (旋转/反射/缩放) 不变. 两个潜在空间
# 差一个等距变换就是"同构"的, 相对表示把同构显式化.
# 与用户哲学的对应: 角度保持 = 同构即等价; 相对关系先于绝对坐标 = 结构先于对象;
# encoder 可替换 decoder 不变 = 锁定结构关系允许不同实现者填充.
# G46 sympy 验证是符号层, G58 加几何层.

def compute_relative_representation(
    embeddings: "dict[str, Any]",
    anchors: list[str],
) -> "Any":
    """计算相对表示 — 每个实现者 vs 每个 anchor 的 cosine similarity.

    论文 eq.(3): r_x = (cos(e_x, e_a1), ..., cos(e_x, e_an))

    embeddings: {实现者名: 向量} — 向量是 1D array-like
    anchors: anchor 实现者名列表 (必须是 embeddings 的 key 子集)

    返回 N×K cosine similarity 矩阵, N=len(embeddings), K=len(anchors).

    ponytail: numpy 实现, 不依赖 LLM/embedding service.
    embeddings 由调用方传入 (LLM / 物理 calculator / vasp_tool 升级路径).
    """
    import numpy as np

    impl_names = list(embeddings.keys())
    # anchor 必须在 embeddings 里 (anchor 自己也要有向量)
    missing = [a for a in anchors if a not in embeddings]
    if missing:
        raise ValueError(f"anchors not in embeddings: {missing}")

    E = np.array([embeddings[n] for n in impl_names], dtype=float)  # N×d
    A = np.array([embeddings[a] for a in anchors], dtype=float)     # K×d
    # cosine similarity: 单位化后内积
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return E_norm @ A_norm.T  # N×K


def validate_relative_isomorphism(
    source_repr: "Any",
    target_repr: "Any",
    tol: float = 1e-3,
) -> "tuple[bool, float]":
    """验证两个相对表示是否同构 (论文 eq.(4)).

    两个潜在空间差一个角度保持变换 T, 则它们的相对表示 (cosine similarity 矩阵)
    相同. distance = ||source - target||_F / ||source||_F.

    返回 (is_isomorphic, distance).
    is_isomorphic = distance < tol.

    ponytail: 用 Frobenius 范数归一化, 升级路径是 cosine distance +
    Stiefel manifold 拟合 (论文用 Procrustes 对齐).
    """
    import numpy as np

    source_arr = np.asarray(source_repr, dtype=float)
    target_arr = np.asarray(target_repr, dtype=float)
    if source_arr.shape != target_arr.shape:
        return (False, float("inf"))
    diff = float(np.linalg.norm(source_arr - target_arr, "fro"))
    norm = float(np.linalg.norm(source_arr, "fro")) + 1e-12
    distance = diff / norm
    return (distance < tol, distance)


def compute_and_validate_relative(
    source_embeddings: "dict[str, Any]",
    target_embeddings: "dict[str, Any]",
    anchors: list[str],
    tol: float = 1e-3,
) -> "tuple[bool, float]":
    """一站式: 计算 source/target 相对表示并验证同构.

    source_embeddings 和 target_embeddings 必须有相同的 key 集合
    (同一组实现者, 不同空间下的向量). anchors 是这组 key 的子集.
    """
    source_repr = compute_relative_representation(source_embeddings, anchors)
    target_repr = compute_relative_representation(target_embeddings, anchors)
    return validate_relative_isomorphism(source_repr, target_repr, tol=tol)


# ── self-check ────────────────────────────────────────────────────

def _self_check() -> int:
    """assert-based demo: 验证 PhysicalStructure + validate_structure_preservation + G58."""
    import tempfile

    # 1. 5 类预定义结构都存在
    assert len(PREDEFINED_STRUCTURES) == 5
    for name, s in PREDEFINED_STRUCTURES.items():
        assert s.relation_type == name
        assert s.relation_expr
        assert s.implementor_slots
        assert s.constraints
        # G58: 5 类预定义结构各有 relative_anchors
        assert s.relative_anchors, f"{name} missing relative_anchors"

    # 2. 同构保持: 替换 CATALYTIC_GEOMETRY 的 active_site (Pt -> Pd)
    src = CATALYTIC_GEOMETRY
    tgt = PhysicalStructure(
        relation_type="catalytic_geometry",
        relation_expr=src.relation_expr,
        implementor_slots={"active_site": "Pd", "adsorbate": "O"},  # 替换实现者
        constraints=src.constraints,
    )
    mapping = StructureMapping(
        source=src, target=tgt,
        slot_replacements={"active_site": "Pt -> Pd", "adsorbate": "X -> O"},
        is_isomorphic=False,
    )
    assert validate_structure_preservation(mapping) is True
    assert mapping.is_isomorphic is True
    assert not mapping.violation_detail

    # 3. 结构破坏: relation_type 不一致
    bad_type = PhysicalStructure(
        relation_type="interface_binding",  # 不一致
        relation_expr=src.relation_expr,
        implementor_slots=src.implementor_slots,
        constraints=src.constraints,
    )
    m2 = StructureMapping(source=src, target=bad_type, slot_replacements={})
    assert validate_structure_preservation(m2) is False
    assert "relation_type mismatch" in m2.violation_detail[0]

    # 4. 结构破坏: 槽位不一致 (缺失槽位)
    bad_slots = PhysicalStructure(
        relation_type="catalytic_geometry",
        relation_expr=src.relation_expr,
        implementor_slots={"active_site": "Pd"},  # 缺 adsorbate
        constraints=src.constraints,
    )
    m3 = StructureMapping(source=src, target=bad_slots, slot_replacements={})
    assert validate_structure_preservation(m3) is False
    assert any("missing slots" in v for v in m3.violation_detail)

    # 5. 结构破坏: 平凡映射 (无替换)
    trivial = PhysicalStructure(
        relation_type="catalytic_geometry",
        relation_expr=src.relation_expr,
        implementor_slots=src.implementor_slots.copy(),  # 完全相同
        constraints=src.constraints,
    )
    m4 = StructureMapping(source=src, target=trivial, slot_replacements={})
    assert validate_structure_preservation(m4) is False
    assert any("trivial" in v for v in m4.violation_detail)

    # 6. 结构破坏: 约束不可解析
    bad_constraint = PhysicalStructure(
        relation_type="catalytic_geometry",
        relation_expr=src.relation_expr,
        implementor_slots={"active_site": "Pd", "adsorbate": "O"},
        constraints=["this is not valid sympy @#$"],
    )
    m5 = StructureMapping(source=src, target=bad_constraint, slot_replacements={})
    assert validate_structure_preservation(m5) is False

    # ── G58: 相对结构同构测试 ──────────────────────────────────────
    # 用 mock embedding 验证: 旋转保持角度 → 同构; 扰动破坏角度 → 非同构
    import numpy as np

    # 4 个实现者 + 4 个 anchor (anchor 是实现者子集)
    rng = np.random.default_rng(42)
    impl_names = ["Pt", "Pd", "Ni", "Cu"]
    source_embeddings = {n: rng.standard_normal(8) for n in impl_names}
    anchors = impl_names  # 全部当 anchor (最简情形)

    # 7. 相对表示形状 N×K = 4×4
    source_repr = compute_relative_representation(source_embeddings, anchors)
    assert source_repr.shape == (4, 4), f"expected 4x4, got {source_repr.shape}"
    # 对角线应是 1 (自己跟自己 cosine = 1)
    assert np.allclose(np.diag(source_repr), 1.0, atol=1e-6)

    # 8. 同构: source 旋转后作为 target — 角度保持, 相对表示不变
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))  # 正交矩阵 = 旋转/反射
    target_embeddings = {n: Q @ source_embeddings[n] for n in impl_names}
    is_iso, dist = compute_and_validate_relative(
        source_embeddings, target_embeddings, anchors, tol=1e-6,
    )
    assert is_iso, f"rotation should preserve relative repr (dist={dist})"
    assert dist < 1e-6, f"rotation distance too large: {dist}"

    # 9. 非同构: target 加随机扰动 — 角度被破坏
    perturbed_embeddings = {
        n: source_embeddings[n] + 0.5 * rng.standard_normal(8)
        for n in impl_names
    }
    is_iso2, dist2 = compute_and_validate_relative(
        source_embeddings, perturbed_embeddings, anchors, tol=1e-3,
    )
    assert not is_iso2, f"perturbation should break isomorphism (dist={dist2})"
    assert dist2 > 1e-3, f"perturbation distance too small: {dist2}"

    # 10. anchor 不在 embeddings 里 → ValueError
    try:
        compute_relative_representation(
            source_embeddings, ["Pt", "nonexistent"],
        )
        assert False, "should raise ValueError for missing anchor"
    except ValueError:
        pass

    # 11. 形状不一致 → (False, inf)
    is_iso3, dist3 = validate_relative_isomorphism(
        np.zeros((4, 4)), np.zeros((3, 4)),
    )
    assert not is_iso3 and dist3 == float("inf")

    # 12. G58 跟 G46 叠加: relative_anchors 非空时可做 relative 验证
    assert CATALYTIC_GEOMETRY.relative_anchors["active_site"] == ["Pt", "Pd", "Ni", "Cu"]
    assert len(CATALYTIC_GEOMETRY.relative_anchors["adsorbate"]) == 4

    print("[PHYSTRUCT] self-check OK (incl G58 relative isomorphism)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
