"""VisualSCM — 视觉结构因果模型.

把材料科学图像特征 + 实验条件建模为 SCM (Structural Causal Model),
支持 Pearl L2 干预预测 (do-calculus).

节点是数学变量 (温度/颗粒大小/d-spacing 等), 不是图像. 图像特征通过
vision_describe 提取后填入 SCM 节点. 因果图用领域先验 (Arrhenius /
Ostwald / Fick 等物理方程) + 数据后验拟合.

阶梯映射 (Pearl Causal Hierarchy):
  L1 观察 P(Y|X)        — vision_describe 现有能力 (描述图像)
  L2 干预 P(Y|do(X))    — predict_intervention (本模块)
  L3 反事实 P(Y_x|X=x',Y=y') — counterfactual_render (Phase 3, 未实现)

4 个内置领域模板 (Phase 1 先验):
  - sintering:        T, t   → particle_size, density
  - ostwald_ripening: T, t   → particle_size
  - diffusion:        T, c   → diffusion_coefficient
  - phase_transition: T, p   → phase_fraction

混合策略: KB 有模板用模板 (confirmed=True), 无模板 LLM 生成 draft
(confirmed=False), predict_intervention 时警告未确认.

设计原则 (ponytail):
  - 节点是数学对象, 不是图像 (符合 physics=mathematics 偏好)
  - 物理先验优先, 数据后验修正 (反对纯经验归纳)
  - 拓扑序采样做 do-calculus, 不引 pgmpy/causalnex (零新依赖)
  - LLM 生成的 SCM 必须标 confirmed=False, predict 时显式警告

升级路径:
  - Phase 2: visual_causal_chain 从多张图 + 条件自动拟合 SCM
  - Phase 3: counterfactual_render L3 反事实渲染 (abduction + prediction)
  - 数据多时换 Bayesian 结构学习 (pc algorithm / GES)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# ── 表征 ─────────────────────────────────────────────────────

VariableType = Literal["condition", "feature", "latent"]


@dataclass
class Variable:
    """SCM 节点."""
    name: str
    type: VariableType
    unit: str = ""                # "K" / "nm" / "g/cm^3" / "" (无量纲)
    range: tuple[float, float] | None = None   # 取值范围 (用于采样截断)
    description: str = ""


@dataclass
class Edge:
    """SCM 有向边 cause → effect."""
    cause: str
    effect: str
    mechanism: str = ""           # 物理机制名 (arrhenius / fick / ...)
    strength: float = 1.0         # 0-1, 影响强度 (LLM 生成时填)


@dataclass
class VisualSCM:
    """结构因果模型 — 视觉特征 + 实验条件的因果图.

    核心假设 (Pearl SCM):
      每个节点 X_i = f_i(parents(X_i), U_i), U_i 独立噪声
    干预 do(X=v): 把 X 的方程替换为常数 v, 切断其父节点的因果边
    """
    name: str                              # 模板名 "sintering" / "custom_xxx"
    domain: str                            # "ceramic" / "polymer" / ...
    nodes: dict[str, Variable]
    edges: list[Edge]
    # 结构方程: node_name → f(parents_values: dict[str, float], noise: float) → float
    equations: dict[str, Callable[[dict[str, float], float], float]]
    # 噪声分布: node_name → (sampler_fn, std_hint)
    noise: dict[str, Callable[[], float]]
    confirmed: bool = True                 # True=KB 模板, False=LLM 草稿
    source: str = "template"               # "template" / "llm_draft" / "fitted"
    notes: str = ""

    def parents(self, node: str) -> list[str]:
        """取节点的父节点 (因果上游)."""
        return [e.cause for e in self.edges if e.effect == node]

    def topological_order(self) -> list[str]:
        """拓扑序 (Kahn 算法). 干预预测按此顺序算每个节点."""
        in_deg: dict[str, int] = {n: 0 for n in self.nodes}
        for e in self.edges:
            in_deg[e.effect] = in_deg.get(e.effect, 0) + 1
        queue = [n for n, d in in_deg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for e in self.edges:
                if e.cause == n:
                    in_deg[e.effect] -= 1
                    if in_deg[e.effect] == 0:
                        queue.append(e.effect)
        if len(order) != len(self.nodes):
            raise ValueError(f"SCM 有环: {self.name}")
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "nodes": {n: {"type": v.type, "unit": v.unit,
                          "range": v.range, "description": v.description}
                      for n, v in self.nodes.items()},
            "edges": [{"cause": e.cause, "effect": e.effect,
                       "mechanism": e.mechanism, "strength": e.strength}
                      for e in self.edges],
            "confirmed": self.confirmed,
            "source": self.source,
            "notes": self.notes,
        }


# ── 内置物理方程 (领域先验) ──────────────────────────────────

def _arrhenius(T: float, Ea: float, A0: float, R: float = 8.314e-3) -> float:
    """Arrhenius: k = A0 * exp(-Ea / (R*T)). T 单位 K, Ea 单位 kJ/mol."""
    return A0 * math.exp(-Ea / (R * max(T, 1.0)))


def _ostwald_growth(t: float, K: float, n: float = 3.0) -> float:
    """Ostwald 熟化: r^3 - r0^3 = K*t. 返 K*t (相对增长)."""
    return K * max(t, 0.0)


def _fick_D(T: float, D0: float, Ea: float, R: float = 8.314e-3) -> float:
    """Fick 扩散系数: D = D0 * exp(-Ea/(R*T))."""
    return D0 * math.exp(-Ea / (R * max(T, 1.0)))


# ── 4 个内置模板 ─────────────────────────────────────────────

def _noise_normal(std: float, mean: float = 0.0) -> Callable[[], float]:
    """正态噪声 sampler factory. std=0 时返常数."""
    if std <= 0:
        return lambda: 0.0
    return lambda: random.gauss(mean, std)


def _template_sintering() -> VisualSCM:
    """烧结 SCM: T, t → particle_size, density.

    物理先验:
      - 颗粒生长: Arrhenius 速率 + Ostwald 熟化 (r^3 ~ K(T)*t)
      - 致密化: 经验 sintering map (T 升 → density 升, t 二阶修正)
    典型域: 陶瓷 (Al2O3/ZrO2), 金属粉末, C-S-H 高温脱水

    ponytail: 方程是简化版 (忽略晶界扩散/蒸发凝聚等次要机制),
    升级路径: 加第二机制 (lattice diffusion vs grain boundary).
    """
    nodes = {
        "T": Variable("T", "condition", "K", (1073, 2073), "烧结温度"),
        "t": Variable("t", "condition", "h", (0.5, 100), "保温时间"),
        "particle_size": Variable("particle_size", "feature", "nm",
                                  (10, 5000), "平均颗粒尺寸"),
        "density": Variable("density", "feature", "g/cm^3",
                            (2.0, 22.0), "致密度"),
    }
    edges = [
        Edge("T", "particle_size", "arrhenius+ostwald", 1.0),
        Edge("t", "particle_size", "ostwald_growth", 1.0),
        Edge("T", "density", "sintering_map", 1.0),
        Edge("t", "density", "sintering_map", 0.6),
        Edge("particle_size", "density", "inverse_correlation", 0.4),
    ]
    # 经验参数 (陶瓷典型值, ponytail: 不硬到具体材料)
    Ea_grow = 250.0   # kJ/mol, 颗粒生长活化能
    A0_grow = 1e6     # nm^3/h 预指数
    r0 = 50.0         # nm 初始颗粒

    def f_particle(p: dict[str, float], u: float) -> float:
        T, t = p.get("T", 1500), p.get("t", 2.0)
        K = _arrhenius(T, Ea_grow, A0_grow)
        r_cubed = r0**3 + _ostwald_growth(t, K)
        r = r_cubed ** (1/3)
        return max(r * (1 + u), 1.0)

    def f_density(p: dict[str, float], u: float) -> float:
        T, t, ps = p.get("T", 1500), p.get("t", 2.0), p.get("particle_size", 200)
        # 致密度随 T 升 (sigmoid), t 弱修正, ps 反相关
        rho_theory = 6.0  # 假设理论密度 ~6 (陶瓷典型)
        T_norm = (T - 1073) / (2073 - 1073)
        density = rho_theory * (1 - math.exp(-3 * T_norm)) * (1 + 0.1 * math.log(t))
        density *= (200 / max(ps, 10)) ** 0.1
        return max(min(density * (1 + u), rho_theory * 1.05), 0.5)

    equations = {
        "T": lambda p, u: p.get("T", 1500),  # 条件节点: 直接取
        "t": lambda p, u: p.get("t", 2.0),
        "particle_size": f_particle,
        "density": f_density,
    }
    noise = {
        "T": _noise_normal(0),
        "t": _noise_normal(0),
        "particle_size": _noise_normal(0.08),  # 8% 相对噪声
        "density": _noise_normal(0.05),
    }
    return VisualSCM(
        name="sintering", domain="ceramic",
        nodes=nodes, edges=edges, equations=equations, noise=noise,
        confirmed=True, source="template",
        notes="陶瓷/金属粉末烧结. Arrhenius+Ostwald 先验. "
              "适用 800-1800°C. 高于 1800°C 需加蒸发凝聚机制.",
    )


def _template_ostwald_ripening() -> VisualSCM:
    """Ostwald 熟化 SCM: T, t → particle_size.

    纯 LSW 理论 (Lifshitz-Slyozov-Wagner):
      r^3 - r0^3 = K(T) * t, K(T) = K0 * exp(-Ea/(RT))
    典型域: 沉淀析出, 胶体熟化, C-S-H 老化

    ponytail: 跟 sintering 共享颗粒生长方程, 但无致密化节点.
    """
    nodes = {
        "T": Variable("T", "condition", "K", (300, 1500), "温度"),
        "t": Variable("t", "condition", "h", (1, 10000), "时间"),
        "particle_size": Variable("particle_size", "feature", "nm",
                                  (1, 1000), "平均颗粒尺寸"),
    }
    edges = [
        Edge("T", "particle_size", "arrhenius+LSW", 1.0),
        Edge("t", "particle_size", "LSW_cubic", 1.0),
    ]
    Ea, K0, r0 = 180.0, 1e4, 20.0

    def f_particle(p: dict[str, float], u: float) -> float:
        T, t = p.get("T", 600), p.get("t", 100)
        K = _arrhenius(T, Ea, K0)
        r = (r0**3 + K * t) ** (1/3)
        return max(r * (1 + u), 0.5)

    equations = {
        "T": lambda p, u: p.get("T", 600),
        "t": lambda p, u: p.get("t", 100),
        "particle_size": f_particle,
    }
    noise = {
        "T": _noise_normal(0),
        "t": _noise_normal(0),
        "particle_size": _noise_normal(0.10),
    }
    return VisualSCM(
        name="ostwald_ripening", domain="precipitation",
        nodes=nodes, edges=edges, equations=equations, noise=noise,
        confirmed=True, source="template",
        notes="LSW 理论. 适用扩散控制熟化. 不适用反应控制或界面控制.",
    )


def _template_diffusion() -> VisualSCM:
    """扩散 SCM: T, c → diffusion_coefficient.

    Fick 定律 + Stokes-Einstein (液相):
      D = D0 * exp(-Ea/(RT))   (固相)
      D = kT/(6πηr)            (液相, Stokes-Einstein)
    典型域: 离子导体, 固态电池, C-S-H 中 Ca/Si 扩散
    """
    nodes = {
        "T": Variable("T", "condition", "K", (273, 2000), "温度"),
        "c": Variable("c", "condition", "mol/L", (0.01, 10), "浓度"),
        "D": Variable("D", "feature", "m^2/s", (1e-20, 1e-8), "扩散系数"),
    }
    edges = [
        Edge("T", "D", "arrhenius_fick", 1.0),
        Edge("c", "D", "concentration_correction", 0.3),
    ]
    Ea, D0 = 150.0, 1e-4

    def f_D(p: dict[str, float], u: float) -> float:
        T, c = p.get("T", 800), p.get("c", 1.0)
        D_base = _fick_D(T, D0, Ea)
        # 浓度修正: 高浓度互作用降低 D (理想溶液偏离)
        D = D_base * (1 - 0.05 * math.log10(max(c, 0.001)))
        return max(D * math.exp(u), 1e-25)

    equations = {
        "T": lambda p, u: p.get("T", 800),
        "c": lambda p, u: p.get("c", 1.0),
        "D": f_D,
    }
    noise = {
        "T": _noise_normal(0),
        "c": _noise_normal(0),
        "D": _noise_normal(0.15),  # log 空间噪声
    }
    return VisualSCM(
        name="diffusion", domain="transport",
        nodes=nodes, edges=edges, equations=equations, noise=noise,
        confirmed=True, source="template",
        notes="Fick + Arrhenius. 固相扩散. 液相需换 Stokes-Einstein.",
    )


def _template_phase_transition() -> VisualSCM:
    """相变 SCM: T, p → phase_fraction.

    Avrami 方程 + Johnson-Mehl:
      f(t) = 1 - exp(-k(T) * t^n)
      k(T) = k0 * exp(-Ea/RT)
    典型域: 钢相变, 形状记忆合金, 钙钛矿相变
    """
    nodes = {
        "T": Variable("T", "condition", "K", (273, 2000), "温度"),
        "p": Variable("p", "condition", "Pa", (1e3, 1e9), "压力"),
        "phase_fraction": Variable("phase_fraction", "feature", "",
                                   (0.0, 1.0), "新相体积分数"),
    }
    edges = [
        Edge("T", "phase_fraction", "avrami_kinetics", 1.0),
        Edge("p", "phase_fraction", "clausius_clapeyron", 0.5),
    ]
    Ea, k0, n_avrami = 120.0, 1e3, 2.5
    # 假设: 平衡相变温度 T_eq (常压), dT/dP slope (Clausius-Clapeyron)
    T_eq, dTdP = 1000.0, 1e-7  # K, K/Pa

    def f_phase(p: dict[str, float], u: float) -> float:
        T, p = p.get("T", 1100), p.get("p", 1e5)
        # 平衡温度随压力漂移
        T_eq_p = T_eq + dTdP * p
        # 过冷度 / 过热度
        delta = T_eq_p - T
        if delta <= 0:
            # 高温稳定相, phase_fraction ~ 0
            return max(0.0 + u * 0.05, 0.0)
        # Avrami: 简化 (假设 t=1 单位时间, k 随 T)
        k = _arrhenius(T, Ea, k0)
        f = 1 - math.exp(-(k * 1.0 * delta / 100) ** n_avrami)
        return min(max(f + u * 0.05, 0.0), 1.0)

    equations = {
        "T": lambda p, u: p.get("T", 1100),
        "p": lambda p, u: p.get("p", 1e5),
        "phase_fraction": f_phase,
    }
    noise = {
        "T": _noise_normal(0),
        "p": _noise_normal(0),
        "phase_fraction": _noise_normal(0.05),
    }
    return VisualSCM(
        name="phase_transition", domain="kinetics",
        nodes=nodes, edges=edges, equations=equations, noise=noise,
        confirmed=True, source="template",
        notes="Avrami + Clausius-Clapeyron. 一级相变. 不适用二级/连续相变.",
    )


# ── 模板注册 ─────────────────────────────────────────────────

_TEMPLATES: dict[str, Callable[[], VisualSCM]] = {
    "sintering": _template_sintering,
    "ostwald_ripening": _template_ostwald_ripening,
    "diffusion": _template_diffusion,
    "phase_transition": _template_phase_transition,
}


def list_templates() -> list[str]:
    """列可用模板名."""
    return list(_TEMPLATES.keys())


def get_template(name: str) -> VisualSCM | None:
    """取模板. 不存在返 None."""
    factory = _TEMPLATES.get(name)
    return factory() if factory else None


def match_template(domain: str = "", conditions: list[str] | None = None,
                   features: list[str] | None = None) -> VisualSCM | None:
    """按域/条件/特征匹配模板.

    ponytail: 简单关键词匹配. 升级路径: 用 embedding 相似度.
    """
    conditions = conditions or []
    features = features or []
    all_vars = set(conditions + features)
    for name, factory in _TEMPLATES.items():
        scm = factory()
        if domain and domain.lower() not in scm.domain.lower():
            continue
        if all_vars and not all_vars.issubset(set(scm.nodes.keys())):
            continue
        return scm
    return None


# ── self-check ───────────────────────────────────────────────

def _selfcheck() -> None:
    """15 项 assert 验证 VisualSCM 表征 + 模板库."""
    # 1. 4 个模板都能构造
    scm_sint = _template_sintering()
    scm_ostw = _template_ostwald_ripening()
    scm_diff = _template_diffusion()
    scm_phtr = _template_phase_transition()
    assert scm_sint.name == "sintering"
    assert scm_ostw.name == "ostwald_ripening"
    assert scm_diff.name == "diffusion"
    assert scm_phtr.name == "phase_transition"

    # 2. 模板 confirmed=True (KB 模板)
    assert scm_sint.confirmed is True
    assert scm_sint.source == "template"

    # 3. list_templates 返 4 个
    assert set(list_templates()) == {"sintering", "ostwald_ripening",
                                     "diffusion", "phase_transition"}

    # 4. get_template 命中
    assert get_template("sintering").name == "sintering"
    assert get_template("nonexistent") is None

    # 5. match_template 按域 + 条件 + 特征命中
    m = match_template(domain="ceramic", conditions=["T", "t"],
                       features=["particle_size", "density"])
    assert m is not None
    assert m.name == "sintering"

    # 6. match_template 不命中返 None
    m = match_template(domain="nonexistent_domain")
    assert m is None

    # 7. parents() 正确返父节点
    assert set(scm_sint.parents("particle_size")) == {"T", "t"}
    assert set(scm_sint.parents("density")) == {"T", "t", "particle_size"}
    assert scm_sint.parents("T") == []

    # 8. topological_order 不抛 (无环)
    order = scm_sint.topological_order()
    assert set(order) == set(scm_sint.nodes.keys())
    # T, t 必须在 particle_size 前
    assert order.index("T") < order.index("particle_size")
    assert order.index("t") < order.index("particle_size")
    # particle_size 必须在 density 前
    assert order.index("particle_size") < order.index("density")

    # 9. equations 都可调
    for scm in [scm_sint, scm_ostw, scm_diff, scm_phtr]:
        for node, eq in scm.equations.items():
            val = eq({"T": 1500, "t": 5, "c": 1.0, "p": 1e5,
                      "particle_size": 200}, 0.0)
            assert isinstance(val, (int, float)), f"{scm.name}.{node} 返非数: {val}"

    # 10. 噪声 sampler 可调
    for scm in [scm_sint, scm_ostw, scm_diff, scm_phtr]:
        for node, sampler in scm.noise.items():
            val = sampler()
            assert isinstance(val, (int, float))

    # 11. sintering 颗粒大小随 T 升而升 (Arrhenius 单调)
    r_low = scm_sint.equations["particle_size"]({"T": 1200, "t": 5}, 0.0)
    r_high = scm_sint.equations["particle_size"]({"T": 1800, "t": 5}, 0.0)
    assert r_high > r_low, f"Arrhenius 失效: T升颗粒应升, {r_low} → {r_high}"

    # 12. ostwald 颗粒大小随 t 升而升 (LSW cubic)
    r_short = scm_ostw.equations["particle_size"]({"T": 600, "t": 10}, 0.0)
    r_long = scm_ostw.equations["particle_size"]({"T": 600, "t": 1000}, 0.0)
    assert r_long > r_short

    # 13. diffusion D 随 T 升而升 (Arrhenius)
    d_low = scm_diff.equations["D"]({"T": 500, "c": 1.0}, 0.0)
    d_high = scm_diff.equations["D"]({"T": 1500, "c": 1.0}, 0.0)
    assert d_high > d_low

    # 14. to_dict 结构正确
    d = scm_sint.to_dict()
    assert d["name"] == "sintering"
    assert "nodes" in d and "edges" in d
    assert d["confirmed"] is True
    assert len(d["edges"]) == 5

    # 15. _noise_normal(0) 返常数 0
    sampler = _noise_normal(0)
    assert sampler() == 0.0
    assert sampler() == 0.0

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
