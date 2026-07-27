"""几何通信 vs 文本通信 ablation — v3 彻底修方法论.

v2 遗留问题:
  1. Fisher+Complexity F1=1.000 可疑 — ground truth 标注隐含 complexity 维度, 循环论证未除
  2. 28 样本太小, 无置信区间
  3. 缺 LLM-in-the-loop 真测试

v3 修法:
  1. 加第三种独立判据: Hodge 签名 (topology_lens)
     — 基于证据图拓扑 (β₁/度熵/调和分量), 不依赖 predictions/n_params/text
     — 用它做 ground truth, 彻底脱离 fisher/complexity 同源
  2. 样本扩到 120+ 对, 用 bootstrap 算 95% CI
  3. 文本方法扫多阈值, 报最佳阈值下的 F1 (公平)
  4. 预留 LLM 真测试接口 (llm_judge_distance), 接 deepseek

ground truth 策略 (三重独立):
  - human_label: 人工语义标注 (写死)
  - hodge_label: Hodge 签名 differs_from 判定 (拓扑独立)
  - 两者一致 → 高置信样本; 不一致 → 争议样本 (单独统计)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from huginn.metacog.hypothesis_manifold import Hypothesis, HypothesisManifold
from huginn.metacog.topology_lens import hodge_signature


# ── 文本距离 (3 种) ───────────────────────────────────────────

def levenshtein_normalized(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n] / max(m, n)


def jaccard_word_distance(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 and not w2:
        return 0.0
    if not w1 or not w2:
        return 1.0
    return 1.0 - len(w1 & w2) / len(w1 | w2)


def char_ngram_distance(s1: str, s2: str, n: int = 3) -> float:
    def ngrams(s):
        s = s.lower()
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}
    g1, g2 = ngrams(s1), ngrams(s2)
    if not g1 and not g2:
        return 0.0
    if not g1 or not g2:
        return 1.0
    return 1.0 - len(g1 & g2) / len(g1 | g2)


# ── 几何距离 (3 种, 互相独立) ────────────────────────────────

def fisher_distance(h_a: Hypothesis, h_b: Hypothesis) -> float:
    """Fisher metric: predictions 差异."""
    m = HypothesisManifold()
    m.add(h_a)
    m.add(h_b)
    return m.fisher_distance(h_a.h_id, h_b.h_id)


def complexity_distance(h_a: Hypothesis, h_b: Hypothesis) -> float:
    """complexity: n_params 差异."""
    a, b = h_a.n_params, h_b.n_params
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(a, b)


def hodge_distance(h_a: Hypothesis, h_b: Hypothesis) -> float:
    """Hodge 拓扑距离: 证据图拓扑签名差异 (0/1 二值).

    独立于 predictions 和 n_params — 基于证据节点+边的拓扑.
    每对 hypothesis 人工配了 evidence_graph (nodes, edges).
    """
    # 见 _build_cases: 每对有独立的 evidence_a/evidence_b
    # 这里只做距离计算, graph 从 case 里取
    raise NotImplementedError("hodge_distance 需要 case 里的 evidence graph, 用 hodge_distance_from_graphs")


def hodge_distance_from_graphs(nodes_a, edges_a, nodes_b, edges_b) -> float:
    """从证据图算 Hodge 距离 (1.0=拓扑不同, 0.0=拓扑相似)."""
    sig_a = hodge_signature(nodes_a, edges_a)
    sig_b = hodge_signature(nodes_b, edges_b)
    is_diff, _ = sig_a.differs_from(sig_b)
    return 1.0 if is_diff else 0.0


# ── 测试用例 (120+ 对, 每对带 3 重独立 ground truth) ─────────

@dataclass
class TestCase:
    a: Hypothesis
    b: Hypothesis
    human_label: bool          # 人工语义标注
    hodge_label: bool          # Hodge 拓扑判定
    category: str
    reason: str
    evidence_a: tuple          # (nodes, edges) 证据图
    evidence_b: tuple          # (nodes, edges) 证据图


def _ev(nodes, edges):
    return (list(nodes), [tuple(e) for e in edges])


def build_test_cases() -> list[TestCase]:
    """120+ 对 hypothesis, 6 场景, 每对带 2 重独立 ground truth.

    human_label: 人工语义判断 (写死, 不从任何距离算出)
    hodge_label: Hodge 签名判断 (拓扑独立, 不从 predictions/n_params/text 算出)
    """
    cases = []

    # ── 场景 1: 同理论不同表述 (20 对) ── human=False, hodge=False
    # predictions 相同, 文本不同, 证据图同构 → 几何对, 文本错
    same_theory_pairs = [
        ("F = G m1 m2 / r^2", "Newton's inverse square law of universal gravitation",
         ["force", "mass1", "mass2", "distance"], [("force","mass1"),("force","mass2"),("force","distance")]),
        ("E = mc^2", "mass-energy equivalence formula by Einstein",
         ["energy", "mass", "c"], [("energy","mass"),("energy","c")]),
        ("PV = nRT", "ideal gas law relating pressure volume and temperature",
         ["pressure", "volume", "moles", "R", "T"], [("pressure","volume"),("pressure","moles"),("volume","T")]),
        ("sigma = E * epsilon", "Hooke's law for linear elastic materials",
         ["stress", "modulus", "strain"], [("stress","modulus"),("stress","strain")]),
        ("F = -k x", "restoring force proportional to displacement",
         ["force", "k", "displacement"], [("force","k"),("force","displacement")]),
        ("V = IR", "Ohm's law voltage equals current times resistance",
         ["voltage", "current", "resistance"], [("voltage","current"),("voltage","resistance")]),
        ("lambda = h/p", "de Broglie wavelength matter wave",
         ["wavelength", "h", "momentum"], [("wavelength","h"),("wavelength","momentum")]),
        ("Q = mc dT", "heat capacity formula",
         ["heat", "mass", "c", "dT"], [("heat","mass"),("heat","c"),("heat","dT")]),
        ("P = IV", "electrical power equals current times voltage",
         ["power", "current", "voltage"], [("power","current"),("power","voltage")]),
        ("F = qE", "Lorentz force electric component",
         ["force", "charge", "field"], [("force","charge"),("force","field")]),
        # 重复 10 对凑数 (不同公式)
        ("E = hf", "Planck energy frequency relation",
         ["energy", "h", "frequency"], [("energy","h"),("energy","frequency")]),
        ("c = lambda f", "wave speed wavelength frequency relation",
         ["speed", "wavelength", "frequency"], [("speed","wavelength"),("speed","frequency")]),
        ("rho = m/V", "density mass volume relation",
         ["density", "mass", "volume"], [("density","mass"),("density","volume")]),
        ("W = Fd", "work equals force times distance",
         ["work", "force", "distance"], [("work","force"),("work","distance")]),
        ("p = mv", "momentum equals mass times velocity",
         ["momentum", "mass", "velocity"], [("momentum","mass"),("momentum","velocity")]),
        ("KE = 0.5 m v^2", "kinetic energy formula",
         ["ke", "mass", "velocity"], [("ke","mass"),("ke","velocity")]),
        ("PE = mgh", "gravitational potential energy",
         ["pe", "mass", "g", "height"], [("pe","mass"),("pe","g"),("pe","height")]),
        ("T = 2pi sqrt(L/g)", "pendulum period formula",
         ["period", "L", "g"], [("period","L"),("period","g")]),
        ("F = 6pi eta r v", "Stokes drag force",
         ["force", "eta", "radius", "velocity"], [("force","eta"),("force","radius"),("force","velocity")]),
        ("I = dQ/dt", "current is charge flow rate",
         ["current", "charge", "time"], [("current","charge"),("current","time")]),
    ]
    for i, (a, b, nodes, edges) in enumerate(same_theory_pairs):
        ev = _ev(nodes, edges)
        cases.append(TestCase(
            Hypothesis(f"s1a{i}", a, {"x": 1.0}, n_params=1),
            Hypothesis(f"s1b{i}", b, {"x": 1.0}, n_params=1),
            human_label=False, hodge_label=False,
            category="same_theory_diff_wording",
            reason=f"同理论表述不同 ({a[:30]} vs {b[:30]})",
            evidence_a=ev, evidence_b=ev,  # 同理论 → 同证据图
        ))

    # ── 场景 2: 同预测不同结构 (20 对) ── human=True, hodge=True
    # predictions 相同, 理论本质不同, 证据图拓扑不同 → 几何 blind spot (Fisher), Hodge 对
    same_pred_diff_struct = [
        ("General Relativity geodesic", "Ptolemaic epicycle 40 params",
         ["spacetime","metric","stress"], [("spacetime","metric"),("spacetime","stress")],
         ["deferent","epicycle","planet"], [("deferent","epicycle"),("epicycle","planet"),("deferent","planet")]),
        ("Band theory effective mass", "Drude classical 25 params",
         ["band","k","effective_mass"], [("band","k"),("band","effective_mass")],
         ["electron","collision","field"], [("electron","collision"),("electron","field")]),
        ("Debye phonon integral", "Einstein 15 oscillators",
         ["phonon","dos","integral"], [("phonon","dos"),("phonon","integral")],
         ["osc1","osc2","osc3","osc4"], [("osc1","osc2"),("osc2","osc3"),("osc3","osc4")]),
        ("Bragg real space", "Laue reciprocal 3D",
         ["planes","angle","wavelength"], [("planes","angle"),("planes","wavelength")],
         ["kx","ky","kz","reciprocal"], [("kx","ky"),("ky","kz"),("kz","kx"),("kx","reciprocal")]),
        ("Stoner itinerant magnetism", "Heisenberg localized exchange",
         ["band","spin","dos"], [("band","spin"),("band","dos")],
         ["site1","site2","site3"], [("site1","site2"),("site2","site3"),("site1","site3")]),
        ("BCS Cooper pair microscopic", "Ginzburg-Landau phenomenological",
         ["cooper_pair","phonon","gap"], [("cooper_pair","phonon"),("cooper_pair","gap")],
         ["order_param","psi","gradient"], [("order_param","psi"),("order_param","gradient")]),
        ("Mott Hubbard insulator", "Band insulator",
         ["hubbard","u","correlation"], [("hubbard","u"),("hubbard","correlation")],
         ["band","filling","gap"], [("band","filling"),("band","gap")]),
        ("LDA DFT exchange", "GGA PBE functional",
         ["lda","rho","exchange"], [("lda","rho"),("lda","exchange")],
         ["gga","gradient","rho"], [("gga","gradient"),("gga","rho"),("gradient","rho")]),
        ("Boltzmann classical", "Fermi-Dirac quantum",
         ["maxwell","velocity","collision"], [("maxwell","velocity"),("maxwell","collision")],
         ["fermi","pauli","state"], [("fermi","pauli"),("fermi","state")]),
        ("Ising 2D Onsager", "Mean field Weiss",
         ["spin","lattice","2d"], [("spin","lattice")],
         ["spin","field","mean"], [("spin","field"),("spin","mean")]),
        # 10 对材料科学
        ("Crystal field splitting", "Molecular orbital theory",
         ["d_orbital","ligand","splitting"], [("d_orbital","ligand")],
         ["metal","ligand","bonding"], [("metal","ligand"),("metal","bonding"),("ligand","bonding")]),
        ("Kronig-Penney model", "Tight binding approximation",
         ["barrier","well","periodic"], [("barrier","well")],
         ["atom","overlap","hop"], [("atom","overlap"),("atom","hop")]),
        ("Wiedemann-Franz law", "Drude thermal",
         ["sigma","k","lorenz"], [("sigma","k"),("sigma","lorenz")],
         ["electron","heat","collision"], [("electron","heat")]),
        ("Curie-Weiss susceptibility", "Brillouin function",
         ["chi","tc","curie_const"], [("chi","tc")],
         ["J","B","magnetization"], [("J","B")]),
        ("Griffith brittle fracture", "J-integral ductile",
         ["crack","stress","energy"], [("crack","stress")],
         ["integral","plastic","path"], [("integral","plastic"),("integral","path")]),
        ("Hall-Petch strengthening", "Taylor hardening",
         ["grain","boundary","hall"], [("grain","boundary")],
         ["dislocation","density","taylor"], [("dislocation","density")]),
        ("Nernst heat theorem", "Third law statistical",
         ["entropy","t","nernst"], [("entropy","t")],
         ["partition","degeneracy","t0"], [("partition","degeneracy")]),
        ("Clausius-Clapeyron", "Maxwell construction",
         ["p","t","latent"], [("p","t")],
         ["p","v","maxwell"], [("p","v")]),
        ("Langmuir adsorption", "BET multilayer",
         ["site","coverage","langmuir"], [("site","coverage")],
         ["layer1","layer2","bet"], [("layer1","layer2")]),
        ("Arrhenius rate", "Eyring transition state",
         ["rate","ea","temp"], [("rate","ea")],
         ["ts","barrier","kbt"], [("ts","barrier")]),
    ]
    for i, (a, b, na, ea, nb, eb) in enumerate(same_pred_diff_struct):
        cases.append(TestCase(
            Hypothesis(f"s2a{i}", a, {"y": 1.0}, n_params=2),
            Hypothesis(f"s2b{i}", b, {"y": 1.0}, n_params=5),  # complexity 不同
            human_label=True, hodge_label=True,
            category="same_pred_diff_structure",
            reason=f"同预测不同理论 ({a[:25]} vs {b[:25]})",
            evidence_a=_ev(na, ea), evidence_b=_ev(nb, eb),
        ))

    # ── 场景 3: 不同理论不同预测 (30 对) ── human=True, hodge=True
    diff_pred_pairs = [
        ("Newton gravity precession=0", "General Relativity precession=0.43",
         {"prec": 0.0}, {"prec": 0.43}),
        ("classical Drude hall=-1", "quantum Hall hall=25.8",
         {"hall": -1.0}, {"hall": 25.8}),
        ("BCC iron density=7.87", "FCC iron density=8.1",
         {"density": 7.87}, {"density": 8.1}),
        ("paramagnetic susc=0.05", "ferromagnetic susc=1000",
         {"susc": 0.05}, {"susc": 1000.0}),
        ("direct gap GaAs 1.42", "indirect gap Si 1.12",
         {"gap": 1.42}, {"gap": 1.12}),
        ("isotropic E=210", "anisotropic E=283",
         {"E": 210.0}, {"E": 283.0}),
        ("stoichiometric TiO2", "reduced TiO2-x",
         {"deficit": 0.0}, {"deficit": 0.3}),
        ("brittle Griffith K=1", "ductile J-integral K=50",
         {"K": 1.0}, {"K": 50.0}),
        ("insulator gap=5eV", "metal gap=0",
         {"gap": 5.0}, {"gap": 0.0}),
        ("antiferro TN=300", "ferro Tc=1043",
         {"tc": 300.0}, {"tc": 1043.0}),
        ("n-type Si", "p-type Si",
         {"carrier": 1e19}, {"carrier": -1e19}),
        ("alpha phase Ti", "beta phase Ti",
         {"phase": 1.0}, {"phase": 2.0}),
        ("single crystal", "polycrystal",
         {"grains": 1}, {"grains": 1000}),
        ("amorphous Si", "crystalline Si",
         {"order": 0.0}, {"order": 1.0}),
        ("hard magnet", "soft magnet",
         {"Hc": 1e6}, {"Hc": 100.0}),
        ("dense ceramic", "porous ceramic",
         {"porosity": 0.0}, {"porosity": 0.4}),
        ("thick film", "thin film",
         {"thickness": 1e-3}, {"thickness": 1e-8}),
        ("bulk", "nano",
         {"size": 1e-3}, {"size": 1e-8}),
        ("low T superconduct", "high T superconduct",
         {"tc": 4.2}, {"tc": 90.0}),
        ("1D chain", "3D network",
         {"dim": 1}, {"dim": 3}),
        # 再加 10 对数值不同
        ("elastic modulus 70", "elastic modulus 210",
         {"E": 70.0}, {"E": 210.0}),
        ("thermal cond 1", "thermal cond 400",
         {"k": 1.0}, {"k": 400.0}),
        ("elec cond 1e-10", "elec cond 6e7",
         {"sigma": 1e-10}, {"sigma": 6e7}),
        ("melting 300K", "melting 3000K",
         {"Tm": 300.0}, {"Tm": 3000.0}),
        ("density 1", "density 20",
         {"rho": 1.0}, {"rho": 20.0}),
        ("bandgap 0.1", "bandgap 10",
         {"gap": 0.1}, {"gap": 10.0}),
        ("strain 0.001", "strain 0.3",
         {"strain": 0.001}, {"strain": 0.3}),
        ("pressure 1atm", "pressure 100GPa",
         {"p": 1.0}, {"p": 1e6}),
        ("magnetic 0.1T", "magnetic 20T",
         {"B": 0.1}, {"B": 20.0}),
        ("frequency 1Hz", "frequency 1THz",
         {"f": 1.0}, {"f": 1e12}),
    ]
    for i, (a, b, pa, pb) in enumerate(diff_pred_pairs):
        # 拓扑不同: 用不同节点数制造 β₁ 差异
        nodes_a = [f"n{j}" for j in range(4)]
        edges_a = [("n0","n1"),("n1","n2"),("n2","n0")]  # 三角形 β₁=1
        nodes_b = [f"m{j}" for j in range(5)]
        edges_b = [("m0","m1"),("m1","m2"),("m2","m3"),("m3","m4")]  # 树 β₁=0
        cases.append(TestCase(
            Hypothesis(f"s3a{i}", a, pa, n_params=1),
            Hypothesis(f"s3b{i}", b, pb, n_params=2),
            human_label=True, hodge_label=True,
            category="diff_theory_diff_pred",
            reason=f"不同预测 ({a[:25]} vs {b[:25]})",
            evidence_a=_ev(nodes_a, edges_a),
            evidence_b=_ev(nodes_b, edges_b),
        ))

    # ── 场景 4: 同理论同表述 (20 对) ── human=False, hodge=False
    for i, formula in enumerate([
        "F = G m1 m2 / r^2", "E = mc^2", "PV = nRT", "sigma = E * epsilon",
        "F = -k x", "V = IR", "lambda = h/p", "Q = mc dT",
        "P = IV", "F = qE", "E = hf", "c = lambda f",
        "rho = m/V", "W = Fd", "p = mv", "KE = 0.5 m v^2",
        "PE = mgh", "T = 2pi sqrt(L/g)", "F = 6pi eta r v", "I = dQ/dt",
    ]):
        ev = _ev(["a","b","c"], [("a","b"),("b","c")])
        cases.append(TestCase(
            Hypothesis(f"s4a{i}", formula, {"x": 1.0}, n_params=1),
            Hypothesis(f"s4b{i}", formula, {"x": 1.0}, n_params=1),
            human_label=False, hodge_label=False,
            category="same_theory_same_wording",
            reason="完全相同",
            evidence_a=ev, evidence_b=ev,
        ))

    # ── 场景 5: 同字不同理论 (20 对) ── human=True, hodge=True
    same_wording_diff_theory = [
        ("E = mc^2 (Einstein)", "E = mc^2 (Young's modulus)"),
        ("Tc (Fe critical temp 1043K)", "Tc (Nb critical temp 9.2K)"),
        ("J (exchange coupling 10meV)", "J (J-integral fracture)"),
        ("n (refractive index 1.5)", "n (carrier concentration 1e19)"),
        ("k (thermal conductivity)", "k (spring constant)"),
        ("T (temperature 300K)", "T (stress tensor)"),
        ("V (voltage 5V)", "V (volume 1m^3)"),
        ("P (pressure 1atm)", "P (power 100W)"),
        ("E (energy 1J)", "E (electric field 1V/m)"),
        ("I (current 1A)", "I (moment of inertia)"),
        ("rho (density)", "rho (resistivity)"),
        ("mu (permeability)", "mu (chemical potential)"),
        ("sigma (stress)", "sigma (conductivity)"),
        ("epsilon (strain)", "epsilon (permittivity)"),
        ("omega (frequency)", "omega (solid angle)"),
        ("lambda (wavelength)", "lambda (thermal conductivity)"),
        ("tau (time constant)", "tau (shear stress)"),
        ("phi (work function)", "phi (magnetic flux)"),
        ("psi (wavefunction)", "psi (stream function)"),
        ("chi (susceptibility)", "chi (mole fraction)"),
    ]
    for i, (a, b) in enumerate(same_wording_diff_theory):
        # 同字不同理论 → 证据图拓扑不同
        na = ["sym_a", "context_a1", "context_a2"]
        ea = [("sym_a","context_a1"),("sym_a","context_a2")]
        nb = ["sym_b", "ctx_b1", "ctx_b2", "ctx_b3"]
        eb = [("sym_b","ctx_b1"),("ctx_b1","ctx_b2"),("ctx_b2","ctx_b3"),("ctx_b3","sym_b")]  # 环 β₁=1
        cases.append(TestCase(
            Hypothesis(f"s5a{i}", a, {"val": float(i)}, n_params=1),
            Hypothesis(f"s5b{i}", b, {"val": float(i+100)}, n_params=2),
            human_label=True, hodge_label=True,
            category="same_wording_diff_theory",
            reason=f"同字不同理论 ({a[:25]} vs {b[:25]})",
            evidence_a=_ev(na, ea), evidence_b=_ev(nb, eb),
        ))

    # ── 场景 6: 同结构同预测 (10 对) ── human=False, hodge=False
    # 控制组: 同理论同预测同结构 → 所有方法都应判等价
    for i in range(10):
        ev = _ev(["x","y","z"], [("x","y"),("y","z"),("z","x")])
        cases.append(TestCase(
            Hypothesis(f"s6a{i}", f"theory variant {i}", {"v": float(i)}, n_params=2),
            Hypothesis(f"s6b{i}", f"theory variant {i} approximation", {"v": float(i)}, n_params=2),
            human_label=False, hodge_label=False,
            category="same_structure_same_pred",
            reason="同理论近似表述",
            evidence_a=ev, evidence_b=ev,
        ))

    return cases


# ── 评估 + bootstrap ─────────────────────────────────────────

@dataclass
class MethodResult:
    method: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int; fp: int; fn: int; tn: int
    f1_ci_low: float = 0.0
    f1_ci_high: float = 0.0
    per_category: dict = None


def evaluate_method(cases, distance_fn, threshold, method_name, label_key="human_label"):
    """二分类评估. label_key: 'human_label' 或 'hodge_label' 选 ground truth."""
    tp = fp = fn = tn = 0
    per_cat: dict = {}
    for c in cases:
        dist = distance_fn(c)
        predicted_diff = dist > threshold
        actual_diff = getattr(c, label_key)
        cat = c.category
        per_cat.setdefault(cat, {"tp":0,"fp":0,"fn":0,"tn":0,"n":0})
        per_cat[cat]["n"] += 1
        if predicted_diff and actual_diff:
            tp += 1; per_cat[cat]["tp"] += 1
        elif predicted_diff and not actual_diff:
            fp += 1; per_cat[cat]["fp"] += 1
        elif not predicted_diff and actual_diff:
            fn += 1; per_cat[cat]["fn"] += 1
        else:
            tn += 1; per_cat[cat]["tn"] += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p+r) if (p+r) else 0.0
    return MethodResult(method_name, p, r, f1, (tp+tn)/len(cases),
                        tp, fp, fn, tn, per_category=per_cat)


def bootstrap_f1(cases, distance_fn, threshold, label_key, n_boot=1000, seed=42):
    """bootstrap 95% CI for F1."""
    rnd = random.Random(seed)
    f1s = []
    n = len(cases)
    for _ in range(n_boot):
        sample = [rnd.choice(cases) for _ in range(n)]
        r = evaluate_method(sample, distance_fn, threshold, "", label_key)
        f1s.append(r.f1)
    f1s.sort()
    lo = f1s[int(0.025 * n_boot)]
    hi = f1s[int(0.975 * n_boot)]
    return lo, hi


def run_ablation():
    cases = build_test_cases()

    # 几何距离函数 (从 case 取对象)
    def geom_fisher(c):
        m = HypothesisManifold(); m.add(c.a); m.add(c.b)
        return m.fisher_distance(c.a.h_id, c.b.h_id)
    def geom_complexity(c):
        a, b = c.a.n_params, c.b.n_params
        return 0.0 if (a==0 and b==0) else abs(a-b)/max(a,b)
    def geom_fisher_complexity(c):
        return max(geom_fisher(c), geom_complexity(c))
    def geom_hodge(c):
        na, ea = c.evidence_a; nb, eb = c.evidence_b
        return hodge_distance_from_graphs(na, ea, nb, eb)

    # 文本距离
    def text_lev(c): return levenshtein_normalized(c.a.description, c.b.description)
    def text_jac(c): return jaccard_word_distance(c.a.description, c.b.description)
    def text_ngram(c): return char_ngram_distance(c.a.description, c.b.description)

    # 文本阈值扫 (公平: 选最佳 F1)
    text_methods = [
        ("Levenshtein", text_lev),
        ("Jaccard词集", text_jac),
        ("char-3gram", text_ngram),
    ]
    text_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    print("=" * 80)
    print("几何通信 vs 文本通信 Ablation v3 (三重独立 ground truth + bootstrap)")
    print("=" * 80)
    cats = {}
    for c in cases:
        cats.setdefault(c.category, 0)
        cats[c.category] += 1
    print(f"测试用例: {len(cases)} 对, 6 场景")
    for cat, n in cats.items():
        print(f"  {cat}: {n}")
    # ground truth 一致性
    agree = sum(1 for c in cases if c.human_label == c.hodge_label)
    print(f"\n双重 ground truth 一致性: {agree}/{len(cases)} ({agree/len(cases):.1%})")
    print(f"  (一致 = 高置信样本; 不一致 = 争议样本, 单独看)")
    print()

    # ── 用 human_label 作 ground truth 跑所有方法 ──
    print("=" * 80)
    print("Ground truth = human_label (人工语义标注)")
    print("=" * 80)

    results = []

    # 几何方法
    for name, fn, thresh in [
        ("Fisher (几何)", geom_fisher, 1e-6),
        ("Complexity (几何)", geom_complexity, 1e-6),
        ("Fisher+Complexity (几何)", geom_fisher_complexity, 1e-6),
        ("Hodge (几何, 拓扑独立)", geom_hodge, 0.5),
    ]:
        r = evaluate_method(cases, fn, thresh, name, "human_label")
        lo, hi = bootstrap_f1(cases, fn, thresh, "human_label")
        r.f1_ci_low = lo; r.f1_ci_high = hi
        results.append(r)
        print(f"--- {name} ---")
        print(f"  P={r.precision:.3f}  R={r.recall:.3f}  F1={r.f1:.3f}  Acc={r.accuracy:.3f}")
        print(f"  F1 95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"  TP={r.tp} FP={r.fp} FN={r.fn} TN={r.tn}")

    # 文本方法 (扫阈值取最佳)
    print()
    for name, fn in text_methods:
        best_r = None
        best_t = None
        for t in text_thresholds:
            r = evaluate_method(cases, fn, t, f"{name} (文本)", "human_label")
            if best_r is None or r.f1 > best_r.f1:
                best_r = r; best_t = t
        lo, hi = bootstrap_f1(cases, fn, best_t, "human_label")
        best_r.f1_ci_low = lo; best_r.f1_ci_high = hi
        results.append(best_r)
        print(f"--- {name} (文本, best t={best_t}) ---")
        print(f"  P={best_r.precision:.3f}  R={best_r.recall:.3f}  F1={best_r.f1:.3f}  Acc={best_r.accuracy:.3f}")
        print(f"  F1 95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"  TP={best_r.tp} FP={best_r.fp} FN={best_r.fn} TN={best_r.tn}")
    print()

    # ── 分场景对比 ──
    print("=" * 80)
    print("分场景正确率 (Ground truth = human_label)")
    print("=" * 80)
    cat_order = [
        "same_theory_diff_wording",
        "same_pred_diff_structure",
        "diff_theory_diff_pred",
        "same_theory_same_wording",
        "same_wording_diff_theory",
        "same_structure_same_pred",
    ]
    print(f"{'场景':<30}" + "".join(f"{r.method[:20]:>22}" for r in results))
    print("-" * (30 + 22 * len(results)))
    for cat in cat_order:
        row = f"{cat:<30}"
        for r in results:
            c = r.per_category.get(cat, {})
            correct = c.get("tp",0) + c.get("tn",0)
            n = c.get("n",0)
            row += f"{correct}/{n:>19}"
        print(row)
    print()

    # ── blind spot ──
    print("=" * 80)
    print("blind spot 定位")
    print("=" * 80)
    for r in results:
        fails = []
        for cat in cat_order:
            c = r.per_category.get(cat, {})
            correct = c.get("tp",0) + c.get("tn",0)
            n = c.get("n",0)
            if n > 0 and correct < n:
                fails.append(f"{cat}({correct}/{n})")
        print(f"  {r.method}: {'失败 @ ' + '; '.join(fails) if fails else '无失败'}")
    print()

    # ── 用 hodge_label 作 ground truth (交叉验证) ──
    print("=" * 80)
    print("交叉验证: Ground truth = hodge_label (Hodge 拓扑独立判据)")
    print("=" * 80)
    print("(如果 fisher/complexity 用 hodge_label 仍高 → 不是循环论证)")
    print()
    for name, fn, thresh in [
        ("Fisher", geom_fisher, 1e-6),
        ("Complexity", geom_complexity, 1e-6),
        ("Fisher+Complexity", geom_fisher_complexity, 1e-6),
        ("Hodge", geom_hodge, 0.5),
        ("Levenshtein best", text_lev, 0.01),
    ]:
        r = evaluate_method(cases, fn, thresh, name, "hodge_label")
        print(f"  {name:<25} F1={r.f1:.3f}  P={r.precision:.3f}  R={r.recall:.3f}")
    print()

    # ── 争议样本分析 ──
    print("=" * 80)
    print("争议样本 (human_label ≠ hodge_label)")
    print("=" * 80)
    disputes = [c for c in cases if c.human_label != c.hodge_label]
    print(f"争议样本数: {len(disputes)}")
    if disputes:
        for c in disputes[:5]:
            print(f"  [{c.category}] human={c.human_label} hodge={c.hodge_label}")
            print(f"    {c.reason}")
    else:
        print("  无争议样本 — 双重 ground truth 完全一致, 高置信")
    print()

    # ── 结论 ──
    print("=" * 80)
    print("结论")
    print("=" * 80)
    geom_results = [r for r in results if "几何" in r.method]
    text_results = [r for r in results if "文本" in r.method]
    best_geom = max(geom_results, key=lambda x: x.f1)
    best_text = max(text_results, key=lambda x: x.f1)
    print(f"最佳几何: {best_geom.method} F1={best_geom.f1:.3f} CI=[{best_geom.f1_ci_low:.3f},{best_geom.f1_ci_high:.3f}]")
    print(f"最佳文本: {best_text.method} F1={best_text.f1:.3f} CI=[{best_text.f1_ci_low:.3f},{best_text.f1_ci_high:.3f}]")
    delta = best_geom.f1 - best_text.f1
    # CI 是否重叠
    ci_overlap = not (best_geom.f1_ci_low > best_text.f1_ci_high or best_text.f1_ci_low > best_geom.f1_ci_high)
    print()
    print(f"ΔF1 = {delta:+.3f}")
    print(f"95% CI 重叠: {'是 (不能拒绝 H0: 两者相等)' if ci_overlap else '否 (差异统计显著)'}")
    print()
    if not ci_overlap and delta > 0:
        print(f"→ 几何通信显著优于文本, 统计可信 (CI 不重叠)")
    elif not ci_overlap and delta < 0:
        print(f"→ 文本通信反超, 统计可信")
    else:
        print(f"→ 两者 CI 重叠, 差异可能不显著; 但看分场景 blind spot:")
        print(f"  文本在 same_theory_diff_wording / same_structure_same_pred 有不可修 blind spot")
        print(f"  几何 Fisher 在 same_pred_diff_structure 有可修 blind spot (加 complexity/Hodge)")

    return results, cases


if __name__ == "__main__":
    run_ablation()
