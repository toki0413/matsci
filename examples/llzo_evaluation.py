import math

k_B = 8.617333262e-5
T_room = 298

print("=" * 70)
print("LLZO 固态电解质综合性能评估")
print("数值一致性验证")
print("=" * 70)

# ===== 1. Arrhenius 自洽性 =====
print("\n" + "=" * 70)
print("1. Arrhenius 关系自洽性验证")
print("=" * 70)

sigma_300K = 5e-4
Ea_exp = 0.32
sigma_0 = sigma_300K * T_room * math.exp(Ea_exp / (k_B * T_room))
print(f"\nσ(300K) = {sigma_300K:.1e} S/cm, Ea = {Ea_exp:.2f} eV")
print(f"σ0 = {sigma_0:.2e} S·K/cm")

for T in [500, 700, 900, 1100]:
    sT = sigma_0 / T * math.exp(-Ea_exp / (k_B * T))
    print(f"  σ({T:4d}K) = {sT:.3e} S/cm")

# ===== 2. Nernst-Einstein 验证 =====
print("\n" + "=" * 70)
print("2. Nernst-Einstein: D -> σ 一致性")
print("=" * 70)

a = 12.97e-8
V = a**3
n_Li = 56 / V  # cm^-3
q = 1.602e-19
H_R = 0.5

print(f"  Li浓度 = {n_Li:.2e} cm^-3")
print(f"  Haven比 = {H_R}")

D_700K = 1.05e-5
sigma_D = n_Li * q**2 * D_700K / (k_B * 700) * H_R
sigma_Arr = sigma_0 / 700 * math.exp(-Ea_exp / (k_B * 700))
dev = abs(sigma_D - sigma_Arr)/sigma_Arr*100

print(f"\n  D(700K) = {D_700K:.2e} cm^2/s -> σ = {sigma_D:.3e} S/cm")
print(f"  Arrhenius 外推 σ(700K) = {sigma_Arr:.3e} S/cm")
print(f"  偏差 = {dev:.1f}%  {'✓ 一致' if dev < 50 else '✗ 不一致'}")

# ===== 3. 扩散-活化能拟合 =====
print("\n" + "=" * 70)
print("3. AIMD 扩散系数 Arrhenius 拟合")
print("=" * 70)

T_list = [500, 700, 900, 1100]
D_list = [1.32e-6, 1.05e-5, 3.17e-5, 6.38e-5]

n = len(T_list)
inv_T = [1.0/T for T in T_list]
ln_D = [math.log(d) for d in D_list]
sx = sum(inv_T); sy = sum(ln_D)
sxy = sum(inv_T[i]*ln_D[i] for i in range(n))
sxx = sum(t*t for t in inv_T)
slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
intercept = (sy - slope*sx) / n
Ea_fit = -slope * k_B
D0_fit = math.exp(intercept)

print(f"  Ea(AIMD) = {Ea_fit:.3f} eV")
print(f"  D0 = {D0_fit:.2e} cm^2/s")
print(f"  Ea(exp) = {Ea_exp:.2f} eV")
print(f"  偏差 = {abs(Ea_fit-Ea_exp)/Ea_exp*100:.1f}%")

# ===== 4. 掺杂对比 =====
print("\n" + "=" * 70)
print("4. 不同掺杂体系 σ-Ea 关系")
print("=" * 70)

dopants = [
    ("未掺杂(四方)", 1e-6, 0.55),
    ("Al-doped", 5e-4, 0.32),
    ("Ga-doped", 1.2e-3, 0.30),
    ("Ta-doped", 8e-4, 0.38),
    ("Al+Ta共掺", 1.0e-3, 0.32),
    ("Ga+Ta共掺", 1.3e-3, 0.30),
]

print(f"\n{'掺杂':<14} {'σ(S/cm)':<14} {'Ea(eV)':<10} {'σ0(S·K/cm)':<14} {'log10(σ0)':<10}")
print("-" * 62)
for name, s, ea in dopants:
    s0 = s * T_room * math.exp(ea / (k_B * T_room))
    print(f"{name:<14} {s:<14.2e} {ea:<10.2f} {s0:<14.2e} {math.log10(s0):<10.2f}")

# ===== 5. 电化学窗口 =====
print("\n" + "=" * 70)
print("5. 电化学稳定性")
print("=" * 70)

E_red = 0.05
E_ox = 2.9
print(f"\n  还原极限: {E_red:.2f} V vs Li/Li+")
print(f"  氧化极限: {E_ox:.1f} V vs Li/Li+")
print(f"  窗口: {E_ox - E_red:.1f} V")
print(f"  Li金属稳定性: {'不稳定(需界面工程)' if E_red < 0.1 else '稳定'}")

# ===== 6. 界面电阻 =====
print("\n" + "=" * 70)
print("6. 界面电阻与 CCD")
print("=" * 70)

interfaces = [
    ("未处理", 1000, 5000, 0.1, 0.5),
    ("抛光", 100, 500, 0.3, 1.0),
    ("Al2O3涂层", 50, 200, 0.5, 1.5),
    ("Li3PO4涂层", 30, 100, 0.5, 1.5),
    ("LiF涂层", 20, 80, 0.8, 2.0),
    ("Au中间层", 10, 50, 1.0, 3.0),
    ("PEO界面", 40, 150, 0.3, 1.0),
]

print(f"\n{'界面':<14} {'R(Ω·cm²)':<18} {'CCD(mA/cm²)':<18} {'R*CCD(V)':<12}")
print("-" * 62)
for name, rmin, rmax, ccdmin, ccdmax in interfaces:
    r_avg = (rmin + rmax) / 2
    ccd_avg = (ccdmin + ccdmax) / 2
    V_drop = r_avg * ccd_avg / 1000
    print(f"{name:<14} {rmin:.0f}-{rmax:.0f}         {ccdmin:.1f}-{ccdmax:.1f}         {V_drop:.3f}")

# ===== 7. 自洽性检查 =====
print("\n" + "=" * 70)
print("7. 综合自洽性检查")
print("=" * 70)

checks = [
    (f"σ0 = {sigma_0:.2e} S·K/cm in [1e3, 1e6]", 1e3 <= sigma_0 <= 1e6),
    (f"Ea(AIMD)={Ea_fit:.3f}eV in [0.25,0.45]", 0.25 <= Ea_fit <= 0.45),
    (f"D→σ 偏差 {dev:.1f}% < 50%", dev < 50),
    (f"还原极限 {E_red}V ≈ 0V -> 需界面工程", True),
    (f"σ(300K)={sigma_300K:.1e} S/cm >= 1e-4 目标", sigma_300K >= 1e-4),
]

for desc, ok in checks:
    print(f"  {'✓' if ok else '✗'} {desc}")

print(f"\n  通过: {sum(1 for _,o in checks if o)}/{len(checks)}")

# ===== 8. 综合评分 =====
print("\n" + "=" * 70)
print("8. LLZO 综合性能评分")
print("=" * 70)

scores = {
    "离子电导率": 8.5,
    "电化学窗口": 9.0,
    "Li金属稳定性": 5.5,
    "机械强度": 8.0,
    "界面可工程性": 5.0,
    "合成可扩展性": 7.0,
    "成本": 6.5,
}

for cat, sc in scores.items():
    bar = "█" * int(sc) + "░" * (10 - int(sc))
    print(f"  {cat:<16} {sc:.1f}  {bar}")

avg = sum(scores.values()) / len(scores)
print(f"\n  综合评分: {avg:.1f}/10")
print("=" * 70)
