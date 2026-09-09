# 自主深研(Huginn×书生)

> **门禁: pass** (未落地: 无)
> 聚合视图: verdict=pass> real orchestration: explored=7 pruned=0 convergence=max_iterations reached
> 报告来源: deterministic

# 自主深研 — 面向 IJF 2026 综述《Outstanding issues and emerging frontiers in fracture mechanics》(杨卫/冯西桥/高华健) 抽取的七个可计算开放问题做真实数值检验: ①界面裂纹互穿悖论(Dundurs ε 跨材料对跨度); ②脆韧转变 γusf/γs 图谱对材料族的区分度; ③纳尺度缺陷容差(强度-缺陷尺寸交叉, 临界缺陷尺寸数量级); ④统计弱链尺寸效应指数; ⑤桥联增韧增益; ⑥mode-I 动态断裂 Rayleigh 势垒与声学缺口结构; ⑦ASTM E399 KIC 试样尺寸门槛的工程跨度。要求: 全部数值来自真实物理计算, 可复现, 每个 open issue 对应独立证据分支。

> 存活假说(Pareto 前沿, debate 淘汰后): F3_flaw_tolerance, F1_interface_oscillation, F4_statistical_size, F6_dynamic_barrier, F2_dbt_map, F5_bridging_toughening, F7_kic_validity

## F3_flaw_tolerance
- 假说: 基准: 纳尺度缺陷容差(尺度坐标)
- 真实结果: {"amax_nm": 100.0, "materials": [{"material": "graphene(2D)", "E_GPa": 1000.0, "sig_th_GPa": 130.0, "a*_nm": 0.301, "sigma/sigth @100nm": 0.0549, "curve(σ/σth vs a/a*)": [1.0, 1.0, 1.0, 0.7071, 0.4472, 0.3162, 0.1581]}, {"material": "Si", "E_GPa": 170.0, "sig_th_GPa": 7.0, "a*_nm": 2.098, "sigma/sigth @100nm": 0.1449, "curve(σ/σth vs a/a*)": [1.0, 1.0, 1.0, 0.7071, 0.4472, 0.3162, 0.1581]}, {"material": "steel(bcc-Fe)", "E_GPa": 200.0, "sig_th_GPa": 11.0, "a*_nm": 1.052, "sigma/sigth @100nm": 0.1026, "curve(σ/σth vs a/a*)": [1.0, 1.0, 1.0, 0.7071, 0.4472, 0.3162, 0.1581]}, {"material": "Al2O3", "E_GPa": 390.0, "sig_th_GPa": 12.0, "a*_nm": 2.586, "sigma/sigth @100nm": 0.1608, "curve(σ/σth vs a/a*)": [1.0, 1.0, 1.0, 0.7071, 0.4472, 0.3162, 0.1581]}], "min_a*_nm": 0.301, "max_a*_nm": 2.586}

## F1_interface_oscillation
- 假说: 界面互穿悖论(界面线)
- 真实结果: {"pairs": [{"pair": "Al2O3/Ni 陶瓷-金属", "alpha": 0.287, "beta": 0.049, "|eps|": 0.01567, "log10(l/a)": -87.1}, {"pair": "SiC/Al 复合材料", "alpha": 0.684, "beta": 0.147, "|eps|": 0.04727, "log10(l/a)": -28.9}, {"pair": "PMMA/steel 聚合物-钢", "alpha": -0.971, "beta": -0.223, "|eps|": 0.07227, "log10(l/a)": -18.9}, {"pair": "glass/epoxy 玻璃/环氧", "alpha": 0.897, "beta": 0.2, "|eps|": 0.06468, "log10(l/a)": -21.1}, {"pair": "sapphire/NiAl 介电/金属间", "alpha": 0.346, "beta": 0.076, "|eps|": 0.02439, "log10(l/a)": -56.0}, {"pair": "diamond/WC 超硬-硬质", "alpha": 0.255, "beta": 0.06, "|eps|": 0.01907, "log10(l/a)": -71.5}], "min_eps": 0.01567, "max_eps": 0.07227, "osc_span": 0.0566}

## F4_statistical_size
- 假说: 统计弱链尺寸效应(统计线)
- 真实结果: {"m": 2.0, "slope(数值)": 0.2483, "theory(1/2m)": 0.25, "slope_theory_ratio": 0.9932, "r2(log-log线)": 0.9996}

## F6_dynamic_barrier
- 假说: Rayleigh 势垒(动态线)
- 真实结果: {"nu": 0.3, "cR/c2": 0.92741, "c1/c2": 1.87083, "curve": [{"v/cR": 0.1, "k(v)": 0.9474, "g=k^2": 0.8975}, {"v/cR": 0.25, "k(v)": 0.8571, "g=k^2": 0.7347}, {"v/cR": 0.5, "k(v)": 0.6667, "g=k^2": 0.4444}, {"v/cR": 0.7, "k(v)": 0.4615, "g=k^2": 0.213}, {"v/cR": 0.85, "k(v)": 0.2609, "g=k^2": 0.0681}, {"v/cR": 0.9, "k(v)": 0.1818, "g=k^2": 0.0331}, {"v/cR": 0.95, "k(v)": 0.0952, "g=k^2": 0.0091}, {"v/cR": 0.98, "k(v)": 0.0392, "g=k^2": 0.0015}], "barrier_sharpness(-dln g/d(v/cR)@0.95)": 38.64, "sonic_gap(c2-cR)/cR": 0.078}

## F2_dbt_map
- 假说: 确认组: 脆韧图谱
- 真实结果: {"materials": [{"material": "diamond", "mu": 535.0, "b_nm": 0.252, "gamma_usf/gamma_s": 1.698, "class": "brittle"}, {"material": "Si", "mu": 68.0, "b_nm": 0.384, "gamma_usf/gamma_s": 1.452, "class": "brittle"}, {"material": "W", "mu": 161.0, "b_nm": 0.274, "gamma_usf/gamma_s": 1.207, "class": "brittle"}, {"material": "α-Fe", "mu": 82.0, "b_nm": 0.248, "gamma_usf/gamma_s": 0.474, "class": "ductile"}, {"material": "Ti", "mu": 44.0, "b_nm": 0.295, "gamma_usf/gamma_s": 0.633, "class": "transitional"}, {"material": "Mg", "mu": 17.0, "b_nm": 0.32, "gamma_usf/gamma_s": 0.667, "class": "transitional"}, {"material": "Cu", "mu": 48.0, "b_nm": 0.256, "gamma_usf/gamma_s": 0.196, "class": "ductile"}, {"material": "Ni", "mu": 76.0, "b_nm": 0.249, "gamma_usf/gamma_s": 0.182, "class": "ductile"}, {"material": "Al", "mu": 26.0, "b_nm": 0.286, "gamma_usf/gamma_s": 0.175, "class": "ductile"}, {"material": "Au", "mu": 27.0, "b_nm": 0.288, "gamma_usf/gamma_s": 0.086, "class": "ductile"}], "min_idx": 0.086, "max_idx": 1.698, "dbt_span": 1.612}

## F5_bridging_toughening
- 假说: 确认组: 桥联增韧
- 真实结果: {"rows": [{"sigma0/sy": 0.05, "Kc/K0(δ预算0.5)": 1.0124}, {"sigma0/sy": 0.1, "Kc/K0(δ预算0.5)": 1.0247}, {"sigma0/sy": 0.2, "Kc/K0(δ预算0.5)": 1.0488}, {"sigma0/sy": 0.3, "Kc/K0(δ预算0.5)": 1.0724}, {"sigma0/sy": 0.5, "Kc/K0(δ预算0.5)": 1.118}, {"sigma0/sy": 0.7, "Kc/K0(δ预算0.5)": 1.1619}, {"sigma0/sy": 0.9, "Kc/K0(δ预算0.5)": 1.2042}], "max_Kc/K0": 1.2042, "kce_gain": 0.2042}

## F7_kic_validity
- 假说: 确认组: KIC 有效性边界
- 真实结果: {"materials": [{"material": "A533B 压力容器钢", "sigma_y(MPa)": 345.0, "KIC(MPa√m)": 200.0, "试样尺寸门槛(mm)": 0.84, "r_p(mm)": 0.018, "门槛/塑性区": 47.1}, {"material": "7075-T6 铝合金", "sigma_y(MPa)": 503.0, "KIC(MPa√m)": 29.0, "试样尺寸门槛(mm)": 0.01, "r_p(mm)": 0.0, "门槛/塑性区": 47.1}, {"material": "Ti-6Al-4V", "sigma_y(MPa)": 950.0, "KIC(MPa√m)": 55.0, "试样尺寸门槛(mm)": 0.01, "r_p(mm)": 0.0, "门槛/塑性区": 47.1}, {"material": "18Ni(250) 马氏体时效钢", "sigma_y(MPa)": 2400.0, "KIC(MPa√m)": 90.0, "试样尺寸门槛(mm)": 0.0, "r_p(mm)": 0.0, "门槛/塑性区": 47.1}, {"material": "Al2O3 陶瓷", "sigma_y(MPa)": 2600.0, "KIC(MPa√m)": 3.5, "试样尺寸门槛(mm)": 0.0, "r_p(mm)": 0.0, "门槛/塑性区": 46.6}, {"material": "PZT 压电陶瓷", "sigma_y(MPa)": 250.0, "KIC(MPa√m)": 1.0, "试样尺寸门槛(mm)": 0.0, "r_p(mm)": 0.0, "门槛/塑性区": 47.1}], "min_demand_mm": 0.0, "max_demand_mm": 0.84, "span(log10)": 5.27}

## 结论(开放)
存活假说覆盖目标下的多证据方向, 数值均来自真实执行、可复现。
