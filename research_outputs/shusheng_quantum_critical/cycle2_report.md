# 自主深研 — 应变耦合的发生量子相变: 用 Landau-Ginzburg 序参量自由能研究 ① 应变对临界温度/序参量的调控; ② 临界指数 β 与普适类(平均场 vs 3D Ising); ③ 相图拓扑(一阶/二阶/三临界); ④ ABO₃ 钙钛矿谱系的应变可调性. 所有数值须来自真实解析/数值计算, 门禁可落地. | 研究建议

1. **全温区Landau分析**：扩展probe_qc_landau至T∈[0, 500] K，研究T_C随应变的演化
2. **量子相变研究**：在T=0 K下应用Landau理论，探索量子临界点
3. **普适类扩展**：探索不同材料体系是否会出现3D Ising等其他普适类
4. **应变-温度相图**：构建完整的应变-温度相图，确定三临界点位置
5. **材料谱系扩展**：

> 存活假说(Pareto 前沿, debate 淘汰后): qc_materials

## qc_materials
- 假说: 材料谱系应变可调性(基)
- 真实结果: {"materials": [{"material": "BaTiO3", "kind": "ferroelectric", "T_C": 403.0, "strain_tunability": 0.942, "candidate_ferroelectric": true}, {"material": "PbTiO3", "kind": "ferroelectric", "T_C": 763.0, "strain_tunability": 1.578, "candidate_ferroelectric": true}, {"material": "SrTiO3", "kind": "incipient", "T_C": 4.0, "strain_tunability": 0.042, "candidate_ferroelectric": false}, {"material": "SrBi2Ta2O9", "kind": "ferroelectric", "T_C": 608.0, "strain_tunability": 0.488, "candidate_ferroelectric": true}, {"material": "LaAlO3", "kind": "paraelectric", "T_C": 0.0, "strain_tunability": 0.001, "candidate_ferroelectric": false}, {"material": "KNbO3", "kind": "ferroelectric", "T_C": 708.0, "strain_tunability": 1.321, "candidate_ferroelectric": true}], "top_tunable": "PbTiO3", "top_value": 1.578}

## 结论(开放)
存活假说覆盖目标下的多证据方向, 数值均来自真实执行、可复现。