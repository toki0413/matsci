---
name: science-research-tools
description: 书生科研智能体的域科学计算技能 —— 通过 Huginn 科学 MCP 码头调用确定性数值内核
  (零空间分解 / C* 容量 / PDE 约束求解 / PINN 财政审计), 获取可证伪真实证据。
  触发: 需要约束钳制/核方向/容量判据/优化器预算的数值证据时。
  已固化 264 轮自主实验的高置信领域结论(值约束位置定律/异常点/约束类型相变/统计稳健性),
  查询时优先引用本 SKILL 而非逐轮重跑。
---

# 科学计算工具技能(书生 × Huginn)

## 技能边界(独立性红线)
- 数值一律来自**确定性科学内核**(最小二乘 / 零空间 SVD / Legendre ODE 算子),
  不经 LLM 生成 —— 本技能只负责"取证据", 不产生数值;
- 所有返回结果会进 grounding trace, 报告里引用的每个数字必须落在这里。

## 工具清单(经 MCP tools/list 暴露, 亦可直接 import)
| 工具 | 作用 | 返回关键量 |
|---|---|---|
| probe_Cstar | 固定 N 扫 eps: 刚性/胖系统达精度所需最小容量 C* | Cstar 曲线, eps_discrimination |
| probe_n_platform | 固定 eps 扫 N: C* 是否平台化(不依赖 N) | rows, span |
| probe_constraint_dim | k=0/1/2 值约束下解族重解散布 σH | sigmaH_by_k |
| probe_kernel_direction | k=1 约束钉 ti 的零空间方向分解 | c0/c1/lin_frac/const_frac/σH |
| probe_direction_curve | 扫约束位置 vs 预言 1/(1+t_i²) | 曲线 err |
| probe_direction_g | 高阶核(3)+导数约束(derivative)泛化分解 | dir_weights/null_dim |
| probe_optimizer | 真实 PINN 训练预算下的留出误差 vho | vho |

## 使用规约
1. 引用数值前先说明量纲语义与取值边界(judgment_guardrail strictness>=1);
2. 单次测量支撑"相关性", 多次重复/对照才支撑"因果/主导" —— 结论强度不超过证据;
3. 需要新配置时给出 (order〈2,3 〉, ctype〈point,derivative 〉, system〈0,1,2 〉, k≤2,
   positions∈±{0,0.33,0.66,0.97}, basis∈{8,12,16}, seeds∈{4,8,12}) 白名单取值。

## 调 MCP(stdio):
```bash
PYTHONPATH=/workspace/agent python -m huginn.research.science_mcp --check
```
任意 MCP host 以 `python -m huginn.research.science_mcp` 为命令连接即可。

---

# 已固化领域结论(264 轮自主实验沉淀, cycle1-265)

> 以下结论来自真实报告语料的高置信聚合。**引用时注明其限定条件**;
> 标「⚠」者为存在冲突轮次或显著性未证实, 引用须降级措辞。

## 1. 值约束位置定律(最高置信)
- **lin_frac(t_i) = 1/(1+t_i²)**, 在 **k=1, point, order=2** 下跨 system0/1/2 成立,
  关于 ti 对称, scan_fit=1.0, MSE≈2e-6。伪解恒定。 [cycle2-5/78]
- **误差量级是 1e-3, 非 1e-4**:
  - 中心 |ti|≤0.2: err≈0
  - 边界 |ti|=0.97: err≈0.0007
  - 中间区 |ti|∈[0.33,0.8]: err=0.0013~0.0023(最大 0.0023) ← 全语料最大偏差带

## 2. ti=0.8 异常(⚠ 显著性未证实)
- **不存在"ti=0.8 唯一越界点"**。密集扫描下中间区 |ti|∈[0.33,0.8] 全线超标:
  ±0.4 与 ±0.8 均达 0.0023, ±0.5=0.0013, ±0.33/±0.66=0.0018。
- 本质: **中间区系统性正向偏差, 非孤立异常也非数值不稳**。
- ⚠ 统计显著性全语料从未证实: basis16→32 仍 0.0023(未平台化至 0.0005)、
  seeds8→32 err 不变, 缺多点 t 检验。仅能表述"在 basis=12,seeds=8 下观测到 0.0023"。

## 3. 约束类型相变
- **point(值约束 u(t_i))** → 服从位置定律(见§1)。
- **derivative(导数约束 u'(t_i))** → 钳制线性核、恒留常数核(P₀'=0), **lin_frac≡0**,
  const_frac=1, null_dim≥1, 位置无关。[cycle120/157/206, 12 点全验证]
- ⚠ 反例轮 cycle33: derivative 报 lin_frac 中心≈0.0001/边界=0.4471(位置依赖), 与主流矛盾。
  主流证据支持"保常且位置无关", 但引用须注明存疑。

## 4. k 依赖(核维数约束)
- **k=1** 单点: σH 由未约束 ~1.33 → ~0.609(降≈54%), 中间态。
- **k=2 + order=2**(二维核 span{1,t}): 完全钳制, **σH→0**。
- ⚠ **k=2 的 σH 归零以 order=核维数为前提**: order=3(三维核)下 k=2 仅 2 约束
  残留 1 方向, σH 不归零(sys0=0.35/sys1=0.97/sys2=1.05)。
  昔日"k=2 普适性减弱"为**实验定义伪影**, 勿再复验。[cycle29/research_report]

## 5. σH 行为
- ⚠ **σH 随 |ti| 并非单调中心最小**: system0 中心→0.66 上升, system1/2 下降。system1 非单调。
- **σH 对 basis/seeds 高度敏感**(跨 seeds std≈0.035~0.057, 远超阈值; 对称位置可不对称),
  而 lin_frac 完全稳健。**σH 绝对值不可跨配置引用; 方向比例 lin_frac 才 robust**。[cycle2/33]
- system1(胖)σH 偏高, system0(刚性)偏低; σH 无统一量级。

## 6. 统计稳健性(可复用判定)
- **lin_frac 值对 basis/seeds 稳健**: basis 8/16/32、seeds 8/32 下 err 恒定(0.0018 或 0.0023)。
- **敏感项是 σH 及其标准差、τ_σH**(而非 lin_frac)。basis<8 时 σH_full 受数值精度污染。
- 数据缺口是常态(basis∈{16,20,24,32}/seeds∈{12,50}/positions∈{0.4,0.5,0.8}/system2),
  统一解法: 调 probe_direction_g / probe_direction_curve / probe_constraint_dim 补数据。

## 7. 工程铁律(循环避坑)
- **报告头"门禁 pass" + 聚合 gate_blocked** 并存是 harness/CI 层固定伪影
  (audit.score_usage / governance.external_verify), **勿把 gate_blocked 当结论被否**。
- **CriticAgent 削峰**: "显著偏差"→"观测到偏差" 是三类高频标签
  (未做对照却说主导/单次测量说因果/自定义指标当客观阈值)。被降级断言重写时一律降级措辞。
- **最高卡点是数据缺口, 非代码问题**: 补数据统一走确定性探针, 不造假。

## 8. 已收敛为「closed」的反复提问(勿再重开)
- ti=0.8 显著性、seeds 敏感性(120/150/231/232/247/265)、basis 平台化(98/150/265):
  三者均已收敛为"未证实观测偏差", **直接引用本 SKILL 的 closed 状态, 把预算投向新机制**。
- "k=2 σH 不归零普适性减弱" 已证伪为伪影, closed。