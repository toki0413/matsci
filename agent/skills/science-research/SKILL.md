---
name: science-research-tools
description: 书生科研智能体的域科学计算技能 —— 通过 Huginn 科学 MCP 码头调用确定性数值内核
  (零空间分解 / C* 容量 / PDE 约束求解 / PINN 财政审计), 获取可证伪真实证据。
  触发: 需要约束钳制/核方向/容量判据/优化器预算的数值证据时。
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