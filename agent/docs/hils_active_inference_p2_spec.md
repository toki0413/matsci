# P2 Spec — HiLS 分层稀疏 attention + 主动推理框架

> 治 P1 之后的两个长期天花板: (1) Ising 贪心是浅层 ground-state 近似, 长程
> memory 关联仍丢失; (2) CRDT 半格只做字段级 join, 没有"信念更新"语义.
> 数学动机: 分层地标稀疏注意力 (HiLS) + 自由能原理 (Free Energy Principle).
> 这是研究轨道, 不是工程 patch — 每项都需数学推导 + 实验验证, 不承诺时间表.

## 动机

P1 (04e63c4) 解决了"召回一致性"和"并行合并无冲突". 但两个深层天花板仍在:

1. **Ising 贪心的局部最优陷阱**. `_ising_rerank` 按 H 降序贪心加入, ΔE<0 接受.
   这是 NP-hard ground-state 的近似, 在 memory 数量 >100 时局部最优可能远离
   全局最优. 700 万步场景 memory 可能 >10K, 贪心基本失效. 升级路径有两条:
   模拟退火 (通用但慢) / Modern Hopfield attention (跟 transformer 同构, GPU
   加速). P2-5 选第二条 — 它跟 HiLS 分层稀疏 attention 自然结合.

2. **CRDT 合并不更新 confidence**. `_crdt_merge` 的 LWW-Register 只取 ts 大者,
   不做 Bayesian 信念更新. subagent A 说 encut=572 (3 次成功), B 说 encut=520
   (1 次成功, 但更新), LWW 选 B (ts 新) — 但 A 的信念更强. P2-6 引入主动推理
   (Friston Free Energy), 把信念更新变成 free energy minimization.

两个天花板的共同数学病灶: **召回/合并没有"长程关联"和"信念"的数学结构**. HiLS
attention 给 1, Free Energy Principle 给 2.

## P2-5: HiLS 式分层稀疏 attention for memory

### 数学结构

HiLS (Hierarchical Landmark Sparse Attention, arxiv 2607.02980) 把 attention
的 O(n²) 降到 O(n log n) 甚至 O(n), 同时保持长程关联. 核心思想:

1. **地标 (landmark)**: 从 N 个 memory 中选 K << N 个"代表", 用聚类或
   importance 采样. 地标集 L = {l₁, ..., l_K}.
2. **分层**: memory 按相似度分到地标下, 形成树结构. query 先跟地标算 attention,
   再只跟 top-h 个地标下的 memory 算精细 attention.
3. **稀疏**: 每层只跟固定数量邻居交互, 总复杂度 O(N·h) 而非 O(N²).

### Modern Hopfield ↔ attention 同构

Ramsauer 2020 证明:

```
ξ_new = softmax(β X^T q) X
```

跟 transformer attention 同构. β 是温度参数, β→∞ 退化为传统 Hopfield.
P2-5 用这个同构:

- X = 所有 memory embeddings (N × d)
- q = query embedding (1 × d)
- β = 1/T (温度, 跟 Ising 的 β 同构)
- 输出 ξ_new = 软组合的 memory pattern

跟 P1-1 `_ising_rerank` 的关系: P1-1 是离散 ground-state (sᵢ ∈ {0, 1}), P2-5
是连续 softmax 组合 (αᵢ ∈ [0, 1]). P2-5 是 P1-1 的连续松弛.

### 分层稀疏化 — 治 700 万步的 N>10K

直接算 `softmax(β X^T q) X` 是 O(N·d). N=10K, d=768 时一次 retrieval 要 7.6M
浮点 ops, 可接受. 但 700 万步场景 N 可能 >100K, 而且每步都要 retrieve, 不可接受.

HiLS 的解法:

```
Layer 0 (地标层):  选 K=256 个地标 (k-means on X)
                  q 跟 256 个地标算 attention → O(K·d) = O(256·768) ≈ 200K ops
                  选 top-h=8 个地标

Layer 1 (精细层):  只跟 top-h 地标下的 memory 算 attention
                  每地标下约 N/K = 400 memory → O(h·N/K·d) = O(8·400·768) ≈ 2.5M ops

Total: ~2.7M ops vs 全 attention 76.8M ops → 28x 加速
```

### 接入点

- `LongTermMemory._ising_rerank` ([longterm.py:589]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/longterm.py#L589))
  升级为 `_hils_attention`. 接口不变 `(query, candidates, top_k) -> ranked`,
  内部用 HiLS attention 替代贪心.
- 地标缓存: `_vector_store` 加 `_landmarks` 字段, lazy init (首次 retrieve 时
  k-means, 之后周期性更新 — env `HUGINN_HILS_LANDMARK_REFRESH=1000` 每 1000 次
  retrieve 重算).
- 不动 `retrieve` 主流程 — 只替换 rerank 函数.

### 边界条件

- N < K (memory 少于地标数) → 退化到全 attention, 不分层.
- embedding 模型未初始化 → 回退到 P1-1 Ising 贪心 (已有 fallback).
- 地标缓存失效 (新 memory 写入) → 增量更新: 新 memory 找最近地标挂上, 不重算.
- query 是空字符串 → 用 `_embedding_cosine` 的 zero vector, 退化到按 importance 排序.

### 数学风险 — 需实验验证

1. **地标选择的影响**: k-means 的 K 和 init 影响结果. 需 A/B: K=128/256/512,
   init=random/k-means++. 评测指标 = retrieval recall@10.
2. **top-h 的 trade-off**: h 太小漏召回, h 太大失去稀疏性. 需 sweep.
3. **地标更新频率**: 太频繁开销大, 太慢 stale. 需测: stale landmark 对 recall
   的影响.

### selfcheck 计划 (待 P2-5 实现时补)

- 38. HiLS attention 基本: 给 100 个 candidates, K=10, h=3, 验证 top_k=5 的
  结果跟全 attention 一致 (容差 1e-3).
- 39. N < K 退化: candidates=5, K=10, 验证走全 attention 路径.
- 40. 地标缓存: 100 candidates, 2 次 retrieve, 验证地标只算一次.
- 41. 增量更新: 加 1 个新 memory, 验证地标不重算, 新 memory 挂到最近地标.

## P2-6: 主动推理框架 (Active Inference)

### 数学结构

Friston 自由能原理 (Free Energy Principle, FEP):

```
F(q, p) = E_q[log q(s) - log p(s, o)]
        = KL[q(s) || p(s|o)] - log p(o)
        ≥ -log p(o)  (variational bound)
```

其中:
- s = 隐状态 (材料的真实 encut/结构/性质)
- o = 观测 (DFT 计算结果)
- q(s) = agent 的后验信念 (variational distribution)
- p(s, o) = 生成模型 (agent 对"材料如何产生观测"的认知)

**主动推理**: agent 通过两个途径最小化 F:
1. **感知 (perception)**: 更新 q(s) 使其接近 p(s|o) — Bayesian 信念更新.
2. **行动 (action)**: 选择能产生最一致观测的 action — 改变世界使 o 跟 q(s) 一致.

### 跟 P1-2 CRDT 的关系

P1-2 `_crdt_merge` 的 LWW-Register 只取 ts 大者, 不更新 confidence. P2-6 把
LWW 升级为 **Bayesian belief update**:

```
# P1-2 (LWW):
best_value = r.ts > current.ts ? r.value : current.value

# P2-6 (Bayesian):
posterior = likelihood(observation) * prior / evidence
            ↑                ↑           ↑
            r.value          current     normalizer
```

具体: 每个字段不再是单值, 而是 **分布** (Gaussian for 连续, Beta for 二值):
- `best_encut`: Gaussian(μ=520, σ=20) — 信念是均值 520, 标准差 20
- 新观测 o=572 (来自 subagent, 带 noise σ_o=10)
- 后验: μ' = (μ/σ² + o/σ_o²) / (1/σ² + 1/σ_o²) = (520/400 + 572/100) / (1/400 + 1/100) ≈ 561.6
- σ'² = 1 / (1/σ² + 1/σ_o²) ≈ 80 → σ' ≈ 8.9 (不确定性降低)

这比 LWW 的"新者胜"合理得多 — 多个独立观测会逐渐收敛, 单个离群观测不会翻转信念.

### 跟 P1-1 Ising 的关系

Ising 能量函数 E(s) 跟自由能 F(q, p) 数学结构相似:

```
E(s) = -Σᵢ Hᵢ sᵢ - β Σᵢⱼ Tᵢⱼ sᵢ sⱼ       (Ising)
F(q) = E_q[log q(s)] - E_q[log p(s, o)]    (Variational)
```

P2-6 把 Ising 的"外场 Hᵢ"重新解释为"log-likelihood of observation", "耦合
Tᵢⱼ"重新解释为"prior correlation between memories". 这统一了 P1-1 和 P2-6
的数学框架.

### 接入点

- `SubagentResult` 加 `belief` 字段 (dict[field, Gaussian/Beta params]),
  默认 None (向后兼容).
- `_crdt_merge` ([subagent_tool.py:39]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py#L39))
  LWW 分支升级为 Bayesian update (当 result 带 belief 时).
- `_resolve_support_finding` ([subagent.py:498]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agents/subagent.py#L498))
  用 belief 的 KL divergence 判定冲突 — KL > threshold 才走 LLM 仲裁.

### 主动推理的"行动"部分

FEP 的 action: agent 选择能最小化未来自由能的 action. 对 autoloop:

```
action = argmin_a E[F(q', p') | a]
where q' = Bayesian_update(q, observe(a))
      p' = updated_generative_model(a)
```

接入到 `_decide_next_action_llm` ([engine.py]
(file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py)):
decider prompt 加 "expected free energy" 字段 — 每个 candidate action 算
预期自由能, LLM 参考 (不强制, 跟 speculator hint 同模式).

### 边界条件

- `belief` 字段缺失 → 回退到 P1-2 LWW (向后兼容).
- 单观测无 prior → 用无信息先验 (Gaussian: μ=obs, σ=∞; Beta: α=β=1).
- 多观测冲突大 (variance 不降反升) → 标记为 "高不确定性", decider 优先探索.
- 连续字段用 Gaussian, 二值字段 (success/fail) 用 Beta, 离散字段 (categorical)
  用 Dirichlet.

### 数学风险 — 需实验验证

1. **生成模型的正确性**: p(s, o) 怎么定? 材料领域没有现成的生成模型. 需从
   DFT 数据学一个 surrogate (e.g. GPR / neural process). 这本身是研究问题.
2. **自由能估计的方差**: Monte Carlo 估计 F 的方差可能大, 影响决策稳定性.
   需测: 不同 random seed 下 action 选择的一致性.
3. **探索-利用 trade-off**: FEP 的 action 是 pure exploitation (最小化 F).
   需加 temperature / KL regularization 鼓励探索, 否则 agent 卡局部最优.

### selfcheck 计划 (待 P2-6 实现时补)

- 42. Gaussian 更新: prior N(520, 20²), obs N(572, 10²), 验证后验 μ≈561.6,
  σ≈8.9.
- 43. Beta 更新: prior Beta(1, 1), 3 success, 1 fail, 验证后验 Beta(4, 2).
- 44. LWW fallback: result 无 belief 字段, 验证回退到 LWW.
- 45. KL 冲突检测: 两个 belief KL > threshold, 验证标记为 "需 LLM 仲裁".

## 实现优先级

| 项 | 优先级 | 估时 | 风险 | 依赖 |
|---|---|---|---|---|
| P2-5 v1 (HiLS attention, 全 attention 不分层) | 中 | 1-2 天 | 低 — P1-1 平滑升级 | P1-1 |
| P2-5 v2 (分层稀疏, K-means 地标) | 低 | 3-5 天 | 中 — 地标选择影响召回 | v1 |
| P2-6 v1 (Gaussian/Beta belief update for CRDT) | 中 | 2-3 天 | 中 — 生成模型需从数据学 | P1-2 |
| P2-6 v2 (active action in decider) | 低 | 1 周 | 高 — FEP 估计方差大 | v1 |

## 不做的事 (YAGNI)

- **不做** P2-5 的 GPU 加速 — N<10K 时 CPU 够用, GPU 加速留给 N>100K 的极限场景.
- **不做** P2-6 的完整 FEP 推导 — v1 只做 belief update (perception), active
  action (action) 留给 v2.
- **不做** 生成模型的端到端学习 — v1 用 Gaussian 假设 (闭合解), 不学 surrogate.
  surrogate 学习是独立研究问题, 不混入 P2.
- **不做** 跨 agent 的 belief 同步 — 当前单机多 subagent, belief 在
  `_crdt_merge` 里合并即可. 跨机器同步留给分布式版本.

## 升级路径

- P2-5 → P3 (量子注意力): HiLS 是经典分层稀疏, 量子注意力 (Quantum Attention,
  e.g. Quantum Perceptron) 用量子叠加态做并行 attention. 数学结构: Hilbert
  空间上的投影算子替代 softmax. 这是远期研究方向.
- P2-6 → P3 (全球模型 learning): v1 用 Gaussian 假设, v2 学 surrogate, P3 用
  neural process 学生成模型本身. 跟 huginn 的 surrogate 模型栈 (GPR / NN)
  自然结合.

## 跟 huginn 现有架构的关系

- **P2-5** 跟 `LongTermMemory._vector_store` (ChromaDB) 共生 — ChromaDB 已有
  HNSW 索引 (O(log N) 近邻搜索), HiLS attention 是它的 retrieval 后的 rerank.
  不替代 ChromaDB, 只替代 `_ising_rerank`.
- **P2-6** 跟 `model_router` ([engine.py:7834]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/autoloop/engine.py#L7834))
  共生 — model_router 选 LLM, P2-6 选 action. 两者正交, 不冲突.
- **P2-6** 的 belief 字段跟 `SubagentSpec.summary_format="json"` ([subagent.py:182]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agents/subagent.py#L182))
  自然结合 — support spec 已经返回结构化 JSON, 加 belief 字段是 schema 扩展.

## 参考

- HiLS: arxiv 2607.02980 (Hierarchical Landmark Sparse Attention).
- Ramsauer et al. (2020). Hopfield Networks Is All You Need. ICLR 2021.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  Nat Rev Neurosci 11:127-138.
- Friston, K. et al. (2017). Active Inference: A Process Theory. Neural
  Computation 29(1):1-49.
- Da Costa, L. et al. (2020). Active inference on discrete state-spaces:
  a synthesis. Math Psychol 99:102447.
- Blei, D. et al. (2017). Variational Inference: A Review for Statisticians.
  J Am Stat Assoc 112(518):859-877.
