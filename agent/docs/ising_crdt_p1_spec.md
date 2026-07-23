# P1 Spec — 伊辛能量函数式 memory recall + CRDT 状态合并

> 治 P0 流式化之后的两个天花板: (1) memory recall 仍是 FTS5 top_k, 不会找"一致
> 子集"; (2) `dispatch_parallel` 多 subagent 结果只是 list, 无合并语义.
> 数学动机: Ising-Hopfield 同构 + 半格 (semilattice). 不引入新依赖, 全部复用
> `longterm.retrieve` / `dispatch_parallel` 已有接口.

## 动机

P0 (a6665af) 解决了"700 万步可观测性"和"subagent 增量上报". 但 `recall` 仍是
SQL `ORDER BY importance DESC, access_count DESC LIMIT top_k` ([longterm.py:637]
(file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/longterm.py#L637)),
召回的 K 条 memory 互相独立 — 没有考虑它们之间的语义关联. 700 万步场景里, 这
意味着召回的 K 条可能互相矛盾 (例如"encut=520 OK" + "encut=520 不够"), decider
拿到冲突 context 反复试错.

`dispatch_parallel` ([subagent_tool.py:251-268]
(file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py#L251-L268))
4 个并行 subagent 结果只是 `list[dict]`, 主 agent 自己 LLM 合并 — 4 份 finding
互相覆盖时无合并语义, 最后写的赢, 前面 3 份丢.

两个天花板共同的数学病灶: **召回/合并没有"一致性"度量**. 伊辛能量函数给 1, CRDT
半格给 2.

## P1-1: 伊辛能量函数式 memory recall

### 数学结构

Ising 模型与 Hopfield 网络同构. Hopfield 1982 的能量函数:

```
E(s) = -½ Σᵢⱼ Tᵢⱼ sᵢ sⱼ - Σᵢ Iᵢ sᵢ
```

直接对应 Ising:

| Hopfield | Ising | 物理意义 |
|---|---|---|
| sᵢ ∈ {-1, +1} | 自旋 | memory item "active/inactive" |
| Tᵢⱼ | Jᵢⱼ (exchange coupling) | memory-memory 关联 (semantic similarity) |
| Iᵢ | Hᵢ (external field) | query-memory 相关性 |
| E 最低态 | ground state | 一致 memory subset |

**召回 = 求 ground state**: 给定 query 产生外场 Hᵢ = sim(query, mᵢ), 用
Tᵢⱼ = sim(mᵢ, mⱼ) 做 memory 间耦合. 召回不再是"top_k 独立行", 而是
**能量最低的 K-子集** — 互相矛盾的子集能量高 (Tᵢⱼ < 0 时同时激活代价大),
互相支撑的子集能量低 (Tᵢⱼ > 0 时共激活).

### Modern Hopfield → attention 同构

Ramsauer 2020 证明 Modern Hopfield 的 retrieval 公式与 transformer attention
同构:

```
ξ_new = softmax(β X^T q) X
```

其中 X 是所有 memory patterns, q 是 query. β → ∞ 时退化为传统 Hopfield.
这给了升级路径: P1-1 v1 用 Ising 能量函数排序 (无新依赖); v2 直接用 attention
层做 retrieval (需要 embedding, 但 longterm.py 已有 `semantic=True` 路径).

### v1 接入 — 最小改动, 不引入新依赖

在 `LongTermMemory.retrieve` 后加一个 `_ising_rerank` post-process:

```python
def _ising_rerank(
    self,
    query: str,
    candidates: list[dict],   # FTS5 拉的 top_k * 3 候选
    top_k: int,
    beta: float = 1.0,
) -> list[dict]:
    """伊辛能量函数 re-rank. 把 FTS5 top_k 独立排序升级为能量最低 K-子集.

    Hᵢ = sim(query, mᵢ) — 外场, 用现有 _embedding_cosine (semantic=True 已有).
    Tᵢⱼ = sim(mᵢ, mⱼ) — memory-memory 耦合, 同样 _embedding_cosine.
    E(S) = -Σᵢ∈S Hᵢ - β Σᵢ<ⱼ∈S Tᵢⱼ  (sᵢ=+1 全激活, S 是选中的子集)
    选 S* = argmin_S E(S), |S| = top_k.

    ponytail: 不做精确 ground state (NP-hard), 用贪心 — 按 Hᵢ 排序逐个加入,
    每步算 ΔE, 若 ΔE < 0 接受, 否则跳过该 candidate. O(top_k * |candidates|²).
    ceiling: 贪心不保证全局最优; 升级路径: 模拟退火 / Modern Hopfield attention.
    """
```

### 接入点

- `LongTermMemory.retrieve` ([longterm.py:579]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/memory/longterm.py#L579))
  末尾在 `semantic=True` 路径调 `_ising_rerank`.
- 不动 `MemoryManager.recall` — 它是 thin wrapper, 自动继承.
- 新增 env `HUGINN_ISING_RERANK=1` 默认开 (off 时行为不变, 回归测试安全).

### 边界条件

- `semantic=False` (无 embedding) 路径跳过 rerank — 没有 Tᵢⱼ 没法算能量.
- `top_k=1` 时退化成原 FTS5 排序 — 单条无耦合.
- `len(candidates) < top_k` 时全保留 — 不靠 rerank 凑数.
- embedding 模型未初始化 (首次启动) → try/except 退化到原排序, 记 logger.warning.

### selfcheck 计划

- **29. 能量函数基本性质**: 给 3 条 candidates (m1, m2, m3), T₁₂=0.9 (高度相似),
  T₁₃=-0.5 (矛盾), H = [0.8, 0.7, 0.6]. 贪心应选 {m1, m2}, 不选 m3 (即便 H₃>0,
  ΔE 引入 m3 时 -H₃ - β(T₁₃+T₂₃) = -0.6 - 1·(-0.5-0.4) = -0.6 + 0.9 = +0.3 > 0,
  拒绝).
- **30. top_k=1 退化**: 单选应等价于 argmax Hᵢ.
- **31. semantic=False 跳过**: 验证 `_ising_rerank` 在无 embedding 时 no-op.
- **32. 整合 retrieve**: mock `_embedding_cosine`, 跑完整 retrieve 路径, 断言
  rerank 后顺序与原 FTS5 排序不同 (当 candidates 有矛盾时).

## P1-2: CRDT 状态合并

### 数学结构

多个 subagent 并行写共享 state, 无锁无冲突的数学结构是 **CRDT (Conflict-free
Replicated Data Types)**. 所有 CRDT 的共同性质: 状态空间是**交换半格 (commutative
semilattice)** `(S, ⊔)`, 满足:

- 交换律: a ⊔ b = b ⊔ a
- 结合律: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
- 幂等律: a ⊔ a = a

`⊔` 是 join (最小上界). 任意顺序合并结果相同 — 这就是无锁的数学根据.

### 量子纠缠启发 — no-signaling 边界

量子纠缠不能超光速传信息 (no-signaling theorem). 启发: 多 subagent 共享**参考
态** (ground truth), 但不能直接传信息. CRDT 的 join `⊔` 正是这个语义 — 每个
subagent 看到自己的局部视图, join 后得到一致全局视图, 但 join 过程不传"新信息"
(只是上确界).

边界: CRDT 不能解决"语义冲突" (例如 subagent A 说 encut=520, B 说 encut=572).
只能保证**字段级**合并无冲突. 语义冲突仍需 LLM 仲裁 (现有 `_resolve_support_finding`
[subagent.py:498-555]
(file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/agents/subagent.py#L498-L555)
已有, 复用).

### 三种 CRDT 类型 + 接入

| 类型 | 数学定义 | subagent 用途 |
|---|---|---|
| **G-Set** (grow-only) | (S, ∪), 仅 add | evidence list — 每个 subagent 加证据, 不删 |
| **LWW-Register** | (val, ts), max(ts) wins | 单值字段 (e.g. best_encut) — 时间戳新者胜 |
| **OR-Set** (observed-remove) | (A, R), add/remove 不冲突 | findings — 后期可删除被推翻的 finding |

### v1 接入 — `dispatch_parallel` 结果合并

在 `subagent_tool._dispatch_parallel` ([subagent_tool.py:206]
(file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py#L206))
末尾加 `_crdt_merge`:

```python
def _crdt_merge(results: list[dict]) -> dict:
    """CRDT-merge parallel subagent results.

    - findings: G-Set (union, dedupe by content hash)
    - evidence: G-Set (union)
    - best_value (任意 LWW 字段): 取 ts 最新
    - limitations: G-Set (union)

    ponytail: 只做字段级 CRDT, 语义冲突仍走 LLM 仲裁 (_resolve_support_finding).
    ceiling: G-Set 单调增, 长跑会膨胀; 升级 OR-Set 可删, 但需要 tombstone.
    """
```

### 接入点

- `SubagentTool._dispatch_parallel` ([subagent_tool.py:206]
  (file:///c:/Users/wanzh/Desktop/matsci-agent/agent/huginn/tools/subagent_tool.py#L206))
  最后 return 前调 `_crdt_merge(all_results)`.
- 新增 env `HUGINN_CRDT_MERGE=1` 默认开 (off 时 return list[dict] 原行为).
- `SubagentResult` 加可选字段 `ts: float` (LWW 用), 默认 `time.time()`.

### 半格一致性校验

CRDT 合并的核心是 `⊔` 满足半格公理. v1 实现需有不变量测试:

```python
# 交换律
assert merge(a, b) == merge(b, a)
# 结合律
assert merge(merge(a, b), c) == merge(a, merge(b, c))
# 幂等
assert merge(a, a) == a
```

### selfcheck 计划

- **33. G-Set union**: 2 subagent evidence list, merge 后无 dup.
- **34. LWW-Register**: 2 subagent best_encut 带 ts, 取 ts 大的.
- **35. 半格三公理**: 随机生成 3 个 subagent results, 验证交换/结合/幂等.
- **36. 整合 dispatch_parallel**: mock 4 个 subagent 返回不同 finding, 验证
  merge 后 finding list 无 dup 且 best_value 是 ts 最新.

## 实现优先级

| 项 | 优先级 | 估时 | 风险 |
|---|---|---|---|
| P1-1 v1 (Ising rerank) | 高 | 半天 | 低 — 纯 post-process, off 时回归 |
| P1-2 v1 (CRDT merge) | 高 | 半天 | 低 — 纯 return 前合并, off 时回归 |
| P1-1 v2 (Modern Hopfield attention) | 中 | 2-3 天 | 中 — 需 embedding 模型, 跟 semantic 路径耦合 |
| P1-2 v2 (OR-Set + tombstone) | 低 | 1-2 天 | 中 — tombstone GC 复杂 |

## 不做的事 (YAGNI)

- **不做** P1-1 v2 的全 attention retrieval — v1 贪心已能治"矛盾召回", v2 留给
  P2 HiLS 分层稀疏 attention 一起做.
- **不做** P1-2 的 Merkle tree 同步 — subagent 不跨进程, 不需要 anti-entropy.
- **不做** 分布式 CRDT (跨机器) — 当前单机多 subagent, 半格合并足够.
- **不做** 新的 embedding 模型 — 复用 `LongTermMemory._embedding_cosine` 已有.

## 升级路径

- P1-1 → P2-5 (HiLS 分层稀疏 attention): Ising 贪心是浅层 ground-state 近似,
  HiLS 的地标稀疏注意力是深层 attention 外推. P1-1 v1 的 `_ising_rerank` 可以
  平滑替换为 HiLS attention layer — 接口都是 `(query, candidates) -> ranked`.
- P1-2 → P2-6 (主动推理): CRDT 半格是状态合并, 主动推理 (Friston) 是信念更新.
  两者数学结构相似 (都是半格上的 join), P1-2 的 `_crdt_merge` 可以扩展为带
  Bayesian 信念更新的合并 — `merge(a, b)` 不只 join 字段, 还 update confidence.

## 参考

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent
  collective computational abilities. PNAS 79(8).
- Ramsauer et al. (2020). Hopfield Networks Is All You Need. ICLR 2021.
- Shapiro, M. et al. (2011). A comprehensive study of Convergent and
  Commutative Replicated Data Types. INRIA RR-7506.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  Nat Rev Neurosci 11:127-138.
