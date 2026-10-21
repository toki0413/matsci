# Spec: Normative Cognition Runtime — 从 PMK 文本一致性到「三账、两闸、一调度器、一梦境周期」

**日期**: 2026-10-22
**milestone**: M-NCR（Normative Cognition Runtime）
**状态**: 设计定稿，待实施
**关联**: `2026-09-13-recursive-compounding-design.md`、`2026-09-13-seam-integrity-gate.md`

---

## 0. 一句话目标

把 persona—memory—knowledge 循环从「三路文本快照 + 冲突暂停」重构为**三本可审计账本 + 两道独立闸门 + 一个无 LLM 的准入调度器 + 一个受治理的梦境周期**：认知账管「信什么」，规范账管「被允许做什么」，经验账管「做过且能复现什么」；记忆治理拆成**保留闸**（可用/归档/巩固/修剪）与**影响闸**（此刻是否、以多大权威塑造推理），检索命中只是候选者，不等于授权进入上下文；梦境产物走**写入侧治理**，不与在线证据同权。PMK 默认开启，冲突不再靠文本对立词判定，而靠两张 sheaf 上的 H¹ 障碍分别裁决。

一句话主张（可反驳）：

> **Identity is an admission cut, not a manifold.**
> Agent 的连续性不需要潜空间皮肤，需要的是授权拓扑上的切口；切口失效则行为漂移必然复现。

若在实际部署中，切口（准入拓扑）无法把跨会话行为漂移压到阈值以下，本主张即被证伪，应回到潜空间约束路线。

---

## 1. 前提修正（相对上一轮讨论）

上一轮的三条判断已被用户纠正，本 spec 以此为前提，不再保留旧假设：

| 旧假设 | 修正后 | 对本 spec 的影响 |
|---|---|---|
| Huginn 是单机研究工作区 | **多部署形态**：桌面 / CLI / MCP / Docker / 远程 HPC / 多 agent | 规范账必须带 `tenant_id / deployment_id / device_id`；事件日志追加式、投影确定性、跨设备收敛；隔离是硬不变量 |
| PMK 默认关闭 | **PMK 默认开启** | `HUGINN_PMK_INJECT` 默认改 `1`；新增 `HUGINN_PMK_MODE=off|shadow|enforce`，默认 `enforce`；同步更新治理锚点，不允许静默翻转 |
| 无大规模轨迹，RL 记忆管理 ROI 低 | **有大规模轨迹可训练** | AgeMem / DeltaMem / NSARL 路线的离线版本进入范围：轨迹索引 + 离线策略学习 + OPE（off-policy evaluation）；硬约束作为 action mask，不进 reward |

---

## 2. 背景证据（实测，非推断）

### 2.1 现状缺口

- PMK 是**文本快照**：[build_pmk_state](file:///workspace/agent/huginn/autoloop/cognitive_guard.py#L117-L278) 返回 `dict[str,str]`，persona 取 `description`、memory 取召回文本、kb 取 top-2 chunk 的 200 字；没有任何可寻址 id、时间戳、置信度。
- 冲突判定是**词面语义**：[`_check_pmk_consistency`](file:///workspace/agent/huginn/runtime/task_lifecycle.py#L435-L497) 用推荐/反对词 + subject token 重合 ≥1；[`_pmk_gluing_obstruction`](file:///workspace/agent/huginn/runtime/task_lifecycle.py#L345-L432) 把 `oppose + oppose` 也计入冲突（L391-L394），语义上「两路都反对同一对象」应是一致。
- persona 被当作**世界真理路**：[L476-L481](file:///workspace/agent/huginn/runtime/task_lifecycle.py#L476-L481) 把 persona 与 memory/kb 放进同一个 stance 集合。身份约束与事实主张混轴，误报是结构性的。
- 规范状态不落盘：[session_state](file:///workspace/agent/huginn/session_state.py#L92-L96) 明确 `pending_confirmation` 是 working state，不上快照；[clear_confirmation](file:///workspace/agent/huginn/session_state.py#L175-L177) 只置 None。用户「接受 / 拒绝 / 授权」的结果没有对象。
- 人格修订无证据门槛：[rcb_step2.py#L1562-L1596](file:///workspace/agent/huginn/cli/rcb_step2.py#L1562-L1596) 每轮把 memory+kb 摘要 prefix 进 `adaptive_layer`，800 字符截断；无版本、无回滚、无 adoption 验证。
- 记忆矛盾只打标：[longterm.py#L1676-L1754](file:///workspace/agent/huginn/memory/longterm.py#L1676-L1754) 检出同 (formula, tag) 数值不一致后写 `contradicts:` 标签，不写 `superseded_by`，不可查询「A 被 B 取代」的路径。
- PMK 覆盖窄：`HUGINN_PMK_INJECT` 默认 `0`（[env_schema.py#L282](file:///workspace/agent/huginn/env_schema.py#L282)、[governance_audit.py#L573-L585](file:///workspace/agent/huginn/cli/governance_audit.py#L573-L585)），只在 autoloop/rcb 路径生效。

### 2.2 已有可复用底座（不要重写）

| 底座 | 位置 | 可复用部分 |
|---|---|---|
| 结论超图 | [kg/hypergraph.py](file:///workspace/agent/huginn/kg/hypergraph.py#L33-L78) | `accepted/contested/superseded/recheck` 状态机、AND/OR 前提、`condition`、`evidence_strength`、[impact_propagation](file:///workspace/agent/huginn/kg/hypergraph.py#L232)、[citation_cycles](file:///workspace/agent/huginn/kg/hypergraph.py#L324) |
| 假设图 | [autoloop/hypothesis_loop.py#L349](file:///workspace/agent/huginn/autoloop/hypothesis_loop.py#L349) | `support/refute` 强制 evidence + anchor、`refine_failed → superseded` 链 |
| 可执行经验 | [research/experience_archive.py#L43-L70](file:///workspace/agent/huginn/research/experience_archive.py#L43-L70) | 可重放、`ground_truth`、`claim_reward` 对账 |
| 治理门面 | [governance.py#L1-L23](file:///workspace/agent/huginn/governance.py#L1-L23) | `can_execute / execute / verify / rollback`、默认 deny、audit chain |
| 采纳门 | [harness/meta_improver.py#L314](file:///workspace/agent/huginn/harness/meta_improver.py#L314)、[L492](file:///workspace/agent/huginn/harness/meta_improver.py#L492) | 真实 outcome 回填、只有 GREEN 才 promote champion |
| 目标账 | [autoloop/goal_store.py#L46](file:///workspace/agent/huginn/autoloop/goal_store.py#L46) | `origin=user|self`、`pending_confirmation / rejected`、confirm/reject |
| 治理台账 | [research/harness_ledger.py#L1-L13](file:///workspace/agent/huginn/research/harness_ledger.py#L1-L13) | observed/unobserved/missing 三态证据，防「配置存在冒充能力可用」 |

---

## 3. 双螺旋定位（S1 → S6）

### S1 Discovery（已在此状态）

- 现象：记忆指标高（0.75）但 Grounding Awareness 敏感度仅 0.11–0.23；给全历史也不改善，加「助手提议」反而放大虚假共同基础。
- 结构直觉：这不是「检索不够好」，而是**规范状态没有独立表示**。把「用户提过」当成「用户接受」，是把认识论状态和规范状态压进同一个向量。
- **L1 coordinates（压缩后必须保留的不变量）**：
  ```
  动作坐标: (proposition_id, act_type, scope, status, valid_time)
  记忆坐标: (memory_id, source_class, epistemic_status, evidential_authority, valid_time)
  影响坐标: (query_id, memory_id, route, influence_authority, decided_at)
  ```
  文本、embedding、chunk、摘要都是可丢弃的路径细节；这三组坐标不可丢。注意「记忆坐标」与「影响坐标」分离：一条记忆被保留，不代表它此刻被授权影响推理。

### S2 Verification

- 反例：当前 PMK 把 persona 的「推荐」当世界主张，会与 memory 的「反对」判冲突——但身份约束与事实真假不是同一自由度。
- 反例：`oppose + oppose` 判冲突，说明判定器没有极性/作用域概念。
- 反例：`clear_confirmation()` 之后，规范状态不可恢复，任何「已授权」推断都只能靠重读聊天记录。

### S3 Switch（模式切换仪式）

- 记录坐标：`(proposition, act, scope, status, time)`。
- **丢弃路径细节**：不再优化「更像人的人格描述」「更强的冲突词表」「更大的 RAG top_k」。
- 切换注意力模式：从「生成后校验文本」切到「生成前查询授权拓扑」。
- 禁止在同一状态混用两条链：本 spec 的 S4 只做构造（schema / 状态机 / 不变量），不在构造阶段再发明新现象。

### S4 Construction（本 spec 主体）

公理、数据契约、接口、不变量见 §5–§9；记忆治理的构造见 §18。

### S5 Unification（同构点）

- 认识论冲突与规范障碍是**同一个 Čech H¹ 算子作用在两套不同 sheaf 上**：
  - 认识论 sheaf：stalk = 某路对某 subject 的立场/证据；restriction = subject 对齐；H¹ ≠ 0 → 不能断言。
  - 规范 sheaf：stalk = 某 actor 对某 scope 的授权状态；restriction = scope 包含关系；H¹ ≠ 0 → 不能执行。
- 因此不需要两套判定器；需要的是**同一个 gluing 检查 + 两张不同的图 + 两种失败语义**。
- 同构边界：这只是算子同构，不是本体同构；规范状态不可由认识论状态推出（`accept` 不蕴含 `authorize`）。

### S6 Feedback

指标、离线策略学习、线上灰度、回滚见 §11–§13。

---

## 4. 三透镜诊断

### 4.1 注意力透镜：当前 PMK 是「无 QKV、无 mask、无位置编码」的注意力

| 注意力概念 | 应有形态 | 当前 PMK |
|---|---|---|
| Query | 当前动作意图 + 目标 + persona 约束 | 只有 `attempted` 字符串 |
| Key | 账本条目 `(subject, scope, act_type/status)` | 无索引，直接向量检索 |
| Value | 证据 payload + provenance | 拼接文本前 200 字 |
| Mask | tenant/session/scope/TTL/revocation/valid-time | 无（`since` 只是时间下限） |
| Positional encoding | `valid_from` + revision + supersedes depth | 无 |
| Multi-head | 认识论头 / 规范头 / 身份头 | 一个混合 stance 集合 |
| 输出 | 紧凑 `AdmissionView`（top-k） | 长文本塞 prompt |

设计结论：**不要把三路文本拼在一起让 LLM 自己对齐**；把 Query/Key/Value 分开，把 mask 做成硬约束，把多头做成三个独立 gate。复杂度按「稀疏检索 + 小 rerank」控制，热路径不调 LLM（IO 优先原则）。

### 4.2 高阶网络透镜：超图而非单纯复形；两张 sheaf 而非一张

- **规范行为是多元关系**：`(actor, act_type, proposition, scope, authorizer)` 是 n 元超边。若三方授权成立，不意味着任意两方授权成立——**不满足向下闭包**，所以用超图，不用单纯复形。
- **认识论结论是合取依赖**：结论 C 需要前提 {A,B,D} 同时成立；子集 {A,B} 不构成独立结论——同样不满足向下闭包，现有 [ClaimHypergraph](file:///workspace/agent/huginn/kg/hypergraph.py#L1-L19) 的选择正确。
- **不使用 SuperHyperGraph**：两级超图（超边挂超边）足够；递归幂集无已发表应用优势，只增加复杂度。
- **H¹ 是代理不是定理**：现有 `cycle_basis` 是 β₁ 近似，不是真正持久上同调；本 spec 保留该近似，不上 gudhi/ripser，除非出现「环检测漏报」的实测证据。
- **拓扑许可动力学**：若规范图里不存在 `authorize(scope)` 到动作节点的边，则该动作模式在结构上不可达。执行门不是分类器，是**切口**：删边比调阈值可靠。
- **Hodge 诊断（P2）**：把「动作压力」分为梯度（目标驱动）、旋度（习惯/循环）、调和（拓扑环：A 授权 B、B 授权 A，或 PMK 冲突反复重定向）。`iteration_history` 的 3/5 redirect 是调和分量的现成代理；调和分量高时，正确操作是**加一条割边**（显式决策/撤销），不是再暂停一次。

### 4.3 双螺旋透镜：哪些是坐标，哪些是路径

- 坐标：`proposition_id / act_type / scope / status / valid_time`。
- 路径（可替换）：JSONL vs SQLite、token 重合 vs embedding、规则 vs LLM 分类器、bandit vs RL。
- 构造阶段必须「折磨每一块砖」：每个不变量都要有可执行测试；每个默认值都要有治理锚点；每个新增表都要有回放测试。

---

## 5. 设计公理（可反驳）

| # | 公理 | 反驳条件 |
|---|---|---|
| A1 | 认识论状态与规范状态必须分轴 | 若存在单一轴同时预测「真假」与「允许」且不产生系统性误报，则 A1 可被削弱 |
| A2 | `propose ≠ accept ≠ authorize ≠ commit` | 若合并四者后 Grounding 敏感度不降，则 A2 可简化 |
| A3 | 规范状态只能由显式行为事件写入，不能由文本相似度推断 | 若纯推断方案在 held-out 上达到同等敏感度且零未授权执行，则 A3 可放宽 |
| A4 | 执行门 fail-closed；检索 fail-open | 若 fail-closed 导致不可接受的 false pause，需给出替代的降级策略而非直接放开 |
| A5 | 人格修订必须过真实 outcome 门，不能过摘要 | 若摘要式修订在行为稳定性指标上不劣于参数化修订，则 A5 可被推翻 |
| A6 | 热路径不调 LLM；LLM 只做离线/缓存分类 | 若 LLM 在 p99 延迟与确定性上满足门限，可局部放宽 |

---

## 6. 架构总览

```
                    ┌──────────────────────────────┐
                    │  System 3 (admission kernel) │
                    │  纯函数，无 LLM，O(k)         │
                    └───────┬──────────┬───────────┘
                            │          │
        ┌───────────────────┘          └───────────────────┐
        ▼                                                  ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│ 规范账          │   │ 认知账          │   │ 经验账              │
│ NormativeLedger │   │ BeliefLedger    │   │ ProceduralLedger    │
│ propose/accept/ │   │ claim/hypothesis│   │ experience_archive  │
│ authorize/...   │   │ supersedes 链   │   │ failed_direction    │
└────────┬────────┘   └────────┬────────┘   └──────────┬──────────┘
         │                     │                       │
         │        ┌────────────┴────────────┐          │
         │        ▼                         ▼          │
         │  ┌───────────────┐        ┌───────────────┐  │
         │  │ 保留闸         │        │ 影响闸         │  │
         │  │ RetentionGate │        │ InfluenceGate │  │
         │  │ 可用/归档/巩固 │        │ 候选→逐条卫生  │  │
         │  │ /修剪          │        │ →授权/隔离/阻断│  │
         │  └───────┬───────┘        └───────┬───────┘  │
         │          │                        │          │
         │          └──────────┬─────────────┘          │
         │                     ▼                        │
         │            ┌──────────────────┐              │
         │            │ 梦境周期（离线）  │              │
         │            │ SleepClock       │              │
         │            │ Dreamer/Analyst/ │              │
         │            │ Consolidator     │              │
         │            │ 写入侧晋升门      │              │
         │            └────────┬─────────┘              │
         └─────────────────────┴────────────────────────┘
                               ▼
                     ┌─────────────────┐
                     │ PMK 双 sheaf    │
                     │ 认识论 H¹ / 规范 H¹│
                     └─────────────────┘
```

硬规则：

1. **规范账只追加**；当前状态是事件日志的确定性投影，不是可变字段。
2. **认知账复用现有图**；只补 `subject_scope / owner / provenance / supersedes`。
3. **经验账只存可复现单元**；不可复现的文本摘要不进入该账。
4. **记忆有两条独立闸**：保留（是否还可用）与影响（此刻是否进入推理上下文）分开判定、分开审计；检索返回候选，不是上下文。
5. **梦境写入是第四条状态迁移**：必须过写入侧晋升门，不能与在线直接证据同权。详见 §18。

---

## 7. 数据契约

### 7.1 规范账（新增）

```sql
CREATE TABLE normative_events (
  act_id           TEXT PRIMARY KEY,
  tenant_id        TEXT NOT NULL,
  deployment_id    TEXT NOT NULL,
  device_id        TEXT NOT NULL,
  session_id       TEXT NOT NULL,
  seq              INTEGER NOT NULL,          -- per-session Lamport
  actor            TEXT NOT NULL,             -- user | agent | system
  act_type         TEXT NOT NULL,             -- propose|acknowledge|accept|reject|authorize|commit|revoke|withdraw
  proposition_id   TEXT NOT NULL,             -- 内容寻址稳定 id
  proposition_digest TEXT NOT NULL,
  scope_json       TEXT NOT NULL DEFAULT '{}',-- {actions:[],resources:[],limits:{}}
  ttl_s            INTEGER,
  supersedes       TEXT,                      -- 被本事件取代的 act_id
  evidence_json    TEXT NOT NULL,             -- {turn_id,message_id,span,quote_digest}
  created_at       TEXT NOT NULL,
  UNIQUE(tenant_id, session_id, seq)
);
CREATE INDEX idx_norm_prop ON normative_events(tenant_id, proposition_id, created_at);
CREATE INDEX idx_norm_scope ON normative_events(tenant_id, act_type, created_at);
```

事件语义：

| act_type | 谁能写 | 含义 | 不蕴含 |
|---|---|---|---|
| `propose` | agent | 提出方案/解释/动作 | 不被接受 |
| `acknowledge` | user | 提及/考虑，明确弱于接受 | 不是接受 |
| `accept` | user | 接受命题为真/偏好 | **不是授权执行** |
| `reject` | user | 拒绝命题 | 不撤销既往授权 |
| `authorize` | user/system | 授予 scope 内权限，可带 TTL | 不是对命题真假的判断 |
| `commit` | agent | 承诺未来动作 | 不是已获授权 |
| `revoke` | user/system | 撤销某授权/承诺/接受 | 不删除历史 |
| `withdraw` | agent | 撤回自己的提议 | 不影响用户已接受部分 |

硬规则：

- `actor=agent` 不能写 `authorize / revoke / accept / reject`；身份由运行时认证，不信任消息内容。
- `revoke` 在投影中**支配**同 actor、同 scope 的更早 `authorize`。
- TTL 在**读取时**按服务端时间判定，不写入过期事件。
- scope 包含关系用结构判定（集合包含 + 限额比较），不用语义相似度。

投影（`GroundingView`）：

```python
@dataclass
class GroundingView:
    open_proposals: list[ActRef]
    accepted: list[ActRef]          # 接受但不授权
    authorizations: list[ActRef]     # active + 未过期 + 未撤销
    commitments: list[ActRef]
    rejected: list[ActRef]
    revoked: list[ActRef]
    def covers(self, scope: dict) -> bool: ...
```

### 7.2 认知账（扩展）

复用 `ClaimHypergraph` / `HypothesisGraph`，新增字段：

```python
subject_scope: Literal["user", "science", "self"]
owner_id: str | None
provenance: list[str]          # memory_id / chunk_id / doi / run_id
valid_from: str
valid_to: str | None
supersedes: str | None
superseded_by: str | None
```

用户信念写入路径：

- `MemoryType.USER / FEEDBACK` 写入时同步产出一条 `BeliefRecord`（原子信念，无 premises）。
- 现有 `contradicts:` 标签改为：写 `superseded_by` 指针，保留双份记录，检索按 `valid_from` 排序（GROWL 的「只合并索引、不合并数据」最小版）。
- 科学结论继续走 `ClaimHypergraph`（AND/OR 前提、影响传播）；不要给用户信念强加科学前提结构。

### 7.3 经验账（扩展）

复用 [ExecutableExperienceArchive](file:///workspace/agent/huginn/research/experience_archive.py#L43-L70)，新增行为指纹：

```python
@dataclass(frozen=True)
class BehaviorFingerprint:
    persona_id: str
    window_start: str
    window_end: str
    tool_family_dist: dict[str, float]
    clarification_rate: float
    evidence_gate_pass_rate: float
    risk_level_hist: dict[str, int]
    authorization_boundary_hits: int
```

用途：跨会话比较同一 persona 的行为分布（JS 散度），替代「人格描述是否一致」这种不可测判据。

---

## 8. System 3：准入调度器

```python
@dataclass
class AdmissionRequest:
    tenant_id: str
    session_id: str
    action_name: str
    risk: RiskLevel
    resources: list[str]
    persona_id: str
    claim_ids: list[str]
    proposed_by: str            # agent | user

@dataclass
class AdmissionDecision:
    allow: bool
    mode: str                   # allow | ask | deny | degrade
    reasons: list[str]
    required_acts: list[str]    # 缺失的最小规范行为，如 ["authorize:write:workspace/x"]
    cited_act_ids: list[str]
    cited_claim_ids: list[str]
```

判定优先级（纯函数、确定性）：

1. **硬拒**：跨租户 / 已撤销 / 已过期 / 未覆盖 scope / 触碰 persona 内核安全约束。
2. **降级**：依赖的 claim 处于 `contested/recheck` → 只允许只读或要求先补证据。
3. **升级提问**：缺少最小规范行为 → `ask`，并给出**具体的 required_acts**，不生成泛化问题。
4. **放行**：scope 覆盖且 claim 不冲突。

硬约束：

- 不调用 LLM。
- 不读聊天全文，只读投影后的 `GroundingView` 与 `ClaimView`。
- 相同输入必得相同输出（可回放）。
- `required_acts` 必须是机器可执行的 act 描述，不允许自然语言泛问。

---

## 9. PMK 重构：两张 sheaf，两种失败语义

### 9.1 重新分路

| 路 | 归属 | 冲突语义 |
|---|---|---|
| persona | **身份/规范约束**，不是世界真理 | 与动作 scope 冲突 → 规范障碍（阻断） |
| memory | 认识论（含用户信念、情节、失败方向） | 与 kb/timeseries 冲突 → 认识论障碍（降级/要求证据） |
| knowledge | 认识论（文献、原则、演化规则） | 同上 |
| timeseries | 认识论（系统动力学） | 同上 |
| 规范账 | 规范（授权、承诺、撤销） | 与动作冲突 → 规范障碍（阻断） |

### 9.2 两个 H¹

```python
@dataclass
class Obstruction:
    kind: Literal["epistemic", "normative"]
    sources: list[str]
    subject: str
    beta1: int
    reason: str
    refs: list[str]        # 可寻址 id：memory_id / chunk_id / act_id / claim_id

def glue(stalks: dict[str, Stance], restriction: Callable) -> tuple[int, list[Obstruction]]: ...
```

- 认识论 H¹ ≠ 0：**不阻断执行**，把相关 claim 标 `contested`，在最终报告里禁止作为确定结论；要求补证据或降级措辞。
- 规范 H¹ ≠ 0：**阻断执行**，返回最小 `required_acts`。
- 两张图分别建边；禁止把 persona 放进认识论 stalk 集合（修掉当前 L476-L481 的混轴）。

### 9.3 QKV 化

```python
@dataclass
class AdmissionQuery:
    intent: str                 # 当前动作
    goal: str
    persona_params: dict        # 只取约束类参数

@dataclass
class LedgerKey:
    subject: str
    scope: dict
    kind: str                   # act_type / claim_status
    valid_time: str

@dataclass
class LedgerValue:
    text: str
    provenance: list[str]
    status: str
    confidence: float
```

- 检索：先按 mask（tenant/session/scope/TTL/revocation/valid-time）过滤，再按 subject 索引取候选，最后小 rerank。
- 输出：`AdmissionView(top_k)`，不把原文塞 prompt。
- 多头：认识论头 / 规范头 / 身份头各自 top-k、各自 gate。
- 位置编码：`valid_from` + `revision` + `supersedes_depth`。
- 成本目标：p99 ≤ 20 ms，O(log n) 索引 + O(k) rerank，热路径零 LLM。

### 9.4 默认开启与治理

- `HUGINN_PMK_INJECT` 默认 `0 → 1`。
- 新增 `HUGINN_PMK_MODE ∈ {off, shadow, enforce}`，默认 `enforce`。
- `off` 仅作故障开关，不再是默认值。
- 必须同步修改：
  - [env_schema.py#L282](file:///workspace/agent/huginn/env_schema.py#L282)
  - [governance_audit.py#L573-L585](file:///workspace/agent/huginn/cli/governance_audit.py#L573-L585) 的 `default: "0" → "1"`
  - 重新生成 [governance-state.md](file:///workspace/agent/docs/governance-state.md#L124)
  - `tests/test_arch_gate_registry.py` 的能力锚点期望
  - `tests/test_temporal_pmk.py` 的默认值断言
- 默认翻转必须留痕：CHANGELOG + 治理台账 + 回滚说明。

---

## 10. Identity Kernel：人格参数化与修订日志

### 10.1 结构

```python
@dataclass(frozen=True)
class PersonaKernel:
    persona_id: str
    role: str
    safety_constraints: tuple[str, ...]
    authorization_boundaries: tuple[str, ...]
    core_digest: str

@dataclass
class PersonaParams:
    clarification_threshold: float = 0.5
    evidence_strictness: float = 0.7
    risk_tolerance: float = 0.3
    exploration_rate: float = 0.2
    verbosity: float = 0.5
    skepticism: float = 0.5
    tool_bias: dict[str, float] = field(default_factory=dict)

@dataclass
class PersonaRevision:
    revision_id: str
    base_version: int
    delta: dict                    # 只允许 PersonaParams 字段
    evidence_ids: list[str]        # claim_id / experience_id / act_id
    candidate_since: str
    status: Literal["candidate", "active", "superseded", "rejected"]
    eval_report: dict              # adoption gate 输出
```

### 10.2 修订规则

- `PersonaKernel` 不可变；任何修订不得触碰安全约束与授权边界。
- `PersonaParams` 只能由 revision 改，不能直接赋值。
- `adaptive_layer` 降级为**只读投影**：由 `params + 当前证据摘要` 渲染，不再 prefix 累加、不再作为真相源。
- 晋升条件（全部满足）：
  1. ≥ N 条**跨会话独立**证据（默认 N=3）。
  2. 真实 outcome 改善（harness / claim_reward），且统计显著。
  3. OOD 不退化。
  4. 安全与授权边界回归测试通过。
  5. 可回滚（旧 revision 保留）。
- 复用 [meta_improver](file:///workspace/agent/huginn/harness/meta_improver.py#L492) 的 GREEN 采纳门；不要新造一套统计。
- 失败路径：`rejected` 也落盘，附理由；下次候选先查重（复用 `reflection._write_rejection` 的思路）。

---

## 11. 学习回路（有大规模轨迹，纳入范围）

### 11.1 轨迹索引

每条准入决策与规范事件、执行结果、claim/harness 结果 join 成一条训练样本：

```python
@dataclass
class AdmissionSample:
    query: AdmissionQuery
    context_features: dict       # risk, scope, claim_status, persona_params, history
    decision: AdmissionDecision
    outcome: dict                # unauthorized? false_pause? harness dims? claim_reward
    tenant_id: str
    ts: str
```

### 11.2 三个可学习组件

| 组件 | 任务 | 方法 | 评估 |
|---|---|---|---|
| ActClassifier | utterance → `act_type + scope` | 监督分类（长尾处理 + 置信度校准） | 敏感度 / 特异度 / 校准误差 |
| AdmissionPolicy | 风险 × 上下文 → `ask / auto-allow / degrade` | 上下文 bandit（复用 meta_improver 臂账本），数据足够后转离线 RL | regret、false pause、未授权执行 |
| MemoryOpPolicy | 何时 store / retrieve / summarize / discard | 离线 RL + 可重放经验 | grounded accuracy / token 成本 |

硬约束作为 **action mask**，不进入 reward：授权不存在时动作不可选；revoke 后动作不可选；跨租户动作不可选。这样避免「学会绕过门」。

### 11.3 离线评估

- 按 tenant + 时间切分，禁止同会话泄漏。
- 用事件日志做 counterfactual replay：同一状态换策略，比较决策与真实 outcome。
- 先 OPE，再影子模式，再灰度；禁止直接上线 RL 策略。
- 红队用例必须包含：「助手提议被误当作用户接受」「用户提及被误当授权」「撤销后仍执行」。

---

## 12. 不变量（每条对应可执行测试）

| # | 不变量 | 测试 |
|---|---|---|
| I1 | 无 active authorization 覆盖 scope 时，写/执行/发布动作必须 deny | `test_admission_gate.py::test_fail_closed` |
| I2 | `propose` 不蕴含 `accept`；`accept` 不蕴含 `authorize` | `test_grounding_awareness.py::test_act_separation` |
| I3 | 同 actor 同 scope 的 `revoke` 支配更早 `authorize` | `test_normative_ledger.py::test_revoke_dominance` |
| I4 | 投影是事件日志的纯函数，回放结果逐字节一致 | `test_normative_ledger.py::test_replay_determinism` |
| I5 | `PersonaKernel` 不可变；active revision 必须有 ≥N 证据 + 通过采纳门 | `test_persona_revision.py` |
| I6 | 认识论冲突必须引用 ≥2 个可寻址节点；纯文本冲突只作 advisory | `test_pmk_sheaves.py::test_addressable_conflict` |
| I7 | 记忆被取代时保留双份；检索按 valid-time 排序 | `test_belief_ledger.py::test_supersede_keeps_both` |
| I8 | System 3 决策确定性、无 LLM | `test_system3_scheduler.py::test_deterministic_no_llm` |
| I9 | 跨租户零泄漏 | `test_normative_ledger.py::test_tenant_isolation` |
| I10 | 准入热路径 p99 ≤ 20 ms，且不产生 LLM 调用 | `test_admission_gate.py::test_latency_no_llm` |
| I11 | `actor=agent` 不得写 `authorize/revoke/accept/reject` | `test_normative_ledger.py::test_actor_capability` |
| I12 | 撤销/过期授权不得出现在后续 `AdmissionView` | `test_grounding_awareness.py::test_mask_revoked` |

记忆治理不变量 I13–I24 见 §18.8；两表共同构成完整不变量集。

---

## 13. 多部署与同步

- 所有账本按 `tenant_id` 分区；`deployment_id / device_id` 只用于溯源，不参与授权判定。
- 事件追加式；同会话用 Lamport `seq` 排序；跨设备晚到事件触发**重投影**，不修改历史。
- 收敛规则：
  - `revoke` 支配 `authorize`（同 actor + 同 scope）。
  - 同 `proposition_id` 多个 `accept/reject`：以最新 `valid_time` 为准，但保留全部事件。
  - TTL 用服务端时间读取时判定，避免设备时钟偏斜。
- 隐私：`evidence_json` 只存 `message_id + span + quote_digest`；原文走既有加密存储与保留策略，不复制到规范账。
- 梦境同步：`DreamObject` 带 `tenant_id / deployment_id`，跨设备只同步投影后的对象与权威向量，不同步原始对话；Soul 层级按租户隔离，禁止 cross-tenant seed。
- 迁移：SQLite WAL + 原子迁移脚本；不回填历史聊天为规范事件（避免伪造共同基础），只从新会话开始记录；记忆治理事件从启用时刻开始记账，不回填旧记忆的伪影响决策。

---

## 14. 指标与验收

### 14.1 离线（held-out）

| 指标 | 目标 | 对照 |
|---|---|---|
| Grounding sensitivity（accept vs propose） | ≥ 0.85 | 论文基线 0.11–0.23 |
| Grounding specificity | ≥ 0.85 | 同上 |
| 未授权执行率（enforce） | 0 | — |
| 规范行为分类校准误差（ECE） | ≤ 0.05 | — |
| PMK 冲突 precision / recall | ≥ 0.80 / ≥ 0.70 | 当前无标注集，需先建 |
| Persona revision 可追溯率 | 100% | 当前 0% |
| 回放确定性 | 100% | — |

### 14.2 线上

| 指标 | 目标 |
|---|---|
| false pause rate | ≤ 5% |
| admission p99 | ≤ 20 ms |
| 跨会话行为指纹 JS 散度 | ≤ 阈值（按 persona 基线定） |
| 重复 PMK 冲突率（3/5 redirect） | 相对基线下降 ≥ 50% |
| 租户隔离事件 | 0 |

### 14.3 Definition of Done

1. 三本账 schema 落地，I1–I24 全绿（其中 I13–I24 见 §18.8）。
2. PMK 默认开启，治理锚点与台账同步，无默认翻转漂移。
3. `session_state` 确认结果落盘；`clear_confirmation` 不再丢规范状态。
4. persona revision 走 GREEN 采纳门；`adaptive_layer` 变为只读投影。
5. 记忆 `contradicts:` 升级为 `superseded_by`，双份保留、按 valid-time 检索。
6. System 3 接入 autoloop 的 pause 判定之前，determinism 测试通过。
7. 离线策略学习完成 OPE 报告，影子模式无未授权执行。
8. 现有 arch 门、PMK 测试、治理台账全部绿。
9. 保留闸与影响闸独立实现，Hygiene Gate 七种 route 全量留痕（I13–I17）。
10. 梦境周期有 SleepClock、DreamObject、写入侧晋升门；`derived/hypothesis` 不授权动作、不改内核（I15、I19、I21、I23）。
11. Soul 四层优先级全序可审计，跨租户记忆/梦境零泄漏（I18、I22）。

---

## 15. 风险与诚实边界

| 风险 | 处理 |
|---|---|
| Act 分类错误导致误授权 | 只对**显式行为**授权；模糊 utterance 一律 `ask`，不从文本推断；分类器只用于建议，不用于放行 |
| 过度阻断（false pause） | 返回最小 `required_acts`，让用户一次点击完成授权；监控 false pause，超阈值降级到 `shadow` 但不改默认 |
| 账本膨胀 | 按 tenant/session 分区、TTL、归档；证据只存 digest + 引用 |
| 设备时钟偏斜 | 服务端时间判定 TTL；Lamport seq 排序；晚到事件重投影 |
| 学习策略奖励作弊 | 硬约束做 action mask，不进 reward；OOD + 红队；未授权执行是硬失败不是低奖励 |
| H¹ 近似失真 | 明确声明 `cycle_basis` 是 β₁ 代理，不是持久上同调；只有实测漏报才升级算法 |
| 潜空间身份无法保证 | 不做；本 spec 的「身份」是运行时可追究的约束集，不是模型内部表示。若行为指纹仍漂移，视为 A0 主张被证伪 |

明确不做：

- 不做 latent manifold / persona 向量微调（外部模型 API，不可控）。
- 不做 SuperHyperGraph / 递归幂集。
- 不做「更聪明的文本冲突判官」；文本只做 advisory。
- 不用 LLM 做准入判定（只做离线/缓存分类）。
- 不把规范状态写进 `adaptive_layer` 文本。
- 不把「检索命中」当作「已授权进入上下文」；影响闸与保留闸必须分开。
- 不给梦境产物自动权威加成；离线反思只提高抽象权威，不提高证据权威。

---

## 16. 分期路线

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| P0-a | 修语义：删 `oppose+oppose` 冲突；persona 移出认识论站位；接 `check_pmk_consistency_llm`（仅 advisory） | PMK 现有测试绿 + 新增误报回归 |
| P0-b | 规范账最小闭环：事件表 + 投影 + `admit()` + `session_state` 落盘 | I1–I4、I9–I12 绿 |
| P0-c | 默认开启与治理同步：`HUGINN_PMK_INJECT=1`、`PMK_MODE=enforce`、锚点更新 | 治理台账无漂移 |
| P1-a | 认知账扩展：`subject_scope/owner/provenance/supersedes`；用户信念写入 & 取代链 | I6–I7 绿 |
| P1-b | PMK 双 sheaf + QKV `AdmissionView`；persona 参数化 + revision log | I5、I8 绿；p99 达标 |
| P2-a | System 3 接入 pause 前；Hodge 调和分量诊断 | 重复冲突率下降 |
| P2-b | 轨迹索引 + ActClassifier + AdmissionPolicy 离线训练 + OPE | 敏感度/特异度达标；影子零未授权 |
| P3 | 灰度 enforce 全路径；多设备同步；行为指纹监控 | 线上指标达标 |

记忆治理补章（§18）的分期 P1-c / P1-d / P2-c / P2-d / P3-b 与本表并行推进，阶段号一一对应；出口以 §18.8 的 I13–I24 与 §18.9 的指标为准。

---

## 17. 附：三条给实施者的提醒（双螺旋压缩）

1. **路径可丢，坐标不可丢**：实现可以换表、换算法、换分类器；三组坐标不能丢——动作 `(proposition, act, scope, status, time)`、记忆 `(memory_id, source_class, epistemic_status, evidential_authority, time)`、影响 `(query_id, memory_id, route, influence_authority, decided_at)`。
2. **先建切口，再谈学习**：没有授权拓扑的硬边，任何策略学习都会学会绕过；先让未授权动作在结构上不可达。
3. **反例即结构在说话**：`oppose+oppose` 误报、`clear_confirmation` 丢状态、`adaptive_layer` 前缀漂移——这三个反例不是 bug 清单，是「混轴」这一个结构问题的三个投影。

---

## 18. 记忆治理补章：保留 ≠ 影响，梦境 ≠ 证据

本节回应 DREAM v3 / Governed Memory / Anamnesis / AutoPersonas OSO / MemDream / Auto-Dreamer / Sleep-Gated Dreaming / Soul-Conditioned Dreaming 的共同主张，并把它们并入同一套账本，而不是新增一套并行框架。

三透镜落点：

- **注意力**：Hygiene Gate = 逐查询的动态权重（七维打分）+ 硬 mask（权威类别、租户、TTL、撤销、Soul）；影响预算 = top-k 的稀疏聚合，不做全量上下文。
- **高阶网络**：保留与影响是**两类不同的边**（同一记忆可同时属于“可留”与“禁入”）；Soul 四层是层级超边，不是可加的标量权重；梦境修复改变的是图结构，不是文本。
- **双螺旋**：记忆坐标与影响坐标分离；表、算法、分类器都是可替换路径，坐标不可丢；先建切口，再谈学习。

### 18.1 第一原则：记住一件事，不等于授权它此刻塑造推理

把记忆生命周期拆成两个**独立决策**，二者不共享阈值、不共享审计行、不共享默认值：

| 决策 | 问题 | 时机 | 输出 | 失败后果 |
|---|---|---|---|---|
| 保留闸 `RetentionGate` | 这条记忆还值不值得留？ | 写入 / 周期维护 / 睡眠 | `keep_available / archive / consolidate / prune` | 删错证据、存储膨胀 |
| 影响闸 `InfluenceGate` | 此刻是否、以多大权威进入推理？ | 每次查询/每步动作前 | `authorize / uncertain / low_impact / arbitrate / quarantine / block` | 上下文污染、错误归因、未授权行为 |

硬规则：

- 检索返回的是**候选者（candidate）**，不是已授权上下文。
- 保留决策不能提升影响权威；影响决策不能删除记忆。
- 一条被 `archive` 的记忆仍可被影响闸以低权威召回（用于审计/反例），反之一条 `keep_available` 的记忆也可以被影响闸 `block`。
- 两个决策各自写一条 `memory_state_event`（§18.7），审计时可独立追责。

接口：

```python
@dataclass
class RetentionDecision:
    memory_id: str
    action: Literal["keep_available", "archive", "consolidate", "prune"]
    policy_id: str
    policy_version: str
    reasons: list[str]

@dataclass
class InfluenceDecision:
    query_id: str
    memory_id: str
    route: Literal["authorize", "uncertain", "low_impact",
                   "arbitrate", "quarantine", "consolidate", "block"]
    influence_authority: float          # [0,1]，逐查询计算
    caps_applied: list[str]
    reasons: list[str]
```

### 18.2 Hygiene Gate：状态感知的逐条卫生闸

对每个候选记忆计算七个维度，并允许两个额外硬信号（权威类别、灵魂显著性）参与 clamp：

| 维度 | 信号来源（现有可复用） | 默认处置 |
|---|---|---|
| 置信度 confidence | `memory.importance`、claim `confidence`、`evidence_strength` | 低 → `uncertain` |
| 不确定性 uncertainty | `epistemic.detect_epistemic_gap`、belief entropy、缺失字段比例 | 高 → `uncertain` / `low_impact` |
| 矛盾压力 contradiction_pressure | `longterm.scan_contradictions`、ClaimHypergraph `contested/recheck`、PMK H¹ | 高 → `arbitrate`，不静默选边 |
| 来源可靠性 source_reliability | `provenance` source class、`claim_reward`、`harness_ledger` observed 比例 | 低 → `low_impact` / `quarantine` |
| 时间有效性 temporal_validity | `valid_from/valid_to`、TTL、`since` | 过期 → `block` |
| 接地状态 grounding_status | `claim_grounding`（数值是否落在工具轨迹）、`experience_archive.verdict` | 未接地 → `uncertain`，不得当事实 |
| 溯源完整性 provenance_completeness | `memory.anchor`、`provenance` 链、`quote_digest` | 缺失 + 高风险 → `block` / `quarantine` |
| （硬）权威类别 authority_class | `direct / derived / hypothesis`（§18.6） | 决定影响上限，不可被分数覆盖 |
| （硬）灵魂显著性 soul_salience | `SoulContext / SalienceProfile`（§18.5） | 只调制种子与边界，不绕过授权 |

评分（可配置、可回放）：

```python
base = (w_rel * relevance
        + w_conf * confidence
        + w_ground * grounding_status
        + w_prov * provenance_completeness
        + w_rec * recency_validity)
penalty = w_unc * uncertainty + w_con * contradiction_pressure
raw = clamp(base - penalty, 0.0, 1.0)
influence = min(raw, cap(authority_class, route))
```

`cap` 默认：

| 权威类别 / 路由 | influence 上限 | 能否作为事实依据 | 能否授权动作 |
|---|---|---|---|
| direct + authorize | 1.0 | 是 | 是（仍需规范账 scope） |
| derived + authorize | 0.6 | 仅作背景/框架 | 否 |
| hypothesis/dream + uncertain | 0.4 | 否 | 否 |
| quarantine / block | 0.0 | 否 | 否 |
| revoked / expired | 0.0 | 否 | 否 |

上下文装配规则：

- 按 `influence` 排序取 top-k，而不是按相似度取 top-k；相似度只作为 `relevance` 的一项。
- 注入时必须带来源标签（`[DIRECT] / [DERIVED] / [DREAM] / [UNCERTAIN]`），禁止无标签注入。
- 总影响预算：`sum(influence_i) <= context_budget`；预算不足时优先保留 direct，再保留高显著性的 derived。
- 若所有候选都被 `block/quarantine`，必须显式输出「无可用证据」，不得用通用知识补位冒充记忆。

实现落点：新增 `huginn/memory/hygiene_gate.py`（纯函数）；阈值走现有 `plugins.memory_maintenance_policy` 策略注册表，不在代码里硬编码；`memory/decay.py` 只保留保留闸的衰减逻辑，不再直接决定召回。

### 18.3 Governed Memory：每一次状态迁移都要过闸并留痕

Anamnesis 的生命周期契约落成一条规则：**写入、索引、巩固、召回、修订、遗忘、再训练、晋升**八类迁移，全部产生 `memory_state_event`，带 `policy_id + policy_version + decision + authority + evidence`。

迁移与闸的对应：

| 迁移 | 闸 | 不可省略的证据 |
|---|---|---|
| ingest / index | 保留闸 | 来源、时间、provenance digest |
| recall | 影响闸 | query_id、七维评分、route |
| consolidate | 保留闸 + 写入侧晋升门 | 源 id 列表、压缩比、可还原路径 |
| revise / supersede | 认知账 + 保留闸 | 新旧 id、矛盾证据、`superseded_by` |
| forget（软） | 保留闸 | 衰减系数、可恢复句柄 |
| prune（硬） | 保留闸 + 双信号 | 低权威 + 预测失败/过期；不可恢复需二次确认 |
| retrain | 治理门 | 训练数据快照、策略版本、action mask |
| promote（梦境/派生→更高权威） | 写入侧晋升门 | 独立佐证、provenance 链、无未解矛盾 |

权威链（Anamnesis）：

```python
@dataclass(frozen=True)
class AuthorityLink:
    node_id: str
    source_class: Literal["direct", "derived", "hypothesis"]
    directness: float              # 0..1
    corroboration: int
    provenance_digest: str
    epistemic_status: str
    parent: str | None             # 指向被派生来源

# 修订 = 追加新 link，不覆盖旧 link；删除 = tombstone，不物理销毁权威链
```

### 18.4 梦境周期：离线诊断、可学习修复、可逆遗忘

把 [MemoryMaintainer](file:///workspace/agent/huginn/memory/maintainer.py#L20-L102) 从「定时衰减线程」升级为受治理的梦境周期，但**不新增 agent 框架**：Dreamer / Analyst / Consolidator 是同一周期里的三个角色。

| 角色 | 职责 | 输入 | 输出 |
|---|---|---|---|
| Dreamer | 探测记忆图的潜在断点 | 孤立节点、低连接簇、陈旧 anchor、矛盾压力、检索探针失败 | `DreamFinding` 列表 |
| Analyst | 诊断失败类型 | findings + 证据 | `Diagnosis`（缺链/重复/过时/过度泛化/冲突） |
| Consolidator | 执行修复 | diagnosis | `RepairAction`（add_link / split_cluster / merge_duplicate / re_anchor / soft_decay / quarantine / summarize） |

约束：

- 修复动作**不得删除 direct 证据**；只能调整索引、链接、显著性、摘要。
- 软衰减可逆：保存衰减系数与恢复句柄；同一预判信号驱动衰减与恢复。
- 修复策略可学习（MemDream 的 GRPO 路线），但必须离线训练 + OPE + 影子模式；reward = held-out 检索改善 − 成本，不能用 LLM 自评。
- Auto-Dreamer 的快慢解耦：会话内快速写入不经过梦境；跨会话巩固才进入睡眠周期，二者用不同策略版本。
- 巩固是可学习能力，不是固定启发式；现有 [cluster.py](file:///workspace/agent/huginn/memory/cluster.py#L36) 的 LLM 摘要作为 Consolidator 的基线，不是终点。

SleepClock：

```python
@dataclass
class SleepClock:
    persona_id: str
    timezone: str
    weekly_windows: list[tuple[str, str, str]]  # (weekday, start, end)
    state: Literal["active", "sleep"]
    last_transition_at: str

def tick(clock: SleepClock, now_server: str) -> SleepClock: ...
```

- 睡眠状态是**后台巩固的准入控制**，不是用户请求的准入控制。
- 活跃状态下不得跑巩固；触发则排队或 deny（policy 决定），并记录 `sleep_leakage` 指标。
- 梦产物在下一轮以 `[DREAM]` 标签注入，受影响闸上限约束；不得作为事实依据。

梦境对象：

```python
@dataclass
class DreamObject:
    dream_id: str
    persona_id: str
    sleep_window: tuple[str, str]
    seed_event_ids: list[str]
    highlights: list[str]
    decisions: list[str]
    risks: list[str]
    followups: list[str]
    affect: dict[str, float]
    abstraction_authority: float
    evidential_authority: float
    influence_authority: float
    provenance: list[str]
    soul_priority: str
    status: Literal["candidate", "promoted", "quarantined", "expired"]
```

Soul-Conditioned：种子选择、边界厚度、Langevin 漂移、EFE 分流由 `SoulContext + SalienceProfile` 调制；禁止全局一套配置。人格不同，梦境分布就应不同，这是可检验的（见 §18.9 E4）。

### 18.5 Soul 层级：价值排序，不是授权排序

四层优先级：

```
TenantSoul (4) ≻ OrgUnitSoul (3) ≻ AgentSoul (2) ≻ UserSoul (1)
```

适用范围：

- **适用**：梦境种子选择、显著性、边界厚度、漂移强度、EFE 分流、价值冲突裁决。
- **不适用**：事实真假（认识论）、动作授权（规范账）。`UserSoul` 优先级最低，不代表可以绕过用户授权；用户同意仍由规范账独立判定。
- 高优先级只能**约束**低优先级，不能替它写入 `authorize`；低优先级可以增加约束，不能覆盖高优先级。
- 每次覆盖/冲突写 `soul_resolution_event`，带 `higher / lower / dimension / decision / reason`。
- 多租户隔离：`TenantSoul` 不可被 org/agent/user 覆盖；跨租户记忆不得进入梦境种子池。

### 18.6 梦境产物的写入侧治理：回答「更高权威还是更低权威」

结论：**不是更高或更低，而是两个轴分开。**

```python
@dataclass(frozen=True)
class AuthorityVector:
    abstraction_authority: float   # 能提供多抽象的框架
    evidential_authority: float    # 能作为多强的证据
    influence_authority: float     # 此刻能多大程度塑造推理（逐查询）
```

默认配置：

| 来源类别 | 例子 | abstraction | evidential | influence cap | 可授权动作 |
|---|---|---|---|---|---|
| `direct` | 工具结果、用户显式行为、已验证实验 | 0.2 | 0.9 | 1.0 | 是（另需 scope） |
| `derived` | 聚类摘要、巩固产物、梦境对象 | 0.7 | 0.4 | 0.6 | 否 |
| `hypothesis` | 梦中新抽象、未验证模式 | 0.9 | 0.1 | 0.4 | 否 |

晋升阶梯（写入侧门）：

```
hypothesis --(provenance 完整 + ≥1 独立在线佐证 + 无未解矛盾)--> derived
derived    --(可重放/直接测量 + provenance 链完整)--> direct
任意        --(矛盾/撤销/过期)--> contested / quarantined
```

硬规则：

1. 梦境产物**不得覆盖** source 记忆；只能新增 `supersedes` 建议，晋升前保留双份。
2. 梦境产物默认 `derived`；只有通过晋升门才可提升 `evidential_authority`。
3. 梦境产物**永远不能**直接授权动作，也不能修改 `PersonaKernel`。
4. 由梦境提出的人格修订走 candidate 通道，证据阈值是在线路径的 2 倍，且必须人工确认涉及安全/授权边界的变更。
5. 离线反思不自动加分：**抽象权威可以高，证据权威必须低**；这正是对「梦是否应获得更高权威」的回答。

一句话：**Dream can frame, not justify; can propose, not authorize.**

### 18.7 数据契约（补）

```sql
CREATE TABLE memory_state_events (
  event_id        TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  memory_id       TEXT NOT NULL,
  event_type      TEXT NOT NULL,  -- ingest|index|consolidate|recall|revise|forget|retrain|promote|quarantine
  policy_id       TEXT NOT NULL,
  policy_version  TEXT NOT NULL,
  decision        TEXT NOT NULL,
  route           TEXT,
  authority_json  TEXT NOT NULL,  -- {abstraction,evidential,influence}
  evidence_json   TEXT NOT NULL,
  actor           TEXT NOT NULL,
  soul_scope      TEXT NOT NULL,  -- tenant|org_unit|agent|user
  created_at      TEXT NOT NULL
);
CREATE INDEX idx_mem_state ON memory_state_events(tenant_id, memory_id, created_at);

CREATE TABLE influence_decisions (
  query_id        TEXT NOT NULL,
  memory_id       TEXT NOT NULL,
  tenant_id       TEXT NOT NULL,
  route           TEXT NOT NULL,
  influence       REAL NOT NULL,
  dims_json       TEXT NOT NULL,   -- 七维原始分
  caps_json       TEXT NOT NULL,
  decided_at      TEXT NOT NULL,
  PRIMARY KEY (query_id, memory_id)
);

CREATE TABLE soul_resolution_events (
  event_id        TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  dimension       TEXT NOT NULL,
  higher_scope    TEXT NOT NULL,
  lower_scope     TEXT NOT NULL,
  decision        TEXT NOT NULL,
  reason          TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
```

### 18.8 不变量（补 I13–I24）

| # | 不变量 | 测试 |
|---|---|---|
| I13 | 保留决策与影响决策独立：保留不得提升影响权威，影响不得删除记忆 | `test_memory_dual_gate.py::test_independence` |
| I14 | 影响权威逐查询计算，且被 authority_class 与 route 双重 clamp | `test_influence_gate.py::test_per_query_cap` |
| I15 | `derived/hypothesis` 不得授权动作，不得修改 `PersonaKernel` | `test_dream_authority.py::test_no_authorize_no_kernel` |
| I16 | 八类记忆迁移全部产生 `memory_state_event`，含 policy_id/version | `test_memory_governance.py::test_audit_completeness` |
| I17 | Hygiene Gate 只允许七种 route，禁止静默放行 | `test_hygiene_gate.py::test_closed_routes` |
| I18 | Soul 优先级是全序；低层不得覆盖高层；覆盖必须留痕 | `test_soul_priority.py::test_total_order_and_audit` |
| I19 | 巩固只在 sleep 状态执行；状态迁移可审计 | `test_sleep_gate.py::test_no_consolidation_in_active` |
| I20 | 软衰减可逆；硬修剪需两个独立信号 + 二次确认 | `test_soft_decay.py::test_reversible_and_two_signal` |
| I21 | 梦境产物不得覆盖 source；`supersedes` 晋升前保留双份 | `test_dream_authority.py::test_no_overwrite` |
| I22 | 跨租户记忆/梦境零泄漏 | `test_memory_governance.py::test_tenant_isolation` |
| I23 | 晋升要求 provenance 链完整 + 无未解矛盾 | `test_dream_authority.py::test_promotion_preconditions` |
| I24 | 注入上下文的每条记忆必须带来源标签；梦境不得无标签注入 | `test_influence_gate.py::test_labeled_injection` |

### 18.9 指标（补）

| 指标 | 定义 | 目标 |
|---|---|---|
| context pollution rate | 注入上下文中与当前 query 无关/被 block 的条目占比 | ≤ 3% |
| influence precision | 被注入且实际影响正确决策的记忆 / 被注入记忆 | ≥ 0.75 |
| retention–influence disagreement | 保留=available 但影响=block 的比例（应非零，证明两闸解耦） | 5%–40% |
| dream promotion precision | 晋升为 derived 的梦境产物中被后续在线证据佐证的占比 | ≥ 0.80 |
| dream contradiction rate | 被在线证据推翻的梦境产物占比 | ≤ 5% |
| retrieval preemption rate | 梦境修复后避免的检索失败 / 探针失败 | ≥ 30% |
| soft decay restore success | 软衰减后恢复原状的成功率 | 100% |
| soul override violations | 低层覆盖高层事件数 | 0 |
| sleep leakage | 活跃状态执行巩固的次数 | 0 |
| no-evidence honesty | 全候选被 block 时显式声明「无可用证据」的比例 | 100% |

可检验实验：

- **E1**：同 token 预算下，influence-gated top-k vs similarity top-k，任务正确率与污染率。
- **E2**：梦境晋升产物在 30 天窗口内的在线佐证率。
- **E3**：两闸解耦后，false「无可用证据」是否上升；若显著上升，说明保留闸与影响闸阈值需要重新校准。
- **E4**：Soul-conditioned 梦境 vs 全局配置，行为指纹 JS 散度与人格一致性评分。

### 18.10 路线补丁

| 阶段 | 内容 | 出口 |
|---|---|---|
| P1-c | 保留/影响双闸 + Hygiene Gate + `memory_state_events` | I13–I17 绿 |
| P1-d | 用户信念 / 记忆的 Anamnesis 权威链；`superseded_by` 双份 | I21、I23 绿 |
| P2-c | SleepClock + Dreamer/Analyst/Consolidator + DreamObject + 写入侧晋升门 | I18–I19、I21 绿 |
| P2-d | 学习化修复/巩固（GRPO/离线 RL）+ OPE + 影子 | E2、E3 达标 |
| P3-b | Soul 四层优先级 + 多租户梦境隔离 + 跨设备同步 | I18、I22 绿 |

### 18.11 风险与反模式

| 风险 | 处理 |
|---|---|
| 过度门控导致上下文过空、答案质量下降 | 设最小 direct 配额与显式「无证据」声明；E3 监控；必要时只降 `influence` 不降 `retention` |
| Hygiene 七维分数被游戏化 | 维度权重走策略注册表 + 版本化；定期 OOD 校准；硬 clamp 不可由分数覆盖 |
| 梦境产物污染事实层 | 默认 `derived`，`evidential` 上限 0.4；晋升需独立佐证；禁止授权动作 |
| 学习化修复奖励作弊 | 只奖励 held-out 检索改善；action mask 禁止删 direct 证据；人工抽检 |
| Soul 优先级被误用为授权 | 明确分离：Soul 管价值/显著性，规范账管授权；两者交叉时必须双门 |
| 睡眠调度跨时区失真 | 服务端时钟 + 时区显式配置；手动 override 留痕 |
| 账本写入放大 | 事件分区 + 聚合视图 + TTL；原文只存 digest/引用 |

明确不做：

- 不把「检索到」等同于「已授权进入上下文」。
- 不给梦境产物自动权威加成；离线反思不是证据。
- 不用一套全局 salience profile 覆盖所有 persona。
- 不把 Soul 优先级当作授权优先级。
- 不用 LLM 自评作为梦境修复的 reward。


