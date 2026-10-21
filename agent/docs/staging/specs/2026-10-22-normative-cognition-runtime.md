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
6. **监控环是兄弟运行时**：不执行用户任务、不写授权、不改内核；主动准备只是候选，不是已授权上下文。详见 §19。
7. **世界模型是预测源，不是事实源/授权源**：预测默认 hypothesis，reconcile 后才升格；跨 worldview 不粘合；详见 §20。

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

记忆治理不变量 I13–I24 见 §18.8；常驻运行时不变量 I25–I34 见 §19.8，监控降级不变量 I35–I40 见 §19.12.8，世界模型不变量 I41–I50 见 §20.10；五表共同构成完整不变量集。

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

1. 三本账 schema 落地，I1–I50 全绿（I13–I24 见 §18.8，I25–I34 见 §19.8，I35–I40 见 §19.12.8，I41–I50 见 §20.10）。
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
12. `PreparedContext` 只作候选、带 TTL、按租户隔离；空闲环不产生用户可见动作（I26–I27、I34）。
13. VIGIL 兄弟运行时只读日志、只写诊断/提案；漂移三级告警可升级、`pause` 不可被任务 Agent 自清（I25、I28–I29、I33）。
14. 教师/蒸馏产物为 `derived`，惊奇只做排序与探索项，不得绕过 Hygiene Gate（I30、I32）。
15. 监控降级有 L0–L4 状态机、有界重启、可见降级；任务 p99 开销 ≤5%，且安全联锁不由监控独占（I35–I40）。
16. 世界模型只作预测/对账/风险信号：预测不升格为事实、不授权动作；跨 worldview 不粘合；旧 epoch 预测不消费（I41–I50）。

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

记忆治理补章（§18）的分期 P1-c / P1-d / P2-c / P2-d / P3-b 与本表并行推进；常驻运行时（§19）的分期 P1-e / P2-e / P2-f / P3-c 同样并行；世界模型结合（§20）的分期 P1-f / P2-g / P3-d 也并行。阶段号一一对应；出口以 §18.8 / §19.8 / §19.12.8 / §20.10 的不变量与各节指标为准。

---

## 17. 附：四条给实施者的提醒（双螺旋压缩）

1. **路径可丢，坐标不可丢**：实现可以换表、换算法、换分类器；三组坐标不能丢——动作 `(proposition, act, scope, status, time)`、记忆 `(memory_id, source_class, epistemic_status, evidential_authority, time)`、影响 `(query_id, memory_id, route, influence_authority, decided_at)`。
2. **先建切口，再谈学习**：没有授权拓扑的硬边，任何策略学习都会学会绕过；先让未授权动作在结构上不可达。
3. **反例即结构在说话**：`oppose+oppose` 误报、`clear_confirmation` 丢状态、`adaptive_layer` 前缀漂移——这三个反例不是 bug 清单，是「混轴」这一个结构问题的三个投影。
4. **监控不是任务的一步**：兄弟运行时的价值在失败隔离；把它塞进任务循环，等于让被监控者自己写体检报告。

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

### 18.4 梦境周期：Dreamer 生成，Analyst 归因，Consolidator 受门执行

把 [MemoryMaintainer](file:///workspace/agent/huginn/memory/maintainer.py#L20-L102) 从「定时衰减线程」升级为受治理的梦境周期，但**不新增 agent 框架**：Dreamer / Analyst / Consolidator 是同一周期里的三个角色。

三者的边界不是流水线工序，而是**生成与验证的不对称**：

| 角色 | 回答的问题 | 输入 | 输出 | 允许动作 | 失败模式 | 度量 |
|---|---|---|---|---|---|---|
| **Dreamer（生成）** | 哪里可能有问题？接下来可能需要什么？什么改变了信念？ | 目标/承诺、近期轨迹、记忆图、检索探针、DreamObject 历史 | `DreamFinding` 列表（含 surprise、expected_impact、cost） | 只读探测；可写 finding；不可改记忆 | 高召回低精度，制造噪声 | finding 召回率、surprise 校准、单周期预算 |
| **Analyst（归因）** | 为什么？属于哪类故障？哪种修复最可能有效？ | findings + 可寻址证据 + 反事实回放 | `Diagnosis`（cause_class、confidence、affected_ids、repair_class） | 只读；可弃权；不可执行修复 | 误诊导致错误修复 | 诊断准确率、弃权率、反事实一致率 |
| **Consolidator（执行）** | 在允许的风险等级内，执行哪一个最小修复？ | 高置信 diagnosis + 策略门 | `RepairAction` + 回滚句柄 | 受门执行；可逆；不得删 direct 证据 | 过度修复、不可逆破坏 | 修复成功率、回滚率、检索改善 |

**Dreamer 的三类探针**

1. **需求探针（ProAct / Pask / ContextAgent）**：从 `goal_store` 的目标、未解 unknown、prospective intentions 和近期轨迹中采样「接下来可能被问到的查询」；对这些查询做 **dry-run 检索**，记录缺口、延迟、低权威命中。它输出的是「预期需求 + 缺什么证据」，不是答案，也不是动作。
2. **结构探针**：扫描记忆图/结论超图的孤立节点、低连接簇、桥边、陈旧 anchor、重复簇、矛盾压力、跨域未连边。复用 [scan_contradictions](file:///workspace/agent/huginn/memory/longterm.py#L1676-L1754)、[anchor](file:///workspace/agent/huginn/memory/anchor.py)、[cluster](file:///workspace/agent/huginn/memory/cluster.py#L36)。
3. **惊奇探针（AutoDiscovery）**：以贝叶斯惊奇度作为优先级，而不是让 LLM 直接判断「有没有用/新不新」。
   ```
   surprise = KL( P(belief | evidence) || P(belief) )
   ```
   - 廉价近似：预测误差、矛盾新出现、dry-run 检索失败、同一 subject 的权威断层。
   - 完整版：对候选假设做 LLM 采样估计先验/后验分布，取 KL；沿用 AutoDiscovery 的 MCTS/渐进展宽做候选扩展。
   - 注意：现有 [belief_entropy](file:///workspace/agent/huginn/utils/belief_entropy.py#L1-L13) 衡量的是**上下文压缩后的不确定度**，不是信念修订惊奇；两者不能混用，需新增 `belief_surprise` 字段。

`DreamFinding`：

```python
@dataclass
class DreamFinding:
    finding_id: str
    probe_type: Literal["demand", "structure", "surprise"]
    subject: str
    evidence_ids: list[str]        # memory_id / claim_id / chunk_id / act_id
    surprise: float                # 0..1，贝叶斯惊奇或代理
    expected_impact: float         # 若修复，预期检索改善
    cost: float                    # 预算估计
    created_at: str
```

**Analyst 的四项归因**

1. **反事实归因**：对每个 finding，枚举候选修复类（缺链/重复/过时/过度泛化/冲突/覆盖缺口/溯源缺口/权威错配），用回放或廉价模拟问：「加上这个修复，探针会通过吗？」通过才给高置信。
2. **目标锚定（RefAct）**：不是只看「下一步做什么」，而是每步计算**当前状态相对于原始目标的关系**。目标来自 `goal_store` 的 objective/required_results，状态来自 `session_state` + 最近执行结果；偏离超过阈值 → `goal_drift` 诊断。这是对 ReAct 式「向前看」的结构性修正。
3. **漂移检测（Smith）**：三类检测器 + 三级告警。
   - 倾向性回归：反思频次/深度相对基线下滑 > 15%。
   - 求真下降：证据引用率/接地率相对基线下滑 > 20%。
   - 因果图编辑距离：目标/计划因果图相对基线 > 0.3。
   - 告警级别：`review`（记录 + 提示）→ `intervene`（注入纠正/降级自主面）→ `pause`（停手等确认）。
   - 阈值走策略注册表，不硬编码；基线按 tenant/persona/task family 分别维护。
4. **校准与弃权**：Analyst 必须输出 confidence，并在证据不足时**弃权**（`abstain`），而不是猜一个诊断。这是对「噪声验证器 + 噪声修复器会破坏正确计划」的直接防御。

`Diagnosis`：

```python
@dataclass
class Diagnosis:
    diagnosis_id: str
    finding_id: str
    cause_class: Literal["missing_link", "duplicate", "stale", "overgeneralized",
                         "contradiction", "coverage_gap", "provenance_gap",
                         "authority_mismatch", "goal_drift"]
    confidence: float              # < 阈值则 abstain
    affected_ids: list[str]
    repair_class: str
    expected_gain: float
    risk: Literal["low", "medium", "high"]
    counterfactual_passed: bool
```

**为什么必须拆开**

- Dreamer 可以错，因为它只产生候选；Analyst 不能乱猜，因为它决定修什么；Consolidator 不能自由发挥，因为它改的是状态。
- 如果把三者合成一个 LLM 调用，就会出现「生成 → 自检 → 重试」的经典失败：验证器和修复器同时有噪声时，修复会损坏本来正确的计划。
- 这与快时的三舱反射（[three_cabin_reflector](file:///workspace/agent/huginn/metacog/three_cabin_reflector.py#L1-L17)：Step/Evaluate/Reflect）是**不同时间尺度**的同一原则：快时做单步反射，慢时做图级诊断；不要用一个角色承担两个时间尺度。

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

---

## 19. 常驻运行时：交互间隔中的监控、预测与自我修改

本节回应 ProAct / Pask / ContextAgent / CER / MUSE / Memento / ATLAS / RefAct / Smith / VIGIL / AutoCrit / Wink / AutoDiscovery 的共同方向：Agent 不再只是「交互时刻才启动的推理过程」，而是**在交互间隔中持续运行、带独立监控层与受控自我修改能力的常驻系统**。

### 19.1 三层时间尺度，三个运行时

| 循环 | 节奏 | 归属 | 写权限 | 失败隔离 |
|---|---|---|---|---|
| **快环** 交互 | 每轮 / 每步 | 任务 Agent | 正常执行，受规范账约束 | 失败影响当轮 |
| **慢环** 梦境 | 空闲 / 睡眠窗口 | Dreamer / Analyst / Consolidator | 只能改索引、链接、显著性、摘要；不得删 direct 证据 | 失败不影响在线任务 |
| **监控环** 常驻 | 持续 / 异步 | VIGIL 兄弟运行时 | 只能写诊断、告警、受保护的 adaptive 更新与只读修复提案 | 失败必须显式降级，不得静默 |

三条硬规则：

1. **监控环是兄弟运行时，不是任务 Agent 的一个阶段**。它不执行用户任务、不写 `authorize`、不改 `PersonaKernel`。
2. **慢环的产物是候选**，进入上下文仍须影响闸（§18.2）授权。
3. **快环的失败不能靠慢环事后补**；慢环只能降低下一次失败的概率。

### 19.2 主动准备：回答先于提问，但不先于授权

ProAct / Pask / ContextAgent 的共同点是利用空闲时间预测需求；在我们的架构里，它落在 Dreamer 的**需求探针**与一个受控的 `PreparedContext` 缓存上。

```python
@dataclass
class PreparedContext:
    prep_id: str
    tenant_id: str
    persona_id: str
    predicted_need: str            # 预期问题/任务
    query_hint: str
    evidence_ids: list[str]
    authority_vector: dict         # abstraction / evidential / influence
    ttl_s: int
    created_at: str
    status: Literal["candidate", "expired", "consumed"]
```

规则：

- 空闲期可以**预取、预计算、预组织**，但不得产生用户可见动作。
- `PreparedContext` 只是候选，不自动注入；命中时仍过影响闸，并按 `query_hint` 与实际 query 的相关性打折。
- 必须带 TTL；过期即 `expired`，不得作为「当前共同基础」使用。
- Pask 式「静默副驾驶」输出的是概念澄清与语义提示，不是事实断言；ContextAgent 式「先判断是否需要主动服务」对应 `predicted_need.requires_service`，但「需要服务」不等于「被授权服务」。
- 预取内容跨租户零共享；敏感字段只存 digest/引用。

反模式：把「提前准备好」做成「提前执行」；把「预测到用户会问」当成「用户已经问过」。

### 19.3 VIGIL 式兄弟监控运行时

把 VIGIL 的结构映射到现有底座，不新增一个会执行任务的 agent：

| VIGIL 概念 | Huginn 落点 | 约束 |
|---|---|---|
| 行为日志摄取 | [harness_ledger](file:///workspace/agent/huginn/research/harness_ledger.py#L1-L13) + audit chain + `task_metrics` | 只读；按租户分区 |
| 情感银行（带衰减） | [persona_emotion](file:///workspace/agent/huginn/persona_emotion.py#L119) 的事件轨迹 + 衰减 | 情绪不是授权，不得改写身份 |
| Roses / Buds / Thorns | `MonitorFinding`：稳定优势 / 新兴机会 / 失败模式 | 必须引用事件 id，不得泛泛而谈 |
| 受保护的提示更新 | [meta_improver](file:///workspace/agent/huginn/harness/meta_improver.py#L492) 的 GREEN 采纳门 | 只改 adaptive/prompt block，不碰内核 |
| 只读代码修复提案 | `CodeFixProposal`（diff + 证据 + 风险） | 只提案，不自动 apply |
| 元程序性自我修复 | 诊断工具失败时输出**精确内部错误 + 降级诊断 + 修复计划** | 禁止优雅降级掩盖失败 |

`MonitorFinding`：

```python
@dataclass
class MonitorFinding:
    finding_id: str
    tenant_id: str
    persona_id: str
    kind: Literal["rose", "bud", "thorn"]   # 稳定优势 / 新兴机会 / 失败
    subject: str
    evidence_ids: list[str]
    decayed_weight: float
    confidence: float
    suggested_surface: Literal["prompt_adaptive", "code_proposal", "none"]
    created_at: str
```

与 AutoCrit 的关系：AutoCrit 的「推理 / 批判 / 执行监控」三元环是**快环内**的；我们的 [three_cabin_reflector](file:///workspace/agent/huginn/metacog/three_cabin_reflector.py#L1-L17) 已经承担其中的单步反射。VIGIL 是**跨轨迹、异步**的外部监控，不替代三舱，二者时间尺度不同。

Wink 的可迁移点：用大规模轨迹建立**不当行为分类学**，做轻量异步干预，而不是每步都跑重型反思。对应本 spec 的轨迹索引（§11.1）与 `intervention_events`；单次干预解决率作为指标。

### 19.4 目标漂移：从「规划下一步」到「持续对齐目标」

RefAct 的修正是结构性的：ReAct 向前看，RefAct 持续检查当前状态相对于原始目标的位置。落地为 `GoalAnchor`：

```python
@dataclass
class GoalAnchor:
    goal_id: str
    objective: str
    required_results: list[str]
    current_state_digest: str
    alignment: float                 # 0..1
    drift_reasons: list[str]
    checked_at: str
```

Smith 的三类检测器 + 三级告警（阈值可配置，默认值仅作起点）：

| 检测器 | 信号 | 默认触发 | 级别 |
|---|---|---|---|
| 倾向性回归 | 反思频次/深度相对基线 | 下降 > 15% | review |
| 求真下降 | 证据引用率 / 接地率相对基线 | 下降 > 20% | intervene |
| 因果图编辑距离 | 目标/计划因果图相对基线 | > 0.3 | pause |

- 告警级别只能由监控环升级；任务 Agent 不能自己清除 `pause`。
- `pause` 是 fail-safe，不替代规范账；恢复需要显式确认或策略放行。
- `drift_events` 必须带检测器版本与基线 id，避免「阈值漂移」无法追责。
- 现有 [detect_drift](file:///workspace/agent/huginn/metacog/target_chain.py#L231) 可作为第一版倾向性信号，但需要补因果图与接地率两个维度。

### 19.5 经验蒸馏与在线适应：改检索策略，不改权重

CER / MUSE / Memento / ATLAS 的共同点是**梯度无关的推理时适应**：

| 机制 | 做法 | Huginn 落点 | 权威类别 |
|---|---|---|---|
| CER | 历史交互合成动态记忆缓冲区，新任务检索复用 | [experience_archive](file:///workspace/agent/huginn/research/experience_archive.py#L43-L70) + 动态 buffer | derived |
| MUSE | 子任务完成后批判执行日志，蒸馏结构化知识 | 子任务结束钩子 + 经验账 | derived |
| Memento | 成功/失败案例库 + 软 Q 学习优化检索策略 | 案例库 + `AdmissionPolicy` 的同族 bandit/RL | 策略，不改权重 |
| ATLAS | 教师-学生分离：教师蒸馏指导经验，学生执行 | 教师只写经验账；学生只执行 | 教师产物 = derived |

硬规则：

- 蒸馏产物默认 `derived`，不得直接当事实依据；晋升走 §18.6 的写入侧门。
- 教师不得执行用户任务；学生不得改写教师记忆。
- 软 Q / 检索策略更新必须可回放、可审计；每次策略切换记录 `policy_version`。
- 经验库按 tenant + task family 分区，防止跨域污染。

### 19.6 贝叶斯惊奇：不要问「这有用吗」，问「什么改变了信念」

AutoDiscovery 的核心可迁移点：有用性/新颖性主观且难优化，**信念变化**可计算、可验证。

```python
surprise = KL( P(belief | evidence) || P(belief) )
```

用途：

1. **Dreamer 优先级**：高惊奇 finding 先诊断，低惊奇进低优先级队列或丢弃。
2. **探索奖励**：离线策略学习里，用 surprise 作为探索项，而不是让 LLM 自评「新颖」。
3. **监控信号**：惊奇骤降可能意味着认知停滞；惊奇骤升可能意味着分布外事件。

约束：

- 惊奇不是权威：高惊奇不能绕过 Hygiene Gate，也不能提升 evidential authority。
- 估计方法必须版本化：廉价代理（预测误差/矛盾出现/检索失败）与完整 LLM 采样版分开记录，避免把代理当定理。
- 校准目标：与人工「惊讶感」评分相关性 ≥ 0.6（AutoDiscovery 报告 0.67，仅作参照，不作承诺）。
- 不能只用 LLM 自评作为 reward；至少两路信号交叉（惊奇 + 下游检索/任务改善）。

### 19.7 数据契约（补）

```sql
CREATE TABLE idle_jobs (
  job_id          TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  persona_id      TEXT NOT NULL,
  job_type        TEXT NOT NULL,   -- demand_probe|structure_probe|surprise_probe|prepare_context
  sleep_state     TEXT NOT NULL,   -- active|sleep
  budget_ms       INTEGER NOT NULL,
  status          TEXT NOT NULL,   -- queued|running|done|failed|expired
  created_at      TEXT NOT NULL,
  finished_at     TEXT
);

CREATE TABLE prepared_contexts (
  prep_id         TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  persona_id      TEXT NOT NULL,
  predicted_need  TEXT NOT NULL,
  evidence_json   TEXT NOT NULL,
  authority_json  TEXT NOT NULL,
  ttl_s           INTEGER NOT NULL,
  status          TEXT NOT NULL,   -- candidate|expired|consumed
  created_at      TEXT NOT NULL
);

CREATE TABLE monitor_findings (
  finding_id      TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  persona_id      TEXT NOT NULL,
  kind            TEXT NOT NULL,   -- rose|bud|thorn
  subject         TEXT NOT NULL,
  evidence_json   TEXT NOT NULL,
  decayed_weight  REAL NOT NULL,
  confidence      REAL NOT NULL,
  suggested_surface TEXT NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE TABLE drift_events (
  event_id        TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  session_id      TEXT NOT NULL,
  detector        TEXT NOT NULL,   -- propensity|truth_seeking|causal_edit
  baseline_id     TEXT NOT NULL,
  observed        REAL NOT NULL,
  threshold       REAL NOT NULL,
  level           TEXT NOT NULL,   -- review|intervene|pause
  evidence_json   TEXT NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE TABLE intervention_events (
  intervention_id TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  session_id      TEXT NOT NULL,
  trigger_event   TEXT NOT NULL,
  action          TEXT NOT NULL,   -- prompt_patch|scope_reduce|pause|code_proposal
  outcome         TEXT,            -- resolved|unresolved|regressed
  created_at      TEXT NOT NULL
);
```

### 19.8 不变量（补 I25–I34）

| # | 不变量 | 测试 |
|---|---|---|
| I25 | 监控环是兄弟运行时：不得执行用户任务、不得写 `authorize`、不得改 `PersonaKernel` | `test_vigil_runtime.py::test_no_task_authority` |
| I26 | `PreparedContext` 只是候选，不自动注入；TTL 过期即不可用 | `test_proactive_prep.py::test_candidate_and_ttl` |
| I27 | 空闲环不得产生用户可见动作；任何动作仍需规范账授权 | `test_proactive_prep.py::test_no_visible_action` |
| I28 | 监控更新只作用于 adaptive/prompt block；内核不可变 | `test_vigil_runtime.py::test_protected_surface` |
| I29 | 告警级别单调升级；任务 Agent 不能自行清除 `pause` | `test_drift_gate.py::test_pause_not_self_clearable` |
| I30 | 教师/蒸馏产物为 `derived`；未经晋升不得作为直接证据 | `test_distillation_authority.py::test_derived_only` |
| I31 | 软 Q / 检索策略更新不改权重，且可回放审计 | `test_case_policy.py::test_replayable_no_weight_update` |
| I32 | 高惊奇不得绕过 Hygiene Gate，不得提升证据权威 | `test_bayesian_surprise.py::test_surprise_not_authority` |
| I33 | 监控自身失败必须显式降级（精确错误 + 降级诊断），禁止静默通过 | `test_vigil_runtime.py::test_explicit_degradation` |
| I34 | 空闲任务按租户隔离且有预算上限；超预算即停 | `test_idle_jobs.py::test_tenant_and_budget` |

### 19.9 指标（补）

| 指标 | 定义 | 目标 |
|---|---|---|
| anticipation hit rate | `PreparedContext` 被后续真实 query 命中的比例 | ≥ 40% |
| wasted prefetch ratio | 过期/未消费的预取占比 | ≤ 30% |
| time-to-evidence | 从问题出现到证据齐备的时间 | 相对基线下降 ≥ 25% |
| drift detection lead time | 漂移发生到告警的步数 | ≤ 3 步 |
| false intervention rate | 干预后无改善的比例 | ≤ 10% |
| intervention success rate | 单次干预解决的不当行为比例 | ≥ 80%（Wink 90% 作参照） |
| error propagation reduction | 下游错误传播相对基线下降 | ≥ 30%（AutoCrit 减半作参照） |
| surprise–human correlation | 惊奇分与人工惊讶感评分相关性 | ≥ 0.6 |
| monitor coverage | 被监控事件 / 应监控事件 | ≥ 95% |
| idle cost per useful prep | 每个有效预取的 CPU/LLM 成本 | 有预算上限，按租户计量 |

### 19.10 路线补丁

| 阶段 | 内容 | 出口 |
|---|---|---|
| P1-e | `PreparedContext` + idle jobs + TTL + 影响闸接入 | I26–I27、I34 绿 |
| P2-e | VIGIL 兄弟运行时 + `drift_events` + 三级告警 + L0–L4 降级状态机 | I25、I28–I29、I33、I35–I40 绿 |
| P2-f | 案例库 + 子任务后蒸馏 + 软 Q 检索策略 | I30–I31 绿 |
| P3-c | 教师-学生分离 + 主动策略离线训练 + 惊奇优先 | E5、E6 达标 |

可检验实验：

- **E5**：主动预取（ProAct 式）在真实任务分布上，time-to-evidence 与准确率是否优于纯被动检索。
- **E6**：贝叶斯惊奇排序 vs LLM 自评排序，在人工标注的 200 条 finding 上，前者 precision@10 是否更高。

### 19.11 风险与反模式

| 风险 | 处理 |
|---|---|
| 常驻系统变成「常驻执行」 | 监控环写权限白名单；用户可见动作一律走规范账 |
| 预取泄漏隐私 / 跨租户 | 预取按 tenant 分区；敏感内容只存 digest；TTL 短 |
| 监控变成第二个任务 Agent | 禁止工具执行权限；只读日志 + 提案；代码修复只出 diff |
| 漂移阈值误报 | 基线按 tenant/persona/task family 维护；三级告警先 review；阈值版本化 |
| 惊奇奖励被游戏化 | 惊奇只做排序/探索项；硬约束 mask；两路信号交叉验证 |
| 教师-学生共谋 | 教师产物 derived；学生执行记录可审计；独立验证器抽查 |
| 监控自身静默失败 | 必须输出精确错误 + 降级诊断；I33 硬门 |

明确不做：

- 不做「监控 Agent 直接接管任务」。
- 不做「预取即执行」。
- 不做「用 LLM 自评新颖性」作为唯一探索信号。
- 不做「漂移阈值写死在代码里」。
- 不做「教师产物直接当事实」。

### 19.12 监控环降级与故障隔离

目标不是「监控永不失败」，而是**监控失败时主任务照常运行、失败可见、安全联锁不被削弱**。

#### 19.12.1 先分职责，再定失败方向

| 类别 | 例子 | 失败方向 | 对主任务 | 审计要求 |
|---|---|---|---|---|
| **建议类** advisory | Roses/Buds/Thorns、主动预取、梦境修复建议、漂移提示 | **fail-open**：没有建议就等于没有建议 | 零影响 | 记 `degradation_event`，但不阻断 |
| **审计类** evidentiary | 监控覆盖率、detector 版本、降级原因、未监控会话标记 | **fail-visible**：不能静默变成「一切正常」 | 零影响，但 `trust` 必须降级 | 强制写 `monitor_health` |
| **安全联锁类** interlock | 授权、预算、沙箱、硬暂停 | **fail-closed**，但必须由**监控之外**的独立门执行 | 按联锁语义阻断 | 独立审计 |

硬规则：**监控环可以请求联锁动作，但不能是任何硬约束的唯一执行点**。否则监控一挂，安全边界随之消失。

#### 19.12.2 隔离边界：五个「不」

1. **不同执行体**：独立进程优先；至少独立线程/事件循环，禁止在任务 Agent 的协程里同步跑诊断。
2. **不共享可变状态**：只读快照或事件日志；不持有记忆库/知识库的写锁。
3. **不阻塞**：任务 Agent 永不 `await` 监控结果；`status()` 带 TTL，过期即视为 `blind`。
4. **不抛异常**：监控边界捕获所有异常并转成 `MonitorStatus`；异常不得穿过边界。
5. **不共享资源池**：独立队列、独立预算、独立 DB 连接；监控的 CPU/IO 不得挤占任务路径。

#### 19.12.3 降级状态机（L0–L4）

| 状态 | 触发 | 监控行为 | `trust` | 主任务行为 |
|---|---|---|---|---|
| **L0 full** | 全部 detector 正常 | 全量诊断 | `full` | 正常 |
| **L1 partial** | 部分 detector 不可用 / LLM 5xx / DB 写入缓冲 | 规则版 + 可用的 detector；标注覆盖缺口 | `partial` | 正常，System3 略增澄清 |
| **L2 blind** | 仅心跳存活，诊断全部失败 | 只发心跳与 `blind` 状态 | `blind` | 正常，但提高审批门槛、限制高风险自主动作 |
| **L3 offline** | 心跳连续 3 个周期缺失 | 由 supervisor 重启；期间无诊断 | `unknown` | 正常；会话标记 `unmonitored` |
| **L4 quarantined** | 自身失控（死循环、输出 schema 非法、越权写、跨租户访问） | 立即 kill，指数退避重启 | `unknown` | 正常；不继承其任何输出 |

恢复路径：`L4/L3 → 半开探测（单个 canary job）→ L1 → L0`；退避 1/2/4/8 分钟，单窗口最多重启 3 次，超过则停在 L3 并告警，禁止 crash loop。

#### 19.12.4 熔断、队列与预算

- **单任务超时**：轻量诊断默认 2s，重型诊断默认 30s；超时即丢该 job，不影响其他 job。
- **有界队列**：默认 1000；满时按优先级丢最旧/最低的 finding，记录 `dropped_findings`。
- **并发上限**：按租户设置，防止一个租户的监控洪峰拖垮全局。
- **预算上限**：CPU 时间、内存、LLM token/小时；超限进入 L1/L2，而不是抢任务资源。
- **熔断器**：1 分钟错误率 > 20% → open 5 分钟 → half-open 放 1 个 canary job；成功则恢复，失败则继续 open。
- **DB 写入**：本地 ring buffer + 异步 flush；DB 锁不可用则丢最旧，不重试阻塞主路径。

#### 19.12.5 监督与重启

- watchdog **在监控之外**（进程管理器 / 独立 supervisor 协程），只读心跳表。
- 心跳写 `<runtime>/monitor_health.json` 或 `monitor_health` 表；3 个周期未更新 → 重启。
- 崩溃恢复是 **crash-only**：监控随时可被杀，任务侧无需清理；重启后从事件日志幂等重建，不补发错过的 finding，只标记 `coverage_gap`。
- 重启次数与原因写入 `degradation_events`；超过阈值 → L3 + 告警。

#### 19.12.6 降级时的补偿控制：降自主，不降可用性

当 `trust=blind/unknown`：

- 主任务继续运行，不阻断。
- System3 提高澄清阈值、要求高风险动作显式授权、收紧预算上限。
- 最终输出标记 `unmonitored`，不得宣称「已通过监控」。
- 授权/预算/沙箱等联锁仍由独立门执行；监控缺席不改变它们的判定。

这是有意识的取舍：**监控失败时失去的是语义漂移检测，不是执行能力**；用「更严格的自主边界」补偿，而不是用「停摆」补偿。残余风险必须写进会话元数据。

#### 19.12.7 数据契约

```sql
CREATE TABLE monitor_health (
  monitor_id      TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  state           TEXT NOT NULL,   -- L0|L1|L2|L3|L4
  trust           TEXT NOT NULL,   -- full|partial|blind|unknown
  coverage_json   TEXT NOT NULL,   -- 各 detector 的可用性
  detector_versions TEXT NOT NULL,
  last_heartbeat  TEXT NOT NULL,
  error_rate_1m   REAL NOT NULL,
  restart_count   INTEGER NOT NULL,
  degraded_reason TEXT,
  updated_at      TEXT NOT NULL
);

CREATE TABLE degradation_events (
  event_id        TEXT PRIMARY KEY,
  monitor_id      TEXT NOT NULL,
  tenant_id       TEXT NOT NULL,
  from_state      TEXT NOT NULL,
  to_state        TEXT NOT NULL,
  trigger         TEXT NOT NULL,   -- timeout|crash|schema_invalid|runaway|db_lock|llm_5xx|clock_skew|queue_full|unauthorized_write
  task_impact     TEXT NOT NULL,   -- none|strict_autonomy|blocked_by_interlock
  evidence_json   TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
```

```python
@dataclass(frozen=True)
class MonitorStatus:
    monitor_id: str
    state: Literal["L0", "L1", "L2", "L3", "L4"]
    trust: Literal["full", "partial", "blind", "unknown"]
    coverage: dict[str, bool]
    last_heartbeat: str
    degraded_reason: str | None

def status(self) -> MonitorStatus:
    """永不抛异常；内部失败返回 trust='unknown'。"""
```

`submit(job)` 非阻塞，返回 `job_id` 或 `DROPPED`；任务侧不关心结果，只读 `status()`。

#### 19.12.8 不变量（补 I35–I40）

| # | 不变量 | 测试 |
|---|---|---|
| I35 | 监控失败不得向任务循环抛异常或阻塞；`status()` 永不 raise | `test_monitor_degradation.py::test_no_exception_no_block` |
| I36 | 监控存在时任务 p99 开销 ≤ 5%，且不共享写锁 | `test_monitor_degradation.py::test_overhead_and_locking` |
| I37 | 监控不得是授权/预算/沙箱的唯一执行点 | `test_monitor_degradation.py::test_interlock_independence` |
| I38 | 任何降级必须可见：`trust != full` 必写 `monitor_health` + `degradation_events`；状态过期视为 blind | `test_monitor_degradation.py::test_visible_degradation` |
| I39 | 重启有界退避；crash loop 停在 L3，不得无限重启 | `test_monitor_degradation.py::test_bounded_restart` |
| I40 | 单租户监控故障不影响其他租户；监控越权访问触发 L4 | `test_monitor_degradation.py::test_tenant_isolation_and_quarantine` |

#### 19.12.9 故障矩阵

| 故障 | 状态 | 主任务 | 处理 |
|---|---|---|---|
| 单 job 超时 | L1 | 无影响 | 丢弃该 job，记超时 |
| 进程 crash | L3 → 重启 | 无影响 | 心跳检测 + 退避重启 |
| 输出 schema 非法 | L2 | 无影响 | 丢弃输出，记 `schema_invalid` |
| 死循环/资源失控 | L4 | 无影响 | kill + 退避；禁止其输出进入任何账本 |
| DB 锁/不可用 | L1 | 无影响 | ring buffer；丢最旧 |
| LLM 5xx | L1 | 无影响 | 规则版 detector；不重试阻塞 |
| 队列打满 | L1 | 无影响 | 按优先级丢弃；记 `queue_full` |
| 时钟偏斜 | L1 | 无影响 | 服务端时间；记 `clock_skew` |
| 越权写 / 跨租户 | L4 | 无影响 | 拒绝 + kill + 告警 |
| 联锁依赖监控 | — | 禁止 | 启动时静态检查 I37 |

测试基线复用现有混沌用例（[test_chaos_fault_injection.py](file:///workspace/agent/tests/test_chaos_fault_injection.py) 的 LLM 超时 / 磁盘满 / 损坏 DB 模式），新增 kill -9、schema 污染、队列洪峰、时钟偏斜四类。

#### 19.12.10 指标（补）

| 指标 | 目标 |
|---|---|
| monitor availability（按租户） | ≥ 99% |
| MTTR（从 L3 到 L0） | ≤ 2 分钟 |
| unmonitored session rate | ≤ 1% |
| dropped findings ratio | ≤ 5% |
| task p99 overhead | ≤ 5% |
| degradation visibility | 100%（无静默降级） |
| false pause during degradation | 0（降级不等于乱停） |

#### 19.12.11 反模式

- 把监控写成任务循环里的一个 `await`。
- 监控持有记忆/知识库写锁。
- 监控失败时静默返回「一切正常」。
- 监控是安全约束的唯一执行点。
- 全局单例监控跨租户共享。
- 无退避的自动重启，形成 crash loop。
- 降级时直接停任务，而不是降自主。

---

## 20. 与世界模型的结合：预测器、对账器、风险信号

### 20.1 先承认仓库里有四个「世界」，且不合并

[world_model_inventory](file:///workspace/agent/huginn/research/law_model.py#L114-L167) 已经明确：世界模型不是单一范式，强行合并会同时破坏两者的本职。本 spec 沿用这一治理立场，把四个实现分别接到不同账本：

| 实现 | worldview | 能做什么 | 不能做什么 | 接哪本账 |
|---|---|---|---|---|
| [research.law_model](file:///workspace/agent/huginn/research/law_model.py#L231-L342) | `physics_causal` | `law()` 方程、`predict()` 前向、`reconcile()` 数值对账、`ModelBasedPlanner` rollout | 不能替代真实执行；不能授权动作 | 认知账（预测/对账）、经验账（真值） |
| [security.world_state](file:///workspace/agent/huginn/security/world_state.py#L274-L421) | `latent_transition` | `StateEstimator` 状态估计、`ForwardPredictor.step()` 前向投影、`observe()` 算 prediction_error | 隐态没有物理语义，不能拿来论证物理结论 | 认知账（预测）、监控环（漂移信号） |
| [security.world_model](file:///workspace/agent/huginn/security/world_model.py#L85-L121) | `physics_causal` | `check_constraints()` 预检、`infer_inverse()` 逆动作、可逆性判断 | 不能预测科学量；不能替代规划 | 规范账（风险/可逆性）、经验账 |
| PyBullet backend | `physics_causal` | 刚体动力学 rollout，作为 reconcile 的仿真真值 | 只是仿真真值，不是现实真值 | 经验账（ground truth） |

硬规则：**`worldview` 是证据标签的一部分**。不同 worldview 的预测不得在同一认识论 sheaf 上直接粘合；跨 worldview 的冲突记为 `worldview_mismatch`，不是 `contradiction`。

### 20.2 世界模型在 spec 里的三种角色

| 角色 | 调用面 | 产出 | 权威类别 | 消费方 |
|---|---|---|---|---|
| **预测器** | `predict()` / `ForwardPredictor.step()` / `ModelBasedPlanner.rollout()` | `WorldModelPrediction`（状态、动作、预测值、不确定度、epoch） | `hypothesis`/`derived` | PMK 认识论 sheaf、Dreamer、Analyst |
| **对账器** | `reconcile()` / `ForwardPredictor.observe()` | `ReconcileRecord`（borne_out / falsified / insufficient、误差、真值来源） | 对账成功才升 `direct` | 认知账、经验账、监控环 |
| **风险信号** | `check_constraints()` / `infer_inverse()` | 约束违规列表、可逆/不可逆 | 硬约束输入 | System3 准入、规范账 |

三条边界：

1. **预测不是事实**：`WorldModelPrediction` 默认 `evidential_authority ≤ 0.3`，`influence cap ≤ 0.4`；只有 `reconcile` 成功才可升格。
2. **对账不是授权**：即使预测完全命中，也只证明「模型预测对了」，不证明「动作被允许」。
3. **风险不是裁决**：`check_constraints` 只提供风险信号；是否执行由 System3 + 规范账决定。

### 20.3 权威晋升链

```
prediction (hypothesis, evidential 0.2)
   --reconcile borne_out--> reconciled (derived, evidential 0.6)
   --independent replication + ground truth--> replicated (direct, evidential 0.9)
   --reconcile falsified--> contested/refuted (evidential 0.1, 保留为负证据)
   --insufficient / 缺指标--> insufficient (不得当通过)
```

规则：

- `reconcile` 必须区分三态：`borne_out / falsified / insufficient`。**当前实现 [reconcile](file:///workspace/agent/huginn/research/law_model.py#L321-L342) 在指标缺失时会返回 `borne_out=True`（errors/mismatch 为空）——这是必须修的语义漏洞**：至少一个指标参与比较，且全部参与指标在容差内，才算 borne_out；否则 insufficient。
- `actual` 必须来自真实执行或可重放经验，带 `source_id`；禁止合成 actual。
- `falsified` 不是失败，是发现：进入认知账的 `contested`，并作为负证据保留。
- `replicated` 需要独立 run / 独立后端 / 独立数据，不能用同一 prediction 重复对账充数。

### 20.4 与 PMK 双 sheaf 的结合

认识论 sheaf 增加一个 `world_model` stalk：

```python
@dataclass
class WorldModelStance:
    world_model_id: str
    worldview: str              # physics_causal / latent_transition / ...
    domain: str
    contract_version: str
    predicted: dict[str, float]
    uncertainty: dict[str, float]
    epoch: int
    created_at: str
```

Restriction map 先查兼容性，再查 subject 对齐：

1. `tenant_id` / `domain` / `contract_version` 不兼容 → 不建边，记 `worldview_mismatch` 或 `contract_violation`。
2. 兼容 → 按 subject 建边，比较 predicted 与 observed；偏差超过容差 → `prediction_error`。
3. 同 worldview 同 domain 的冲突才是认识论 `contradiction`；跨 worldview 的差异不是。

这样 PMK 的 `timeseries` 路从「关键词 trend 匹配」升级为「世界模型预测 vs 观测」的结构化对账；原来的关键词逻辑保留为无世界模型时的降级路径。

### 20.5 与 System3 准入的结合

- `check_constraints()` 返回非空 → `deny` 或 `degrade`，并给出具体违规项。
- `infer_inverse()` 返回 `None`（不可逆）→ 提高授权要求：更窄 scope、更短 TTL、显式人工确认；不是直接拒绝。
- 高风险 + 高不确定度预测 → `ask`，required_acts 写明「授权 scope」。
- 世界模型不可用 → 只失去预测信号，不阻断任务；`trust=partial`，不是 `blind`。
- 世界模型给出的预测**永远不能**进入 `authorize` 的判定依据。

### 20.6 与监控环的结合

`ForwardPredictor` 已经有现成的两个信号：

- `last_prediction_error`（[world_state.py#L345](file:///workspace/agent/huginn/security/world_state.py#L345)）：预测 vs 实测的 RMS 误差。
- `prediction_error_to_reward`（[world_state.py#L570](file:///workspace/agent/huginn/security/world_state.py#L570)）：误差到连续奖励的映射。

接入方式：

1. **漂移前置信号**：`prediction_error` 连续 N 步上升 → 写 `drift_events`，`detector=prediction_error`，级别先 `review`；超过阈值再 `intervene`。它比文本反思更早、更客观。
2. **惊奇排序**：prediction_error 作为贝叶斯惊奇的廉价代理，喂给 Dreamer 的惊奇探针；高误差 finding 优先诊断。
3. **幽灵防御**：复用 `epoch` / `restored` 语义——旧 epoch 的预测不得当作当前预测消费；`restored=True` 的预测只能作为历史痕迹，不能进入当前 `AdmissionView`。
4. **监控降级**：世界模型不可用 → `trust=partial`；预测过期 → 丢弃并记 `stale_prediction`，不得用旧预测补位。

### 20.7 与经验账的结合

给 [ExecutableExperience](file:///workspace/agent/huginn/research/experience_archive.py#L43-L70) 增加字段：

```python
world_model_id: str
worldview: str
law: str
contract_version: str
prediction_error: float | None
reconcile_status: Literal["borne_out", "falsified", "insufficient"]
ground_truth_source: str          # tool_run_id / pybullet_run_id / replay_id
```

规则：

- 只有 `reconcile_status=borne_out` 且 `ground_truth_source` 可回查的经验，才允许 `evidential_authority ≥ 0.6`。
- 只有预测、没有真实对账的经验，最高 `derived`。
- 检索按 `worldview + domain + contract_version` 过滤；单位/有效域不兼容的经验不得混用。
- replay 失败 → `degraded`，不伪造 ground truth。

### 20.8 与梦境周期的结合

- **Dreamer**：用 `ModelBasedPlanner.rollout()` 生成反事实探针；`prediction_error` 高或 `reconcile=falsified` 的区域作为高惊奇 finding。
- **Analyst**：把 `falsified` 归因为 `law_mismatch / contract_violation / missing_link / stale_model` 之一；用 rollout 做反事实：「若修正参数/补一条边，预测是否通过？」
- **Consolidator**：可以提出世界模型修订（参数、定律、有效域），但**只能提案**；修订走 meta_improver 的 GREEN 采纳门，不能由梦境直接改模型。
- 梦中新定律默认 `hypothesis` 权威；至少一次独立 `reconcile` 才能升 `derived`。

### 20.9 数据契约（补）

```sql
CREATE TABLE world_model_predictions (
  prediction_id   TEXT PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  world_model_id  TEXT NOT NULL,
  worldview       TEXT NOT NULL,
  domain          TEXT NOT NULL,
  contract_version TEXT NOT NULL,
  state_before_digest TEXT NOT NULL,
  action_digest   TEXT NOT NULL,
  predicted_json  TEXT NOT NULL,
  uncertainty_json TEXT,
  epoch           INTEGER NOT NULL,
  restored        INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL
);

CREATE TABLE reconcile_records (
  reconcile_id    TEXT PRIMARY KEY,
  prediction_id   TEXT NOT NULL,
  tenant_id       TEXT NOT NULL,
  actual_source   TEXT NOT NULL,   -- tool_run_id / pybullet_run_id / replay_id
  actual_json     TEXT NOT NULL,
  tol             REAL NOT NULL,
  errors_json     TEXT NOT NULL,
  status          TEXT NOT NULL,   -- borne_out|falsified|insufficient
  prediction_error REAL,
  worldview_consistent INTEGER NOT NULL,
  created_at      TEXT NOT NULL
);
```

### 20.10 不变量（补 I41–I50）

| # | 不变量 | 测试 |
|---|---|---|
| I41 | 预测默认 `hypothesis/derived`，未经 reconcile 不得作为 `direct` 证据 | `test_world_model_binding.py::test_prediction_not_direct` |
| I42 | 预测必须带 `worldview/domain/contract_version`；缺失即 quarantine | `test_world_model_binding.py::test_required_labels` |
| I43 | `actual` 必须来自真实执行/可重放经验并带 `source_id`；合成 actual 禁止 | `test_reconcile.py::test_actual_provenance` |
| I44 | 世界模型不得授权动作；`authorize` 只能由规范账写 | `test_world_model_binding.py::test_no_authority` |
| I45 | 不可逆动作（`infer_inverse=None`）必须提高授权 scope/TTL 或人工确认 | `test_admission_gate.py::test_irreversible_requires_scope` |
| I46 | `prediction_error` 超阈值必须写 `drift_events`；禁止静默 | `test_drift_gate.py::test_prediction_error_signal` |
| I47 | 世界模型修订必须过采纳门；监控/梦境不得直接改模型 | `test_world_model_binding.py::test_revision_gate` |
| I48 | 世界模型不可用只降预测能力，不阻断任务 | `test_world_model_binding.py::test_availability_degrade` |
| I49 | 跨 worldview 预测不得粘合；冲突记为 `worldview_mismatch` | `test_pmk_sheaves.py::test_worldview_mismatch` |
| I50 | 旧 epoch / `restored` 预测不得作为当前预测消费 | `test_world_state.py::test_ghost_prediction_blocked` |

### 20.11 指标（补）

| 指标 | 定义 | 目标 |
|---|---|---|
| prediction calibration (ECE) | 预测不确定度与命中率的校准误差 | ≤ 0.10 |
| reconcile pass rate | `borne_out / (borne_out+falsified)` | 按域设基线，不跨域比较 |
| prediction-error lead time | 预测误差上升领先真实漂移的步数 | ≥ 2 步 |
| worldview mismatch rate | 跨 worldview 误粘合次数 | 0 |
| contract violation rate | 单位/有效域冲突次数 | 0 |
| irreversible action authorization rate | 不可逆动作中显式授权占比 | 100% |
| synthetic actual rate | 合成 actual 占比 | 0 |
| stale prediction consumption | 旧 epoch 预测被消费次数 | 0 |

### 20.12 路线补丁

| 阶段 | 内容 | 出口 |
|---|---|---|
| P1-f | 预测/对账记录落库；证据补 `worldview/domain/contract_version` | I41–I43 绿 |
| P2-g | PMK 增加 `world_model` stalk；`prediction_error` 接漂移告警 | I46、I49–I50 绿 |
| P3-d | reconcile 升格链 + replay 真值 + Analyst 反事实 rollout | I44–I45、I47–I48 绿 |

### 20.13 反模式

- 把 `predict()` 的输出直接写进报告当事实。
- 把 `latent_transition` 的隐态预测当作物理结论。
- 用同一个 `reconcile` 结果给多个预测充数。
- 指标缺失时仍返回 `borne_out=True`。
- 用世界模型决定「能不能执行」。
- 把 PyBullet 仿真真值当成现实真值。
- 旧 epoch 的预测没有失效标记就继续消费。
- 跨 worldview 的差异当成「矛盾」去 pause。

### 20.14 诚实边界

- `LawModel` 是域内模型，不存在「通用世界模型」；跨域只能借结构，不能借参数。
- `reconcile` 只校验声明的 metrics，不校验全状态；未覆盖的维度是「未检查」，不是「通过」。
- `ForwardPredictor` 的隐态转移没有物理语义，只能用于预测误差/漂移信号，不能用于物理论证。
- PyBullet 是仿真真值，只在仿真契约内成立。
- 世界模型回答「会发生什么」，不回答「是否被允许」；后者只属于规范账。

---

## 21. 实施进度（滚动）

已完成（有测试覆盖）：

- [x] `reconcile` 三态：`borne_out / falsified / insufficient`；缺失指标不再冒充通过（§20.3，I41–I43 基础）。
- [x] 规范账：追加式 SQLite 事件、确定性投影、TTL、revoke 支配、actor 能力、租户隔离（§7.1，I1–I4、I9、I11–I12）。
- [x] 准入调度器：纯函数 `admit()`、fail-closed、最小 `required_acts`、claim 降级、persona 边界（§8，I1–I2、I10）。
- [x] 会话接线：`SessionGrounding`；`request_confirmation` 写 `propose`，`clear_confirmation(accepted=...)` 写 `accept/reject`（§7.1）。
- [x] 生产路径：计划确认写 `accept/reject`；显式工具审批写 `authorize/reject`（带 TTL 与单次限额）。
- [x] 工具执行路径 shadow 准入；`HUGINN_NORMATIVE_ENFORCE=1` 时高风险动作缺授权才阻断（I1、I10）。
- [x] PMK 默认开启（`HUGINN_PMK_INJECT=1`），`HUGINN_PMK_MODE=shadow` 默认评估不暂停；`oppose+oppose` 误报已修。
- [x] 记忆双层治理纯函数：保留闸 / 影响闸 / Hygiene Gate / 上下文装配（§18.1–18.2、18.6，I13–I15、I17、I20–I21 基础）。

待完成（按依赖顺序）：

- [ ] persona 站位拆分：persona 从认识论 sheaf 移出，改为身份/规范约束；跨 worldview 与人格冲突改走规范 H¹。
- [ ] PMK `enforce` 接线：LLM 语义判定接入、误报率达标后再切默认。
- [ ] 记忆状态事件账（`memory_state_events`）与 `recall_for_prompt` 的 `assemble_context` 接入。
- [ ] 梦境周期（SleepClock / Dreamer / Analyst / Consolidator）与写入侧晋升门。
- [ ] VIGIL 兄弟监控运行时与 L0–L4 降级状态机。
- [ ] 世界模型 binding：`world_model` stalk、reconcile 升格链、prediction_error 漂移信号。








