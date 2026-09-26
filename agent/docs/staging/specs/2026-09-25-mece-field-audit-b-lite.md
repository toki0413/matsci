# Spec: MECE 字段级审计面合并（B-lite）—— 传输轴 → 角色关系轴

**日期**: 2026-09-25
**milestone**: 未定（待评审后再挂）
**状态**: **已落地** —— 第 1/2（裁剪为 15/16 近重复面）/2.5/4/5/6 步已落地；第 3 步（14/17 迁移）明确不做（见 §实施记录）
**前置**: 第十三～十七面（响应结构 / WS 请求负载 / SSE 事件负载 / WS 事件负载 / HTTP 请求字段）已各自落地。

---

## 目标（一句话）

把第十三～十七面里**逐字同构的机器**（迭代 → 门控 → 集合差 → 覆盖面计数 → 分诊盖章 → contract 装配 → markdown 脚手架）抽成一个参数化引擎，使每个"边界"降级为**薄适配层（配置 + 解析器 + 面特有渲染）**；并**纯新增**一层"按角色关系分组的汇总视图"，把所有字段级面（第十二～十七面的违例）收敛成**一次**人工确认，而不是每面各确认一遍。

行为必须**逐字不变**：本设计是**去重重构 + 加汇总**，不是改变任何一条判定。

---

## 背景证据（实测，非推断）

近五面子真实仓的边际产出：

| 面 | 机制 | 当前分诊表条目 / 历史发现 |
|---|---|---|
| 12 请求负载面 | 前端漏发后端必填 | 0（历史发现 2，已修） |
| 13 响应结构面 | 前端声明读 vs 后端 return | 0（历史发现 1，已修 `/viewer3d/load`） |
| 14 WS 请求负载面 | 模型声明 / handler 读 / 前端发 | 0 |
| 15 SSE 事件负载面 | 帧 payload 顶层键 vs 前端读 | 0 |
| 16 WS 事件负载面 | 帧 payload 顶层键 vs 前端读 | 0 |
| 17 HTTP 请求字段面 | 模型声明 / handler 读 / 前端发 | 0 |

（实测：六张分诊表 `_PAYLOAD_CONFIRMED_VIOLATIONS` / `_RESP_CONFIRMED_VIOLATIONS` / `_WS_PAYLOAD_CONFIRMED` / `_SSE_PAYLOAD_CONFIRMED` / `_WS_EV_PAYLOAD_CONFIRMED` / `_HTTP_FIELD_CONFIRMED` 当前**条目数全为 0**，六面 `untriaged` 亦为空——即这些轴上真实仓已一致，边际缺陷产出已归零。）

同时五面在**机制**上并不互斥：都是"某边界两侧字段集一致"这**同一个不变量**套在不同传输上（HTTP body / WS 入站 / SSE 帧 / WS 出站帧）。按传输切片导致每片重写一遍同样的机器：

- 每个面的**分诊三联**（`_X_KIND_DOC` + `_X_TRIAGE_DOC` + `_X_CONFIRMED` + `_X_triage()` + `_X_violation_mark()`）是 ~30 行的**同构模板**，五份 ≈ 150 行（文案面特有，必须保留；但**结构**可归一）。
- 每个面的 **contract 字典装配** + **render 脚手架**（"违例类型 / 覆盖 / 违例表 / 静态核对覆盖面 / 诚实边界"章节）另各 ~40–60 行同构。
- 各面 `build_*` 的**主循环**结构一致：`for obs → 单元键不在权威面则 skip_frame → 形状开放则 skip_shape → 计数 checked → 集合差 → 造违例`。

`_sse_payload_reads` 与 `_ws_ev_payload_reads` 的返回结构（`{channel, frame, field, rel, line}`）亦高度一致。

---

## 现状骨架盘点（五面 → 角色映射，全部实测自 `huginn/cli/contract_audit.py`）

| 面 | 权威面（authority） | 观测面（observations） | 硬方向 | 单元键 | 覆盖面计数 |
|---|---|---|---|---|---|
| 13 `build_response_contract` | `_resp_backend_shapes` → `{keys, closed}` keyed by `(method, endpoint)` | `_resp_scan_frontend`（前端 `api.*<T>` 声明字段，批式） | 声明读 ⊄ 后端返 | `(method, endpoint)` | checked / skip_shape / skip_decl / skip_ambiguous |
| 14 `build_ws_payload_contract` | `_ws_payload_model_fields`（`WSMessage` 字段集，**全局集**） | ①`_ws_payload_handlers`（handler 读 `msg.<f>`）②`_ws_payload_scan_sends`（前端发送键） | ①handler 读 ⊄ 模型声明 ②前端发 ⊄ 模型声明 | `(inbound_type,)` / `(inbound_type,)` | handlers_resolved / sends_static / sends_unknown |
| 15 `build_sse_payload_contract` | `_sse_frame_emitters`+`_sse_progress_shapes`+`_sse_event_bus_shape` → keyed by `(channel, frame)` | `_sse_payload_reads`（前端 `JSON.parse(e.data)` 顶层读） | 前端读 ⊄ 后端发 | `(channel, frame)` | checked / skip_frame / skip_shape |
| 16 `build_ws_ev_payload_contract` | `_ws_ev_payload_shapes` → keyed by `(channel, frame)` | `_ws_ev_payload_reads`（前端 type 分支读 `data.<f>`） | 前端读 ⊄ 后端发 | `(channel, frame)` | checked / skip_frame / skip_shape |
| 17 `build_http_field_contract` | `_http_field_models`（请求体模型字段集，**逐模型**；body-dict 无权威） | ①`_http_field_body_reads`（handler 读 `body.<f>`/`body["k"]`）②`_http_field_sends`（前端 body 键） | ①handler 读 ⊄ 模型声明 ②前端发 ⊄ 模型声明 ③body-dict handler 下标读 ⊄ 前端发 | `(method, endpoint)` | handler_* / fe_* / dict_* |

**结论**：13/15/16 是"**单权威 + 单元键 + 字段隶属**"的**同构三元组**（重复度最高）；14/17 是"**多方向**"，方向数不同但每个方向都落回同一套"权威字段集 vs 观测字段集"的集合运算。故可统一为**一个 Check 原语**，而**不是**三个独立不变量（早先"3 条不变量"的说法不准确，见下）。

---

## 统一抽象（引擎，设计级接口）

按**角色关系**而非传输建模。核心是 `Check` 原语：

```python
@dataclass(frozen=True)
class Shape:
    keys: frozenset[str]
    closed: bool                       # False = 形状开放 (含 ** / 变量 / Response 对象 …)

@dataclass(frozen=True)
class Obs:
    key: tuple                         # 单元键: ("POST","/x") | ("agent","text_delta") | ("WSMessage",)
    fields: frozenset[str]             # 该观测点读/写/发的字段集
    rel: str
    line: int
    open: bool = False                 # 该观测点自身形状开放 → 跳过
    meta: Mapping[str, Any] = field(default_factory=dict)   # 面特有 (如 detail 文案)

@dataclass(frozen=True)
class Check:
    kind: str                          # 违例 kind 标签 (面内唯一)
    authority: Callable[[Path], Mapping[tuple, Shape] | Shape]   # 键控 dict 或全局单 Shape
    observations: Callable[[Path, Path], list[Obs]]
    relation: str                      # "read_not_in_authority" | "emit_not_in_authority"
                                       # | "consumed_not_in_authority" | "required_not_consumed"
    coverage_ok: str                   # 通过计数名
    coverage_skip: tuple[str, ...]     # 跳过计数名 (顺序即优先级)
```

**唯一的判定内核**（所有 Check 共用）：

```python
def _run_check(chk: Check, root: Path, frontend: Path) -> tuple[list[dict], dict[str, int]]:
    authority = chk.authority(root)
    cov = {chk.coverage_ok: 0, **{n: 0 for n in chk.coverage_skip}}
    out: list[dict] = []
    for o in chk.observations(root, frontend):
        sh = _lookup(authority, o.key)
        if sh is None:            cov[chk.coverage_skip[0]] += 1; continue   # 单元不在权威面
        if not sh.closed:         cov[chk.coverage_skip[1]] += 1; continue   # 权威形状开放
        if o.open:                cov[chk.coverage_skip[-1]] += 1; continue  # 观测形状开放
        cov[chk.coverage_ok] += 1
        missing = _relation_diff(chk.relation, o.fields, sh.keys)
        out += [ {"kind": chk.kind, "key": o.key, "field": f,
                  "rel": o.rel, "line": o.line, **o.meta} for f in sorted(missing) ]
    return out, cov
```

**边界装配**（面级）：

```python
@dataclass
class BoundarySpec:
    name: str
    checks: list[Check]
    triage: Callable[..., tuple[str, str] | None]     # 面分诊表查询
    kind_doc: Mapping[str, str]                       # 面文案 (保留)
    triage_doc: Mapping[str, str]
    extra: Callable[[Path, Path], dict]                # 面特有 contract 字段 (channels/rows/candidates/…)
    render_head: Callable[[dict], list[str]]           # 面标题 + 说明 + 覆盖 + 违例表
    render_extra: Callable[[dict], list[str]]          # 面特有表格
    render_tail: Callable[[dict], list[str]]           # 面诚实边界

def build_boundary(spec: BoundarySpec, root: Path | None, frontend: Path | None) -> dict:
    # 统一: 跑所有 Check → 汇总 violations/untriaged/kind_counts/coverage → 盖章分诊
    # 面特有字段由 spec.extra() 注入; contract 的通用四键由引擎保证
```

**引擎负责（消除重复）**：迭代+门控、集合差、覆盖面计数、`violations`/`untriaged`/`kind_counts`/`coverage` 四键装配、分诊盖章与 `violation_mark` 模板、render 的"覆盖/违例表/诚实边界"骨架。
**适配层保留（面特有，不可归一）**：各 `authority` / `observations` **解析器**（解析对象是 TS 泛型、SSE `e.data`、WS type 分支、`WSMessage` 注解、请求体注解——语言与形状各异），以及各面**文案**与 **`extra`/`render_*`** 内容。

> 说明：解析器是各面代码量的主体，抽引擎**不会**消掉它们；引擎消掉的是围着它们的那圈**同构机器**。这是本设计对收益的**诚实上限**。

---

## 不变量汇总视图（纯新增，非破坏）

不把它叫"三条不变量"——因为硬方向实为**角色对之间的包含关系**，且 12/14/17 的方向与 13/15/16 不同。汇总按**关系**分组，一次罗列全部字段级面（12～17）的违例：

| 关系标签 | 含义 | 归属 kind |
|---|---|---|
| `R1 消费⊆权威` | 读/下标取用的字段，权威（模型/后端生产）必须声明 | resp `missing-field`；sse/ws_ev `read-undeclared`；ws_payload `handler-undeclared`；http_field `handler-undeclared` |
| `R2 生产⊆权威` | 生产方发出的字段，权威必须声明 | ws_payload `fe-undeclared`；http_field `fe-undeclared` |
| `R3 权威必填⊆送达` | 权威标必填的字段，生产方必须送达 | payload（12）漏发必填 |
| `R4 下游下标读⊆上游发送` | body-dict 端点下标读的键，上游调用必须发 | http_field `dict-key-unsent` |

新增 `render_field_rollup_markdown(snap)`：输出一张 `关系 × 面` 的矩阵 + 每关系的违例地点清单。它**只读**各面已有的 `violations`，**不新增判定、不改任何违例身份**。插入 `render_mece_markdown` 正文，作为"字段级面"章节的抬头汇总。人工确认因此从"逐面过 17 张分诊表"收敛为"过这一张矩阵，缺漏再下钻到面"。

---

## 落地分期（每步独立可回滚）

1. **冻结黄金快照（护栏先行）**：对五面各 `build_*_contract()` 在**真实仓 + 现有合成树**上跑一次，把 contract 字典（去掉排序无关的 set→list 顺序）序列化为 golden JSON 存 `tests/golden/`；新增 `test_field_audit_golden_parity` 断言现实现 == golden。**此步不改任何产品代码**，先把"不许变"钉死。
2. **抽引擎原语**：引入 `Shape/Obs/Check/BoundarySpec/_run_check/build_boundary`；先让 13/15/16（同构三元组）改为薄适配层，跑第 1 步 parity。
3. **迁移 14/17**：多方向面改用多个 `Check` + 面特有 `extra/render_*`，跑 parity。
4. **去重分诊三联**：抽 `_TriageTrio`（每个仍传自己的文案表），保持每面公开符号名不变（外部测试/文档引用 `_HTTP_FIELD_CONFIRMED` 等，需保留别名）。
5. **加汇总视图**：`render_field_rollup_markdown` + INDEX/文档登记。
6. **冻结门禁**：把 `contract_audit --check --baseline …`（棘轮门）挂 CI（见 §门禁），**之后不再加面**。

---

## 实施记录（2026-09-26 收口）

| 步 | 状态 | 落点 |
|---|---|---|
| 1 冻结黄金快照 | ✅ | `tests/golden/field_audit/{response,ws_payload,sse_payload,ws_ev_payload,http_field}.json` + `test_field_audit_golden_parity`（`HUGINN_REGEN_FIELD_GOLDEN=1` 重生成） |
| 2 抽引擎原语 | **裁剪** ⚠ | 仅抽第十五/十六面（逐字近重复的一对）→ `_frame_payload_contract`，两个 `build_*` 降为薄适配层。**未**做 `Shape/Obs/Check/BoundarySpec/_run_check` 全量原语，**未**并入 13/14/17 |
| 2.5 reads 共同抽取 | ✅ | `_fe_ts_files` / `_fe_field_read_rows`，供 `_sse_payload_reads` 与 `_ws_ev_payload_reads` 共用 |
| 3 迁移 14/17 | ❌ 未做 | 多方向面仍各自成文（多方向 + 面特有 `extra`，归一化收益低于复杂度） |
| 4 去重分诊三联 | ✅ | `_TriageTrio`（`kind_doc` + `key` + `vargs`）统一七面的查表 / `stamp` / `mark`；各面只传键构造函数，面文案仍逐字差异化。全部公开符号（`_X_KIND_DOC` / `_X_TRIAGE_DOC` / `_X_CONFIRMED*` / `_X_triage` / `_X_violation_mark`）保留为指向实例成员的别名，外部引用不变 |
| 5 加汇总视图 | ✅ | `render_field_rollup_markdown` + `_ROLLUP_*` 常量；插入 `render_mece_markdown` 字段级面章节之前 |
| 6 冻结门禁 | ✅ | CI 挂 `--check --baseline tests/golden/mece_findings_baseline.txt`（棘轮门，见 §门禁）；`--update-baseline` 重生成基线 |

**第 2 步裁剪原因（实测）**：设计期的 `_run_check` 假定**每字段一条违例**，但第十三面（响应结构）实为**每端点一条**（违例体带 `declared`/`produced`/`missing` 列表），且把端点解析与歧义跳过编进了主循环 —— 与 15/16 只是"看起来同构"，强行并入需要给引擎加"按单元聚合"与"面特有门控"两个变体，接口反而比现状更宽。第十五/十六面才是真正的逐字孪生（仅解析器、文案、分诊表三处不同），故只抽这一对：净 −38 行，行为由 golden parity 逐字兜底。收益诚实地有上限：解析器才是各面代码量主体。

**后续可选**（未批准，未做）：仅剩第 3 步（14/17 多方向面并入引擎）。

---

## 验证（如何知道没改坏）

- **parity**：第 1 步 golden JSON 是硬闸——重构后每面 contract 必须**逐字段相等**（含 `violations` 的 kind/key/field/rel/line、`coverage`、面特有字段）。任一不等即回滚该面。
- **文档守卫**：既有 `test_mece_audit_doc_not_drifted`（见 `tests/test_contract_audit.py:2199`）要求 `docs/mece-audit.md` == `render_mece_markdown(build_mece_snapshot())`；重构后重新生成并人工 diff，确认**只有排序/格式**变化（若语义无变，diff 应为空或纯汇总新增）。收口时该用例在 HEAD 上即**预存失败**（文档停留旧源码的词汇/工具/钩子站点集），已重生成补齐（54 ± 行，字段级面章节与「发现汇总」逐字不变）。
- **回归**：`pytest tests/test_contract_audit.py`（现 **137 passed**）须全绿；`ruff check` 零告警。
- **棘轮门**：`contract_audit --update-baseline` 生成/刷新 `tests/golden/mece_findings_baseline.txt`（47 项）；`--check --baseline` 对基线内项 exit 0、对基线外新发现 exit 1（已手测：删基线一条 → 只报该条）。
- **合成树覆盖**：五面均有合成树用例（handler-undeclared / fe-undeclared / dict-key-unsent / extra=allow / 形状开放跳过 等），迁移后必须原样通过。

---

## 门禁（C 的落点）

`contract_audit --check` 原有口径是**任一 MECE 发现**即 `exit 1`（`find_issues(snap)` 非空）。实测该口径在真实仓返回 **47 项**，且多为「同轴惩罚 / 跨模块同名 / 词表漂移」类**候选登记**（工具自述"只提示不判死"，`test_find_issues_reports_expected_categories` 等**断言其存在**）—— 故「零发现」硬门**不可行**。落点改为**棘轮门（ratchet）**：

- `--baseline FILE`：`--check --baseline …` 只对**基线外的新发现** `exit 1`；基线内既有项不拦。修好旧项**无需**改基线（基线里多余条目被忽略）。
- `--update-baseline`：把当前发现重生成到基线（缺省 `tests/golden/mece_findings_baseline.txt`）。
- 无 `--baseline` 时 `--check` 行为**不变**（绝对门，有发现即失败），向后兼容。
- CI（`.github/workflows/ci.yml`）挂 `python -m huginn.cli.contract_audit --check --baseline tests/golden/mece_findings_baseline.txt`，置于重型套件之前的 guard 段；测试侧同义门 `test_findings_baseline_covers_current_findings`。
- 字段级接线契约另由各面 `test_*_real_repo_no_untriaged_violations`（六处）在主 pytest job 把关；二者互补 —— 前者拦**全轴新增**（含待分诊、跨模块同名、词表漂移），后者守**字段级不变量**。

> 诚实说明：设计期判断"无需新增代码"（假定零发现），实测**证伪** —— 47 项既有候选登记使绝对门不可用，故必须引入基线 artifact 与棘轮语义。

---

## 诚实边界 / 本设计**不**统一的部分

- **解析器不归一**：五种 `authority`/`observations` 各自解析不同语言/形状，是各自代码量主体；本设计**不**合并它们（强行合并会做出一个什么都懂的巨型解析器，反而不如现在清晰）。
- **面特有 `extra` 与渲染不归一**：`channels`/`frame_reads`/`zero_read`/`dead_fields`/`rows`/`candidates_*` 语义不同，保留在适配层。
- **文案不归一**：各面 `KIND_DOC`/`TRIAGE_DOC` 是给人看的领域说明，必须保留逐字差异化。
- **不新增审计面**：本设计明确**止于合并**，第十八面及以后不再开——除非出现**有真实可达性/爆炸半径**的新缺陷类（届时按 §决策 重新评估，而非惯性加面）。
- **`_sse_payload_reads` 与 `_ws_ev_payload_reads` 的共同抽取**只作为**可选**第 2.5 步：两者结构相近但触发语法不同（`JSON.parse(e.data)` vs type 分支），先不动，若迁移中发现能安全共享再议。

---

## 决策记录（ADR）

- **采用 B-lite（保留现有 17 面报告轴 + 新增汇总），否决 B-full（把报告轴改成 3 段不变量）**：既然要走 C（冻结加面），B-full 需把 17 张分诊表身份、doc-drift 契约、INDEX 全翻一遍，长期收益（新面更干净）在"不再加面"前提下**无法兑现**，收益/风险比最差。B-lite 非破坏、可回滚，且拿到"一次确认"的实益。
- **护栏优先**：先落 golden parity，再动产品代码；没有 parity 不重构。
- **止于合并**：合并完成后不加新面，精力转有真实可达性/爆炸半径的缺陷类或功能。

---

## 范围（out of scope）

- 不改任何**判定逻辑**与**违例身份**（kind/key/field 语义不变）。
- 不改 `docs/mece-audit.md` 的既有面章节语义（仅重生成 + 新增汇总章节）。
- 不新增审计面；不改运行时（本工具是静态扫描，纯离线）。
- 不动 `tests/test_contract_audit.py` 既有用例的断言（只**新增** parity 用例）。

---

## 评审结论

1. **先落 golden parity 护栏** —— 接受，已落地（第 1 步）。
2. 第 2.5 步（`sse/ws_ev` reads 共同抽取）—— 做，已落地（`_fe_ts_files` / `_fe_field_read_rows`）。
3. 汇总视图关系标签 **R1–R4 + R0 兜底** —— 接受，已落地（`_ROLLUP_*`）。
4. 第 2 步范围 —— 裁剪为「只抽 15/16 近重复面」；第 4 步（分诊三联归一）与第 6 步（棘轮门挂 CI）后续落地，第 3 步（14/17 迁移）明确不做。