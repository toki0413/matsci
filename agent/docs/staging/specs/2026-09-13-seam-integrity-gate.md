# Spec: 共享缝防腐蚀边界门禁（发现②治理）

**日期**: 2026-09-13
**milestone**: M1（见 docs/ROADMAP.md，若已建）

## 目标（一句话）
给 `core_types`（最深共享内核）与 `config`（根配置）固化**防腐蚀/依赖方向边界**：业务层可以依赖它们，但它们不得反向依赖业务层——把这条纪律从"纸面约定"变成 CI 门禁，防止共享内核随时间变成杂物间（腐化风险②的 remedy）。

## 背景证据（实测，非推断）
- `huginn/core_types.py`：实质是合法的跨层契约内核（`ToolResult/ToolContext/PermissionMode/RiskLevel/ErrorKind/AgentMessage/CostEstimate/BudgetPolicy/HandleType/PermissionResult`），且**只 import 标准库**（contextvars/dataclasses/datetime/enum/typing），**零 `huginn.*` 反向依赖**。
- `huginn/config.py`：只 import 基建（`huginn.crypto/checkpointer/models.router/models.registry/config_integrity/feature_flags`），**不依赖业务/应用层**。
- 现有 `tests/test_arch_*` 覆盖单据点网关/依赖 allowlist/学术↔计算分层，但**没有**专门锁"core_types 不得 import 任何 huginn.*、config 不得 import 业务层"这条防腐蚀边界。

## contract / invariant（gate 门禁要锁的不变量）
- `huginn/core_types.py` 顶层/方法体内**不得出现任何 `huginn.*` import**（内核隔离）。
- `huginn/config.py` 不得 import 以下**业务/应用层**包：`tools / autoloop / agent / agents / metacog / workflows / knowledge / causal / memory / evolution / perception / bench / academic / execution / exploration`（允许基建：`crypto / checkpointer / models / feature_flags / config_integrity` + 标准库）。

## decisions
- **只加门禁测试，不改运行时**：新 `tests/test_arch_seam_integrity.py`（2 个 AST 扫描测试，复用现有 arch 测试的 AST 手法），零行为改动。
- **不搬类型**：诊断显示 core_types/config 当前成员均合法（跨层契约/基建），无 egregious 误置项可下沉 → 本次不迁移动，避免为搬而搬（防止过度工程）。
- **门禁性质**：invariant-lock 回归测试（现行为已满足，测试立即绿，锁住"不许退化"），与现有 `test_arch_*` 同为表征/不变量测试；无生产代码改动，不适用红-绿-red 循环。

## test（如何知道锁住了）
- `tests/test_arch_seam_integrity.py`：
  1. `test_core_types_kernel_has_no_business_imports`：AST 扫 `huginn/core_types.py`，断言无 `huginn.` import（任何形态，含 try/except 内）。
  2. `test_config_does_not_import_application_layers`：AST 扫 `huginn/config.py`，断言 `from huginn.<业务层>` 为空（基建 import 白名单放行）。
- 回归：`pytest tests/test_arch_seam_integrity.py` + 既有 `test_arch_cleanliness / test_arch_single_gateway` 全绿。

## deferred（暂不定）
- 为一个方向校验 config 业务 import 的"允许基建白名单"精确集合，测试里用黑名单(业务包名列表)更稳，未列入白名单。
- 未来若 core_types/config 确有成员需要下沉，单独开 PR（本 spec 不预判）。

## 范围（out of scope）
- 不改 core_types/config 任何代码。
- 不新增运行时加载/迁移。
- 不动现有 `test_arch_*` 既有用例。

## Status（2026-09-13 实施完成）
- 落地 `tests/test_arch_seam_integrity.py`：①core_types 零 `huginn.*` import ②config 不反向依赖业务层(基建白名单放行)。
- 验证：新门禁 + `test_arch_cleanliness` + `test_arch_single_gateway` 全绿（16 passed）。