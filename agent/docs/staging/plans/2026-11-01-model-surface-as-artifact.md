# 计划：模型面即产物（表示层 + 主路径迁移 + 披露门）

spec: `docs/staging/specs/2026-11-01-model-surface-as-artifact.md`
execution: inline

## B 表示层

- [x] T1: Surface 表示层
  goal: 新增 `SurfaceSection` / `Surface` / `compose`，段携带 name 与 source，重复名抛错
  files: `huginn/agent/model_surface.py`、`tests/test_model_surface_contract.py`
  acceptance: 单元测试覆盖顺序/空段跳过/来源/重复名 ValueError；模块 import 不拉任何 huginn 重栈
  spec: `#b-表示层`

## C 主路径迁移

- [x] T2: 主路径改为 Surface 组装
  goal: `_effective_system_prompt` 内部改为 `_effective_surface()` + `render()`，各 `_esp_*` 返回 fragment，删除字符串拼接
  files: `huginn/agent/context.py`
  acceptance: stub context 下 default/research/extreme 三个 prompt 的 sha256 与迁移前基线逐字节相同（10630/e6043184、10899/7a2ae636、11008/50fa17409ae2de3）
  spec: `#c-主路径迁移`

## D 披露面与门

- [x] T3: 机制普查段
  goal: `KIND_DISCLOSURE` 覆盖 7 类 kind，`census_text()` 生成 `## MECHANISMS` 段，Catalog 懒加载、失败返回空串
  files: `huginn/agent/mechanism_census.py`
  acceptance: 导入不产生重依赖；`hidden` 无理由 → 抛错；`census_text()` 对注册的 skill/plugin 名称逐条出现
  spec: `#d-披露面与门`

- [x] T4: 把普查段接进主路径并加门
  goal: 普查段注册为 prompt 段（main/s7/autoloop 三路径生效）；契约门断言结构不变量与披露双向
  files: `huginn/agent/context.py`、`huginn/agent/prompt_builder.py`、`huginn/plugins/prompt_segments.py`、`tests/test_model_surface_contract.py`
  acceptance: stdlib 门全绿；runtime 部分在 `importorskip` 下断言真实 Surface 含 `## MECHANISMS`；三路径实测均含该段
  spec: `#d-披露面与门`

## E 收编台账

- [x] T5: 组装根台账（只减不增）
  goal: 扫描 `agent/autoloop/agents/coder/interaction` 中直接构造 `SystemMessage(content=...)` 的文件，双向核对 15 个已登记根
  files: `tests/test_model_surface_contract.py`、`huginn/cli/governance_audit.py`、`docs/governance-state.md`
  acceptance: 新增一个 SystemMessage 文件 → 门红；`PROMPT_SPECS`=14 与注册表一致；`governance-state.md` 重生成后逐字一致
  spec: `#e-收编台账`
