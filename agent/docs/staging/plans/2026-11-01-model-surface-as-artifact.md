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
  goal: 扫描 `agent/autoloop/agents/coder/interaction`，双向核对未用 `compose()` 的 `SystemMessage(content=...)` 根
  files: `tests/test_model_surface_contract.py`、`huginn/cli/governance_audit.py`、`docs/governance-state.md`
  acceptance: 新增一个字符串根 → 门红；`PROMPT_SPECS`=14 与注册表一致；`governance-state.md` 重生成后逐字一致
  spec: `#e-收编台账`

## F 第二批迁移与顽疾核查

- [x] T6: CodeAct 并入表示层，披露覆盖第四条路径
  goal: `_build_surface` 具名分段、`_build_system_prompt` 只 `render()`；`mechanisms` 段进 CodeAct；台账 15 → 14
  files: `huginn/agent/code_act_loop.py`、`tests/test_model_surface_contract.py`、`docs/env-contract.md`、`docs/feature-flags-contract.md`
  acceptance: 剔除 mechanisms 段后 sha256 不变（后缀关 802/a4ca1cf93cbea693、后缀开 1852/b0d5d717b2ca6bf5）；`test_codeact_path_discloses_mechanisms` 与台账双向核对全绿；两份行号类生成文档重生成后逐字一致
  spec: `#f-第二批迁移与顽疾核查`

- [x] T7: 删除 ponytail 的 on_llm_request 死钩子
  goal: 移除从未生效的 `inject_rules`、`inject_system_prompt` 配置承诺与对应测试，不改任何 prompt 字节
  files: `huginn/plugins/ponytail/main.py`、`huginn/plugins/ponytail/_conf_schema.json`、`tests/test_ponytail_plugin.py`、`docs/plugins-contract.md`、`docs/pluginized-segments-design.md`
  acceptance: `collect_handlers()` 只含 `ON_MESSAGE_RECEIVED`；`docs/plugins-contract.md` 重生成后事件钩子表为空且含 `mechanisms` 行；`test_ponytail_plugin` / `test_model_surface_contract` / `test_doc_assertions` 全绿
  spec: `#f-第二批迁移与顽疾核查`
