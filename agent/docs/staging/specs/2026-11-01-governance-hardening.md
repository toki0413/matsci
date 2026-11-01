# 治理面加固: 把「未登记的东西」纳入可见决策

日期: 2026-11-01
milestone: 无 (单批治理加固, 不跨里程碑)

## 决策

### D1. Surface 段名台账

- contract: `huginn/agent/context.py` 新增 `SURFACE_SPECS: dict[str, str]`，键是
  `compose_main_surface` 产出的段名，值是 `source` 字符串。
- invariant: 渲染出的每个 `SurfaceSection.name` 必须出现在台账内；台账内标 `required`
  的段必须在标准 ctx 下渲染出非空文本。
- 台账分两组：`required`（主路径必有）与 `optional`（依环境/模式出现的段，如
  `environment` / `project_context` / `csm`）。
- deferred: 不约束 `SurfaceSection.text` 内容；不给 `source` 加枚举校验。
- test: `tests/test_model_surface_contract.py` 用合成 ctx 渲染，双向核对。

### D2. Catalog handle 契约

- contract: `huginn/catalog/manager.py` 新增
  `KIND_HANDLE_SPECS: dict[str, str | None]`，kind → 该 kind 采集所依赖的 handle 键
  （无需 handle 的 kind 为 `None`）。
- invariant: `KIND_HANDLE_SPECS` 的键集 == `models.KINDS`；非 `None` 值构成的集合 ==
  `discover_handles()` 返回的键集；`discover_all` 签名接受全部声明键。
- contract: `discover_all` 拒绝未知 handle 关键字（多余参数现在被静默忽略，等于
  「传错了也不算错」）。
- test: `tests/test_arch_catalog_handles.py`（登记为 full-suite-only）。

### D3. `__all__` 双向一致性

- contract: 新增 weightless AST 门，扫 `huginn/**/__init__.py`。
- invariant: `__all__` 中每个名字都能在该模块命名空间解析（模块顶层定义 / 导入 /
  子模块名）；模块顶层定义的公共名（非 `_` 开头）必须出现在 `__all__`。
- deferred: 不检查类型标注、不检查 `__init__.py` 之外的重导出模块。
- test: `tests/test_arch_package_exports.py`。

### D4. 静默异常 per-file 棘轮

- contract: `tests/silent_exceptions_baseline.json` 记录每个文件的当前计数。
- invariant: 每个文件的计数 ≤ 基线；基线里消失的文件自动从比较集中剔除（只减不增），
  新增文件计数必须为 0。
- test: `tests/test_silent_exceptions.py` 扩展。

### D5. 文档分类全量

- contract: `scripts/doc_assertions.py` 的 `TARGET_DOCS` 改为目录扫描：
  `docs/*.md` 全部纳入，首行必须 `<!-- generated: <cmd> -->` 或
  `<!-- manual: true -->`。
- invariant: 未分类文档 = 红；generated 文档继续做字节比对。
- deferred: `docs/staging/**` 与 `docs/research-notes/**` 不纳入（过程性文档，
  非交付契约）。
- test: `tests/test_doc_assertions.py` 扩展。

### D6. 发布 bundle 内容白名单

- contract: `scripts/mcpb_vendor.py` 新增 `BUNDLE_MANIFEST`：允许进入 bundle 的
  顶层目录/文件模式。
- invariant: bundle 内每个相对路径必须匹配白名单；新增文件未登记 → 红。
- test: `tests/test_publish_mirror_drift.py` 扩展。

## Non-goals

- 不重构既有 17 个 `test_arch_*` 门。
- 不引入新依赖。
- 不改任何生产行为（D2 的未知 handle 拒绝除外，属错误路径收紧）。

## 验收

六个门全绿 + 既有门不回归 + ruff 全绿 + `docs/governance-state.md` 重生成一致。

## Working notes

- 用户以「都做」批准本批范围；spec 即决策清单，不另起方案比较。
- 新门文件名避开 `test_arch_*` 前缀的，须在 `governance_audit.FULL_SUITE_ONLY` 登记。
