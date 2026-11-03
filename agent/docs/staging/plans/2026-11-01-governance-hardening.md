# 计划: 治理面加固

spec: `specs/2026-11-01-governance-hardening.md`
execution: inline

- [x] T1: Surface 段名台账
  goal: 主路径每个具名段都在台账里，缺段/多段都红。
  files: huginn/agent/context.py, tests/test_model_surface_contract.py
  acceptance: `python -m pytest tests/test_model_surface_contract.py -q -o addopts=""`
  spec: `#d1-surface-段名台账`

- [x] T2: Catalog handle 契约
  goal: kind 与所需 handle 双向闭合，未知 handle 显式报错。
  files: huginn/catalog/manager.py, tests/test_arch_catalog_handles.py
  acceptance: `python -m pytest tests/test_arch_catalog_handles.py -q -o addopts=""`
  spec: `#d2-catalog-handle-契约`

- [x] T3: `__all__` 双向一致性门
  goal: 每个包 `__all__` 与模块公共名一一对应。
  files: tests/test_arch_package_exports.py
  acceptance: `python -m pytest tests/test_arch_package_exports.py -q -o addopts=""`
  spec: `#d3-__all__-双向一致性`

- [x] T4: 静默异常 per-file 棘轮
  goal: 单文件静默异常数只能降，不能靠跨模块腾挪。
  files: tests/test_silent_exceptions.py, tests/silent_exceptions_baseline.json
  acceptance: `python -m pytest tests/test_silent_exceptions.py -q -o addopts=""`
  spec: `#d4-静默异常-per-file-棘轮`

- [x] T5: 文档分类全量
  goal: docs/*.md 每篇都有 generated 或 manual 标记。
  files: scripts/doc_assertions.py, tests/test_doc_assertions.py, docs/*.md
  acceptance: `python scripts/doc_assertions.py --regen`
  spec: `#d5-文档分类全量`

- [x] T6: 发布 bundle 白名单
  goal: bundle 内只允许清单登记的路径。
  files: scripts/mcpb_vendor.py, tests/test_publish_mirror_drift.py
  acceptance: `python -m pytest tests/test_publish_mirror_drift.py -q -o addopts=""`
  spec: `#d6-发布-bundle-内容白名单`

- [x] T7: 收尾同步
  goal: 台账文档与镜像同步，全量门绿。
  files: huginn/cli/governance_audit.py, docs/governance-state.md, publish/mcpb/**
  acceptance: `python -m pytest tests/test_arch_gate_registry.py tests/test_contract_audit.py tests/test_doc_assertions.py tests/test_publish_mirror_drift.py -q -o addopts="" && python -m ruff check .`
  spec: `#验收`

- [x] T8: 治理门自标记与归属
  goal: 门改名不再逃逸；标记与登记双向闭合，每个门有归属。
  files: huginn/cli/governance_audit.py, tests/test_arch_gate_ownership.py, tests/**
  acceptance: `python -m pytest tests/test_arch_gate_ownership.py -q -o addopts=""`
  spec: `#d7-治理门自标记与归属`

- [x] T9: 全仓 lint 棘轮
  goal: 逐文件存量违规只减不增，基线外文件零违规。
  files: tests/ruff_baseline.json, tests/test_arch_ruff_baseline.py
  acceptance: `python -m pytest tests/test_arch_ruff_baseline.py -q -o addopts=""`
  spec: `#d8-全仓-lint-棘轮`

- [x] T10: 门强度 / 证据 / TTL (DSH 生命周期)
  goal: 豁免从永久白名单升级为带证据与复核期限的租约，新门可先 OBSERVE 影子期，过期回落孤儿门。
  files: huginn/cli/governance_audit.py, tests/test_arch_gate_ownership.py, docs/governance-state.md
  acceptance: `python -m pytest tests/test_arch_gate_ownership.py tests/test_arch_gate_registry.py -q -o addopts=""`
  spec: `#d9-门强度生命周期`

- [x] T11: 去厂商/人格绑定
  goal: 交付面清除实验人格名（含 examples 文件名），报告/工具描述/错误文案/技能描述中性化；历史运行证据不动。
  files: huginn/**, examples/**, docs/**, tests/test_arch_no_vendor_persona.py
  acceptance: `python -m pytest tests/test_arch_no_vendor_persona.py -q -o addopts=""`
  spec: `#d10-去厂商人格绑定`

- [x] T12: 产物目录迁移层
  goal: 新目录 ai4s_* 写入，旧目录 shusheng_* 兼容读取（文件复制迁移、目录只回落不复制）。
  files: examples/_output_paths.py, examples/ai4s_*.py, tests/test_output_dir_migration.py
  acceptance: `python -m pytest tests/test_output_dir_migration.py -q -o addopts=""`
  spec: `#d10-去厂商人格绑定`
