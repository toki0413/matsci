# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- H6 双吸引子 band 路由: `ModelRouter` 增加 persona 稳定带 (spec/react) 维度, `classify_band` 把任务/阶段外部量化到稳定带 (避开 mixed 相变陷阱), `select_band` 选带稳定模型 (spec↔react 不互通, 通用模型承接), `ModelConfig.bands` 支持配置标注稳定带

## [1.3.0] - 2026-08-16

### Added
- 极简模式 (ModelTier FULL/BALANCED/MINIMAL) × 思考强度 (ThinkingIntensity low/medium/high/max) 正交两轴, 分别控制认知编排开销与模型推理深度
- 统一成本账本 (CostLedger) + 价值感知阶段伸缩预算 (ValueBudget) + 预算边缘 Checkpoint/暂停 (BudgetPause)
- 分支评分 (BranchScore/UCB) 与分支休眠/复活 (BranchState/DecisionPoint) 的科研创新矛盾消解
- 细粒度权限契约 (PermissionMode/RiskLevel/PermissionConfig/PermissionChecker) 与前端设置面板接入
- 权限契约文档与成本参与契约文档 (docs/permission-contract.md, docs/cost-participation-contract.md)

### Fixed
- CI 测试隔离模型路由 (engine.model_router=None), 避免全局 config 缓存污染导致测试路由到真实模型
- 安全 PoC 测试路径修正 (tests/security/pentest_*.py)

## [1.2.0] - 2026-08-15

### Added
- 5xx retry with 1s backoff in desktop API client for remote backend deployment
- Version bump script (`scripts/bump_version.ps1`) to sync version across all config files
- Tag-driven stable release workflow (`.github/workflows/release.yml`)
- 视觉像素 MCP server (`vision-pixel`): 移植 dsh-vision-router 的像素闭环能力 (裁剪/主色/像素对比/抠图/SVG 矢量化/看图摘要), 纯 PIL/numpy 实现无 Node 依赖
- 接入 5 个插件: dsh-vision-router 视觉工具、dsh-auto-blame、ModLens、OpenPencil、Argo
- MCP 连接健壮性: 超时提到 60s 并在后台初始化, 逐个连接避免 anyio cancel-scope 竞态

### Changed
- Desktop CI builds remain as prerelease (`desktop-ci-N` tag)
- Stable releases now triggered by `v*` tags (e.g. `v0.2.0`)

## [0.1.0] - 2026-07-08

### Added
- LangGraph ReAct agent with 7-phase autoloop (perceive → hypothesize → plan → execute → validate → learn → report)
- 40+ simulation tools (VASP, QE, CP2K, Gaussian, ORCA, LAMMPS, Gromacs, Abaqus, Comsol, Elmer, FEniCS, OpenFOAM, RDKit, OpenMM, AutoDock Vina, etc.)
- Tool execution hooks (PRE/POST_TOOL_USE) with 15 built-in science hooks
- Event-driven simulation pipeline with 14 workflow rules
- Scientific workflow DAG visualization (Mermaid)
- Compression-aware intelligent prefetching
- Provenance registry with JSONL snapshots
- Agent trajectory logging
- Tauri desktop app with React 18 + TypeScript + Tailwind
- WebSocket streaming chat with plan confirmation
- 9 tool panels (evolve, benchmark, explore, coder, execute, workflow, diagnose, hpc, team)
- Credential management panel
- 37 science-skills for biomedical/chemical/materials databases
- CI test suites: API contract (1227 cases), security (63), chaos (26), a11y (21), performance (17)
- GitHub Actions CI: Python 3.10-3.13, Integration tests, Desktop build, Stress test
- Desktop CI prerelease builds with MSI/EXE/wheel artifacts
- Executable resolver with user-guided path selection (local/HPC/mock)
- Transolver++ PDE solver integration
