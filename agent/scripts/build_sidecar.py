#!/usr/bin/env python3
"""Build Huginn as a Tauri sidecar (lightweight onedir package).

Optimized for fast builds by only including core dependencies.
The output is copied to desktop/src-tauri/sidecars/ for Tauri bundling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
SIDECAR_DIR = PROJECT_ROOT / "desktop" / "src-tauri" / "sidecars"


def build_sidecar():
    """Build a lightweight onedir package for Tauri sidecar."""
    entry_point = "huginn.cli:main"
    output_name = "huginn-sidecar"

    assets_src = PROJECT_ROOT / "huginn" / "assets"
    assets_dst = "huginn/assets"

    # Core hidden imports only — avoid heavy optional deps
    core_imports = [
        # Tools (core only, exclude heavy simulation packages)
        "huginn.tools.base",
        "huginn.tools.bash_tool",
        "huginn.tools.code_tool",
        "huginn.tools.file_edit_tool",
        "huginn.tools.file_read_tool",
        "huginn.tools.file_write_tool",
        "huginn.tools.git_tool",
        "huginn.tools.diff_tool",
        "huginn.tools.validate_tool",
        "huginn.tools.diagnose_tool",
        "huginn.tools.extract_tool",
        "huginn.tools.job_tool",
        "huginn.tools.database_tool",
        "huginn.tools.potential_tool",
        "huginn.tools.structure_tool",
        "huginn.tools.report_tool",
        "huginn.tools.orchestrate_tool",
        "huginn.tools.memory_tool",
        "huginn.tools.lean_tool",
        "huginn.tools.skill_tool",
        "huginn.tools.evidence_fusion_tool",
        "huginn.tools.tda_tool",
        "huginn.tools.unit_tool",
        "huginn.tools.numerical_tool",
        "huginn.tools.high_throughput_tool",
        "huginn.tools.symmetry_tool",
        "huginn.tools.gp_tool",
        "huginn.tools.uq_tool",
        "huginn.tools.descriptor_tool",
        "huginn.tools.autodiff_tool",
        "huginn.tools.symbolic_regression_tool",
        "huginn.tools.symbolic_math_tool",
        "huginn.tools.visualize_tool",
        "huginn.tools.active_learning_tool",
        "huginn.tools.ml_potential_tool",
        "huginn.tools.characterization_tool",
        "huginn.tools.experimental_data_tool",
        "huginn.tools.materials_database_tool",
        "huginn.tools.registry",
        "huginn.tools.bourbaki_tool",
        # 新增：工具系统改进（fail-closed + 装配过滤）
        "huginn.tools.defaults",
        "huginn.tools.assembly",
        # Security
        "huginn.security.safe_eval",
        "huginn.security.math_eval",
        # 新增：认证SWR缓存 + 401去重刷新（接入 health.py 和 llm_retry.py）
        "huginn.security.auth",
        # Core modules
        "huginn.agent",
        "huginn.config",
        "huginn.llm",
        "huginn.server",
        "huginn.server_context",
        "huginn.types",
        # CLI
        "huginn.cli.main",
        "huginn.cli.context",
        # 新增：CLI改进（懒加载 + availability过滤 + 设计系统）
        "huginn.cli.lazy_loader",
        "huginn.cli.availability",
        "huginn.cli.design_system",
        # Skills & workflows
        "huginn.skills.base",
        "huginn.skills.registry",
        "huginn.skills.presets",
        "huginn.workflows.high_throughput",
        # Utils
        "huginn.utils.units",
        "huginn.utils.numerical",
        "huginn.utils.conversation_tree",
        "huginn.utils.prompt_cache",
        "huginn.utils.tokens",
        "huginn.utils.context",
        "huginn.utils.cache",
        # Subsystems
        "huginn.coder.loop",
        "huginn.coder.checkpoint",
        "huginn.exploration.core",
        "huginn.exploration.orchestrator",
        "huginn.exploration.strategies",
        "huginn.workflows.engine",
        "huginn.workflows.stages",
        "huginn.workflows.checkpoint",
        "huginn.workflows.templates",
        "huginn.bench.runner",
        "huginn.bench.task",
        "huginn.evolution.engine",
        "huginn.evolution.logger",
        "huginn.kg.builder",
        "huginn.kg.graph",
        "huginn.kg.entities",
        "huginn.kg.extractor",
        "huginn.memory.manager",
        "huginn.memory.session",
        "huginn.memory.longterm",
        # 新增：记忆系统改进（类型化分类 + 截断保护 + 索引结构）
        "huginn.memory.types",
        "huginn.memory.truncation",
        "huginn.memory.index",
        "huginn.rag.encrypted_rag",
        "huginn.rag.vector_store",
        "huginn.security.sandbox",
        "huginn.security.audit",
        "huginn.security.container_executor",
        "huginn.security.crypto",
        "huginn.crypto",
        "huginn.privacy.scanner",
        # 新增：LLM重试模块（429/529分级重试 + 模型降级）
        "huginn.llm_retry",
        # 新增：上下文管理（memoized + git status + 窗口计算）
        "huginn.context_manager",
        # 新增：MCP客户端增强（call_tool_with_retry + connect_batch）
        "huginn.mcp_client",
        # 新增：工具调用钩子系统（PreToolUse/PostToolUse）
        "huginn.hooks",
        # 新增：项目记忆（AGENTS.md 契约加载）
        "huginn.project_memory",
        # 新增：多模型团队编排（按能力路由不同 LLM 到不同角色）
        "huginn.agents.team",
        # 新增：技能加载器（frontmatter解析 + 条件激活）
        "huginn.plugins.skill_loader",
        # 新增：科学技能桥接器（37个数据库技能的发现和注册）
        "huginn.plugins.science_skills_bridge",
        # 新增：懒加载模块保险（函数内 import, PyInstaller 可能漏判）
        "huginn.advisor.knowledge",
        "huginn.advisor.model_advisor",
        "huginn.lean.interface",
        "huginn.lean.auto_pipeline",
        "huginn.lean.sympy_to_lean",
        "huginn.lean.pipeline",
        "huginn.perception",
        "huginn.perception.cognitive_integration",
        "huginn.perception.semantic_alignment",
        "huginn.perception.simulator_log_tailer",
        "huginn.perception.webbridge_monitor",
        "huginn.perception.terminal_capture",
        "huginn.perception.filesystem_watcher",
        "huginn.unified.core",
        "huginn.unified.discretize",
        "huginn.unified.derive",
        "huginn.unified.solve",
        "huginn.unified.models",
        "huginn.unified.visualize",
        "huginn.unified.bridge",
        "huginn.tools.parameters",
        "huginn.tools.mcp_adapter",
        "huginn.tools.browser_tool",
        "huginn.security.user_store",
        "huginn.security.rbac",
        "huginn.memory.decay",
        "huginn.export_manager",
        "huginn.visualize",
        "huginn.telemetry",
        "huginn.codebase",
        "huginn.constraints.operators",
        "huginn.constraints.reference",
        "huginn.diagnostics.convergence",
        "huginn.evaluation.core",
        "huginn.execution.remote_executor",
        "huginn.execution.remote_job_store",
        "huginn.hpc.client",
        "huginn.hpc.resource_selector",
        "huginn.models.registry",
        "huginn.models.router",
        "huginn.permissions",
        "huginn.personas",
        "huginn.persona_emotion",
        "huginn.persona_loader",
        "huginn.persona_matcher",
        "huginn.pet",
        "huginn.plugins.autoresearch",
        "huginn.project_context",
        "huginn.prompts",
        "huginn.scheduler",
        "huginn.skills.base",
        "huginn.skills.registry",
        "huginn.utils.cache",
        "huginn.utils.context",
        "huginn.utils.tokens",
        "huginn.validation.physics",
        # 接入原死代码模块：弹性张量校验 + 层次化路由检索 + 系统状态快照
        "huginn.mechanics",
        "huginn.rag.router_retriever",
        "huginn.system",
        # Autoloop (new)
        "huginn.autoloop",
        "huginn.autoloop.engine",
        # LangChain
        "langchain",
        "langchain_core",
        "langchain_core.messages",
        "langchain_core.tools",
        "langgraph",
        # CLI
        "click",
        "rich",
        "rich.console",
        "rich.panel",
        "rich.progress",
        "rich.table",
        # Data
        "pydantic",
        "numpy",
        "networkx",
        "yaml",
        "dotenv",
        "cryptography",
        "aiohttp",
        "websockets",
        # Scientific (lightweight only — heavy deps excluded below)
        "sympy",
    ]

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--onedir",
        "--name", output_name,
        "--noconfirm",
        *(f"--add-data={assets_src}{os.pathsep}{assets_dst}".split() if assets_src.exists() else []),
        *[f"--hidden-import={imp}" for imp in core_imports],
        # Exclude heavy test/dev tools
        "--exclude-module=pytest",
        "--exclude-module=pytest_asyncio",
        "--exclude-module=pytest_benchmark",
        "--exclude-module=pytest_cov",
        "--exclude-module=black",
        "--exclude-module=ruff",
        "--exclude-module=mypy",
        "--exclude-module=pre_commit",
        "--exclude-module=pip_audit",
        "--exclude-module=memory_profiler",
        # Exclude heavy simulation packages (not core)
        "--exclude-module=ase",
        "--exclude-module=pymatgen",
        "--exclude-module=phonopy",
        "--exclude-module=spglib",
        # Exclude heavy ML/OCR/vision packages — too large for installer
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=torchaudio",
        "--exclude-module=transformers",
        "--exclude-module=easyocr",
        "--exclude-module=skimage",
        "--exclude-module=sklearn",
        "--exclude-module=scipy",
        "--exclude-module=matplotlib",
        "--exclude-module=numba",
        "--exclude-module=llvmlite",
        "--exclude-module=pyarrow",
        "--exclude-module=pandas",
        "--exclude-module=shapely",
        "--exclude-module=altair",
        "--exclude-module=narwhals",
        "--exclude-module=lxml",
        "--exclude-module=PIL",
        "--exclude-module=cv2",
        "--exclude-module=chromadb",
        "--exclude-module=sentence_transformers",
        "--exclude-module=onnxruntime",
        "--exclude-module=tensorflow",
        str(PROJECT_ROOT / "scripts" / "entry.py"),
    ]

    print("Building Huginn sidecar (lightweight)...")
    print(f"Command: {' '.join(cmd[:20])} ...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    # Copy to Tauri sidecars directory
    exe_src = DIST_DIR / output_name / f"{output_name}.exe"
    if not exe_src.exists():
        print(f"Expected output not found: {exe_src}")
        sys.exit(1)

    # Tauri expects the sidecar with the target triple suffix
    # 同时生成 gnu 和 msvc 后缀，兼容两种工具链
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    src_size = exe_src.stat().st_size
    copy_targets: list[Path] = [
        SIDECAR_DIR / "huginn-sidecar-x86_64-pc-windows-gnu.exe",
        SIDECAR_DIR / "huginn-sidecar-x86_64-pc-windows-msvc.exe",
        SIDECAR_DIR / "huginn-sidecar.exe",
    ]

    failures: list[str] = []
    for exe_dst in copy_targets:
        # 目标可能被 Tauri / 上次运行的 sidecar 锁住, 先尝试删
        used_atomic = False
        if exe_dst.exists():
            try:
                exe_dst.unlink()
            except PermissionError as e:
                # 锁住了: 用临时名 + os.replace 原子替换
                print(f"  [warn] {exe_dst.name} 被占用, 尝试原子替换: {e}")
                tmp_dst = exe_dst.with_suffix(".exe.tmp")
                try:
                    shutil.copy2(exe_src, tmp_dst)
                    os.replace(tmp_dst, exe_dst)
                    used_atomic = True
                except OSError as e2:
                    failures.append(f"{exe_dst.name}: {e2}")
                    print(f"  [FAIL] {exe_dst.name}: {e2}")
                    continue
            except OSError as e:
                failures.append(f"{exe_dst.name}: {e}")
                print(f"  [FAIL] {exe_dst.name}: {e}")
                continue

        # 目标不存在或刚被 unlink 删除, 走普通拷贝
        # (PermissionError 分支已经用原子替换完成, 跳过这里)
        if not used_atomic:
            try:
                shutil.copy2(exe_src, exe_dst)
            except OSError as e:
                failures.append(f"{exe_dst.name}: {e}")
                print(f"  [FAIL] {exe_dst.name}: {e}")
                continue

        # 校验: 大小必须和源一致, 否则拷贝不完整
        if not exe_dst.exists():
            failures.append(f"{exe_dst.name}: 目标未生成")
            print(f"  [FAIL] {exe_dst.name}: 目标未生成")
            continue
        dst_size = exe_dst.stat().st_size
        if dst_size != src_size:
            failures.append(
                f"{exe_dst.name}: 大小不匹配 (src={src_size} dst={dst_size})"
            )
            print(f"  [FAIL] {exe_dst.name}: 大小不匹配 src={src_size} dst={dst_size}")
            continue
        print(
            f"  [OK] {exe_dst.name} ({dst_size / (1024*1024):.1f} MB, "
            f"mtime={time.ctime(exe_dst.stat().st_mtime)})"
        )

    # _internal 是 onedir 必需的依赖目录, 同样需要校验
    internal_src = DIST_DIR / output_name / "_internal"
    internal_dst = SIDECAR_DIR / "_internal"
    if internal_src.exists():
        if internal_dst.exists():
            try:
                shutil.rmtree(internal_dst)
            except OSError as e:
                print(f"  [warn] 清理旧 _internal 失败: {e}, 继续覆盖")
        try:
            shutil.copytree(internal_src, internal_dst)
            total_size = sum(
                f.stat().st_size for f in internal_dst.rglob("*") if f.is_file()
            )
            print(f"  [OK] _internal ({total_size / (1024*1024):.1f} MB)")
        except OSError as e:
            failures.append(f"_internal: {e}")
            print(f"  [FAIL] _internal: {e}")
    else:
        print("  [warn] 源 _internal 不存在, 跳过")

    if failures:
        print(f"\n[ERROR] {len(failures)} 个文件拷贝失败:")
        for f in failures:
            print(f"  - {f}")
        print("提示: 请关闭正在运行的桌面应用 / sidecar 进程后重试.")
        sys.exit(1)

    print(f"\nSidecar ready for Tauri bundling!")


if __name__ == "__main__":
    build_sidecar()
