"""架构门禁：单网关分层检查 (ADR-0001)。

强制规则 (CI fast-fail)：
  R1 外部消费者（scripts/ examples/ servers/ desktop/src-tauri/ sidecar/）
     不得直接 `import huginn.*` 业务模块，除非在 ALLOWED_EXTERNAL_IMPORTS 冻结清单。
  R2 冻结清单只许缩、不许涨：任何新出现的"外部直连业务模块"都会让 CI 变红；
     若某冻结条目已不再直连（已迁移），则必须从清单移除，否则报错。

设计：
  - 用 AST 解析（比正则可靠），能捕获 try/except 或函数内嵌套 import。
  - 仓库根 = 本文件向上三级（agent/tests/xxx.py -> ../../..）。
  - 只读扫描，不改任何源码。

迁移须知：
  清单里的条目是已知的历史旁路直连，迁移目标是逐个改为调用 huginn.server 的
  HTTP 端点（如 /v1/skills、/v1/knowledge 等），迁移后从 ALLOWED_EXTERNAL_IMPORTS
  移除。任何新增脚本若要读业务模块，必须改走 API，不能往清单里加。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# 仓库根 = 本文件 (agent/tests/test_arch_single_gateway.py) 向上三级
REPO_ROOT = Path(__file__).resolve().parents[2]

# 外部消费者目录（相对仓库根）。这些目录里的 .py 不许直接 import 业务模块。
EXTERNAL_CONSUMER_DIRS = [
    "scripts",
    "examples",
    "servers",
    "sidecar",
    # desktop/src-tauri 是 Rust，无 Python import；desktop/src（React）也不
    # import huginn.*。这里保留以便将来新增 Python 消费者时默认纳入。
]

# 业务包根（不允许被外部消费者直接 import 的包）。
BUSINESS_PACKAGE = "huginn"

# CLI 子命令 → Python 子进程委托（ADR-0001 "CLI 不再是第二前门"）。
# Rust CLI 仍通过 `python -m huginn.cli` spawn 子进程作为第二前门；本清单
# 冻结 *当前仍在委托* 的子命令，只许缩、不许涨。任何新增的委托子命令都会让
# CI 变红 —— 新子命令必须做成 HTTP/WS 客户端连接 huginn.server，而不是再
# spawn Python 子进程。迁移一个子命令为 HTTP/WS 客户端后，从本清单移除。
CLI_DELEGATED_SUBCOMMANDS: set[str] = {
    "chat",
    "explore",
    "serve",
    "coder",
    "bench",
    "evolve",
    "execute",
    "workflow",
    "diagnose",
    "hpc",
    "encrypt-config",
}

# 冻结清单：已知的历史旁路直连（相对仓库根）。只能缩，不能涨。
# 迁移方向见 ADR-0001 与各条 reason。
ALLOWED_EXTERNAL_IMPORTS: dict[str, dict[str, str]] = {
    # 技能库盘点/诊断脚本 —— 迁移到 /v1/skills 后从清单移除
    "scripts/skill_audit.py": {
        "reason": "技能库图论/拓扑盘点，需读 SkillRegistry；迁移到 /v1/skills 后移除",
        "migrate_to": "/v1/skills",
    },
    "scripts/skill_hole_diag.py": {
        "reason": "技能库 β₁ 洞诊断，需读 SkillRegistry；迁移到 /v1/skills 后移除",
        "migrate_to": "/v1/skills",
    },
    # RAG 摄取脚本 —— 迁移到 /v1/knowledge 后移除
    "scripts/ingest_sobko_to_rag.py": {
        "reason": "RAG 知识摄取，需读 VectorStore；迁移到 /v1/knowledge 后移除",
        "migrate_to": "/v1/knowledge",
    },
    # 一次性/按需验证脚本（try/except 内 import）—— 迁移到对应用户态 API 后移除
    "scripts/verify_symbolic_layer.py": {
        "reason": "符号层验证，需读 SymbolicMathTool/符号回归；迁移后移除",
        "migrate_to": "execution/verify",
    },
    "scripts/verify_all_optimizations.py": {
        "reason": "全优化项回归验证，需读 workflows 模板；迁移后移除",
        "migrate_to": "execution/verify",
    },
    "scripts/verify_execution_layer.py": {
        "reason": "执行层验证，需读 input_generator/result_parser/autofix；迁移后移除",
        "migrate_to": "execution/verify",
    },
    "scripts/verify_fea_cfd_integration.py": {
        "reason": "FEA/CFD 集成验证，需读 workflows 模板；迁移后移除",
        "migrate_to": "execution/verify",
    },
}


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _find_business_imports(path: Path) -> list[str]:
    """返回文件里所有 `import huginn` / `from huginn ...` 的『第一段标识』列表。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == BUSINESS_PACKAGE or alias.name.startswith(
                    BUSINESS_PACKAGE + "."
                ):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == BUSINESS_PACKAGE
            or node.module.startswith(BUSINESS_PACKAGE + ".")
        ):
            hits.append(node.module)
    return sorted(set(hits))


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_external_consumers_must_not_import_business_package():
    """R1+R2：外部消费者不得直连业务模块；冻结清单只许缩不涨。"""
    found: dict[str, list[str]] = {}  # rel_path -> imports
    for rel_dir in EXTERNAL_CONSUMER_DIRS:
        for path in _iter_python_files(REPO_ROOT / rel_dir):
            imports = _find_business_imports(path)
            if imports:
                found[_rel(path)] = imports

    # R2a：清单里已不在"直连"的条目 → 已完成迁移，必须从清单移除
    stale = [p for p in ALLOWED_EXTERNAL_IMPORTS if p not in found]
    assert not stale, (
        "以下冻结条目已不再直连业务模块——请从 ALLOWED_EXTERNAL_IMPORTS 移除"
        "（证明迁移完成）：\n  " + "\n  ".join(stale)
    )

    # R1+R2b：任何清单外的直连 → 新旁路，阻断（不能往清单里加，只能改走 API）
    new_violations = {
        p: imports for p, imports in found.items() if p not in ALLOWED_EXTERNAL_IMPORTS
    }
    assert not new_violations, (
        "检测到新增『外部直接 import huginn.* 业务模块』，违反 ADR-0001 单网关原则。"
        "请改为调用 huginn.server 的 HTTP API，不要直接 import 业务模块，"
        "也不要往 ALLOWED_EXTERNAL_IMPORTS 里加条目。\n"
        + "\n".join(f"  - {p}: {', '.join(imports)}" for p, imports in new_violations.items())
    )


def test_allowlist_reasons_documented():
    """清单里每条都要有 reason 与迁移目标，防止无脑加清单。"""
    for path, meta in ALLOWED_EXTERNAL_IMPORTS.items():
        assert meta.get("reason"), f"{path}: ALLOWED_EXTERNAL_IMPORTS 缺 reason"
        assert meta.get("migrate_to"), f"{path}: ALLOWED_EXTERNAL_IMPORTS 缺 migrate_to"


def test_arch_gate_self_test():
    """门禁自检：确认识别逻辑能抓到已知直连（防止门禁自己失效）。"""
    # 用 skill_audit.py 验证 AST 识别能命中
    sample = REPO_ROOT / "scripts" / "skill_audit.py"
    if not sample.exists():
        return
    imports = _find_business_imports(sample)
    assert any("huginn" in i for i in imports), (
        "arch-gate 自检失败：识别不到 skill_audit.py 的 huginn 导入，门禁可能失效"
    )


def _cli_delegated_commands(main_rs: Path) -> set[str]:
    """扫描 cli/src/main.rs，提取所有 delegate_to_python(...) 的子命令名。

    子命令名是 delegate_to_python 的第一个字符串参数（如 "chat"、"explore"）。
    """
    text = main_rs.read_text(encoding="utf-8", errors="ignore")
    delegated: set[str] = set()
    i = 0
    while True:
        idx = text.find("delegate_to_python(", i)
        if idx == -1:
            break
        rest = text[idx + len("delegate_to_python("):]
        inner = rest.split(")", 1)[0]
        # 取第一个双引号字符串参数
        start = inner.find('"')
        if start != -1:
            end = inner.find('"', start + 1)
            if end != -1:
                delegated.add(inner[start + 1:end])
        i = idx + len("delegate_to_python(")
    return delegated


def test_cli_must_not_add_new_subprocess_delegation():
    """ADR-0001：CLI 委托清单只许缩、不许涨。

    冻结当前仍通过 `python -m huginn.cli` spawn 子进程的子命令。任何新增的
    委托子命令都会让 CI 变红，强制新子命令改走 HTTP/WS 客户端。
    """
    main_rs = REPO_ROOT / "cli" / "src" / "main.rs"
    if not main_rs.exists():
        return  # cli 未构建时不判断

    actual = _cli_delegated_commands(main_rs)

    # 已迁移为 HTTP/WS 客户端、从清单移除的子命令 → 不能再出现在委托里
    migrated = CLI_DELEGATED_SUBCOMMANDS - actual
    assert not migrated, (
        "以下子命令已从 CLI_DELEGATED_SUBCOMMANDS 移除（视为已迁移），但 main.rs "
        "仍在 delegate_to_python —— 请确认迁移完成再移除清单条目：\n  "
        + "\n  ".join(sorted(migrated))
    )

    # 清单外新增的委托 → 新旁路，阻断
    new_bypass = actual - CLI_DELEGATED_SUBCOMMANDS
    assert not new_bypass, (
        "检测到新增『CLI spawn python -m huginn.cli 子进程』，违反 ADR-0001 "
        "单网关原则。请把子命令改为 HTTP/WS 客户端连接 huginn.server，不要新增 "
        "子进程委托，也不要往 CLI_DELEGATED_SUBCOMMANDS 里加条目：\n  "
        + "\n  ".join(sorted(new_bypass))
    )


if __name__ == "__main__":
    # 允许本地直接跑：python tests/test_arch_single_gateway.py
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
