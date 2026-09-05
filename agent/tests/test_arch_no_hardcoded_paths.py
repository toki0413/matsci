"""架构门禁：禁止机器特定硬编码路径 (ADR-0001 配套)。

强制规则 (CI fast-fail)：
  R1 已 git 跟踪的代码文件不得包含 Windows 用户绝对路径 (C:\\Users\\, C:/Users/)。
  R2 已 git 跟踪的代码文件不得包含黑名单中的机器专属 token (如 wanzh)。
     黑名单只许缩、不许涨。

设计要点（规避误报）：
  - 只扫描 *git 跟踪* 的文件 (git ls-files)：本地残余 (如 pyext/.cargo/config.toml,
    已被 .gitignore) 与历史提交已删除的文件都不扫。
  - 只扫代码后缀 (.py/.rs/.toml/.ts/.tsx/.yml/.yaml/.json/.sh/.js)，不扫 .md 文档
    (文档里的历史路径引用是记录，不是运行时硬编码)。
  - 通用的示例路径 (/home/user, /tmp/a.cif) 不判硬编码 —— 只盯 Windows 用户
    绝对路径 + 黑名单 token 这两种明确的机器专属特征。
  - 合法出现（如 privacy_guard.py 里的隐私检测正则）放入 ALLOWED 冻结清单，
    只许缩、不涨。

迁移须知：
  任何新增的 `C:\\Users\\...` 路径或黑名单 token 都会让 CI 变红。机器相关路径
  应改为 env 变量展开 (如 workspace = "env:HUGINN_WORKSPACE")，不要提交到源码。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# 仓库根 = 本文件 (agent/tests/xxx.py) 向上两级到 /workspace，与 git ls-files 输出基准一致
REPO_ROOT = Path(__file__).resolve().parents[2]

# 只扫描的代码文件后缀（排除 .md 文档、.txt、二进制等）
CODE_SUFFIXES = {
    ".py", ".rs", ".toml", ".ts", ".tsx", ".js", ".yml", ".yaml", ".json", ".sh",
}

# R1: Windows 用户绝对路径 —— 机器专属硬编码的确定性特征
WINDOWS_USER_PATH_RE = re.compile(r"C:[\\/]Users[\\/]", re.IGNORECASE)

# R2: 机器专属 token 黑名单。只许缩、不许涨 —— 发现新机器名才加。
MACHINE_TOKENS: set[str] = {"wanzh"}

# 冻结清单：合法出现这些模式的文件（相对仓库根）。只许缩、不许涨。
# 值 = 为什么这个文件可以合法包含该模式。
ALLOWED_HARDCODED_MATCHES: dict[str, str] = {
    # 隐私守卫用 C:\\Users\\ 作为"路径类敏感特征"的正则，是检测逻辑而非硬编码。
    "agent/huginn/privacy_guard.py": "隐私检测正则，C:\\\\Users\\\\ 是特征模式，非硬编码路径",
    # LAMMPS 可执行文件的通配 glob 模式（C:\\Users\\*），非具体机器路径。
    "agent/huginn/tools/sim/lammps_tool.py": "LAMMPS 查找通配 glob (C:\\\\Users\\\\*)，非具体机器路径",
    # 测试数据：验证编码/转义处理，非真实路径。
    "agent/tests/test_i18n_encoding.py": "测试数据 (C:\\\\Users\\\\test)，非真实路径",
    # WSL↔Windows 路径转换工具的单测：C:\\Users\\ 是输入/期望夹具 (
    # /mnt/c 与 C:\\ 双向映射), 非机器专属路径, 与被测对象 test_i18n_encoding.py 同类。
    "agent/tests/test_wslpath.py": "WSL↔Windows 路径转换测试夹具，非真实机器路径",
    # 本守卫自身的 test_gate_self_test 用字面量验证 R1/R2 正则，非真实路径。
    "agent/tests/test_arch_no_hardcoded_paths.py": "网关自检数据 (C:\\\\Users\\\\wanzh)，非真实路径",
}


def _tracked_code_files() -> list[Path]:
    """返回所有被 git 跟踪的代码文件（相对 REPO_ROOT）。

    用 `git ls-files` 精确排除 .gitignore 的本地残余与历史已删文件。
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    files: list[Path] = []
    for raw in proc.stdout.split("\0"):
        if not raw:
            continue
        p = Path(raw)
        if p.suffix in CODE_SUFFIXES and "node_modules" not in p.parts:
            # 解析为基于 REPO_ROOT 的绝对路径，保证任何 cwd 下 exists()/read 都正确
            files.append((REPO_ROOT / p).resolve())
    return files


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def test_no_windows_user_paths_in_tracked_code():
    """R1：git 跟踪的代码文件不得含 C:\\Users\\ 绝对路径。"""
    violations: list[str] = []
    for path in _tracked_code_files():
        rel = _rel(path)
        if rel in ALLOWED_HARDCODED_MATCHES:
            continue
        if not path.exists():
            continue  # index 残留但工作区已删的文件，跳过
        text = path.read_text(encoding="utf-8", errors="ignore")
        if WINDOWS_USER_PATH_RE.search(text):
            violations.append(rel)
    assert not violations, (
        "检测到新增『Windows 用户绝对路径』硬编码 (C:\\Users\\...)，违反机器无关原则。"
        "请改为 env 变量展开或相对路径，不要提交机器专属路径，"
        "也不要往 ALLOWED_HARDCODED_MATCHES 里加条目：\n  "
        + "\n  ".join(violations)
    )


def test_no_machine_tokens_in_tracked_code():
    """R2：git 跟踪的代码文件不得含黑名单机器 token。"""
    if not MACHINE_TOKENS:
        return
    violations: list[str] = []
    for path in _tracked_code_files():
        rel = _rel(path)
        if rel in ALLOWED_HARDCODED_MATCHES:
            continue
        if not path.exists():
            continue  # index 残留但工作区已删的文件，跳过
        text = path.read_text(encoding="utf-8", errors="ignore")
        for tok in MACHINE_TOKENS:
            if tok in text:
                violations.append(f"{rel} (token={tok})")
    assert not violations, (
        "检测到新增『机器专属 token』硬编码，违反机器无关原则。"
        "请移除该 token（通常是某台机器的用户名/路径片段），"
        "不要往 MACHINE_TOKENS 里加条目：\n  "
        + "\n  ".join(violations)
    )


def test_allowlist_reasons_documented():
    """冻结清单每条都要有理由，防止无脑加白名单。"""
    for rel, reason in ALLOWED_HARDCODED_MATCHES.items():
        assert reason.strip(), f"{rel}: ALLOWED_HARDCODED_MATCHES 缺 reason"


def test_gate_self_test():
    """门禁自检：确认识别逻辑能抓到已知硬编码（防止门禁自己失效）。"""
    assert WINDOWS_USER_PATH_RE.search(r"C:\Users\wanzh\Desktop"), "R1 正则失效"
    assert WINDOWS_USER_PATH_RE.search("C:/Users/wanzh/Desktop"), "R1 正斜杠变体失效"
    assert "wanzh" in MACHINE_TOKENS, "R2 黑名单为空，门禁失效"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
