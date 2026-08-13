"""制度化闭环：mat-db MCP 真实 Materials Project 路径可达性。

mat-db server 无 MP_API_KEY 时会回退 mock；真实路径要走 mp-api 包。本门禁保证：
  1. server 真实查询路径引用了 MP_API_KEY（不会退化掉）；
  2. pyproject 声明了 `db` 可选依赖组含 mp-api（真实路径可安装）；
  3. bulk_modulus 提取无运算符优先级隐患（回归保护）。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
MATDB_SERVER = REPO_ROOT / "servers" / "mat-db-mcp" / "server.py"
PYPROJECT = AGENT_ROOT / "pyproject.toml"


def test_matdb_real_path_keyed_by_mp_api():
    src = MATDB_SERVER.read_text(encoding="utf-8")
    assert 'os.environ.get("MP_API_KEY")' in src, "mat-db 真实路径必须读 MP_API_KEY"
    assert "from mp_api.client import MPRester" in src, "mat-db 真实路径必须用 MPRester"


def test_mp_api_declared_as_optional_dependency():
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    extra = data["project"]["optional-dependencies"]
    assert "db" in extra, "pyproject 缺少 [project.optional-dependencies].db 组"
    assert any("mp-api" in item for item in extra["db"]), "db 组必须含 mp-api"


def test_matdb_search_does_not_pass_limit_kwarg():
    """mp-api 新版 search() 不接收 limit(那是 _search 的 kwarg), 传了会抛错被吞成 mock。

    回归保护: formula 分支必须用 `search(formula=formula)` + 取首条, 不能带 limit=。
    """
    src = MATDB_SERVER.read_text(encoding="utf-8")
    assert "search(formula=formula)" in src, "formula 分支必须用 search(formula=formula)"
    assert "search(formula=formula, limit=" not in src, "search() 禁止传 limit kwarg"


def test_matdb_bulk_modulus_precedence_is_safe():
    src = MATDB_SERVER.read_text(encoding="utf-8")
    # 抽到 bulk_modulus 提取行，确认布尔条件已加括号避免 and/or 优先级歧义
    for line in src.splitlines():
        if "bulk_modulus" in line and "props" in line and "doc" in line:
            assert " or (" in line, (
                "bulk_modulus 提取条件必须显式括号（`in props or (not props and ...)`），"
                "否则 and/or 优先级会引入 bug"
            )
            return
    raise AssertionError("未找到 bulk_modulus 真实提取行")
