import runpy
import sys

import coverage
import pytest

_np = runpy.run_path("tests/_fakenp_plugin.py")
sys.modules["numpy"] = _np["_build"]()

# 清空 SQLite 缓存, 避免上次运行的 "File not found" 结果被缓存命中,
# 导致 _call_cached 的 not-exists 分支 (line 248) 测不到.
from huginn.tools.tool_cache import ToolCache

ToolCache.shared().clear()

cov = coverage.Coverage(branch=True, source=["huginn.tools.structure_tool"])
cov.start()
rc = pytest.main(
    ["-q", "-p", "no:cov", "-o", "addopts=", "tests/test_structure_tool_ext.py"]
)
cov.stop()
cov.save()
try:
    cov.report(show_missing=True)
except Exception as e:
    print("report err:", e)
sys.exit(rc)
