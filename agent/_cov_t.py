import sys
import runpy
import coverage
import pytest

import types
_np = runpy.run_path("tests/_fakenp_plugin.py")
sys.modules["numpy"] = _np["_build"]()

cov = coverage.Coverage(branch=True, source=["huginn.tools.sim.transolver_tool"])
cov.start()
rc = pytest.main(
    ["-q", "-p", "no:cov", "-o", "addopts=", "tests/test_transolver_tool_integration_ext.py"]
)
cov.stop()
cov.save()
try:
    cov.report(show_missing=True, skip_covered=True)
except Exception as e:
    print("report err:", e)
sys.exit(rc)