"""Tests for the security layer: sandbox, audit, safe_eval, config key resolution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from huginn.security import (
    SandboxConfig,
    SandboxError,
    SandboxExecutor,
    AuditLogger,
    safe_eval,
    SafeEvalError,
)
from huginn.config import HuginnConfig


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------

class TestSandboxExecutor:
    def test_dry_run(self):
        cfg = SandboxConfig(dry_run=True)
        sandbox = SandboxExecutor(cfg)
        result = sandbox.run(["python", "--version"])
        assert result.dry_run is True
        assert "python --version" in result.stdout
        assert result.returncode == 0

    def test_whitelist_blocks_unknown_executable(self):
        # Python exists on PATH but is not in this strict whitelist
        cfg = SandboxConfig(allowed_executables={"lake", "lean"})
        sandbox = SandboxExecutor(cfg)
        with pytest.raises(SandboxError, match="not in sandbox whitelist"):
            sandbox.run(["python", "-c", "print('blocked')"])

    def test_string_command_forbidden(self):
        sandbox = SandboxExecutor()
        with pytest.raises(SandboxError, match="String commands are forbidden"):
            sandbox.run("python script.py")  # type: ignore[arg-type]

    def test_empty_command_forbidden(self):
        sandbox = SandboxExecutor()
        with pytest.raises(SandboxError):
            sandbox.run([])

    def test_timeout_clamping(self):
        cfg = SandboxConfig(max_timeout=5.0)
        sandbox = SandboxExecutor(cfg)
        # Command that sleeps longer than allowed
        result = sandbox.run(["python", "-c", "import time; time.sleep(10)"], timeout=2.0)
        assert result.success is False
        assert result.returncode == -1  # timeout marker

    def test_real_execution_success(self):
        sandbox = SandboxExecutor()
        result = sandbox.run(["python", "-c", "print('hello sandbox')"])
        assert result.success is True
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout

    def test_cwd_validation_strict(self):
        cfg = SandboxConfig(
            strict_work_dir=True,
            allowed_work_dirs={Path("/tmp")},
        )
        sandbox = SandboxExecutor(cfg)
        with pytest.raises(SandboxError, match="outside allowed roots"):
            sandbox.run(["python", "-c", "pass"], cwd="/etc")

    def test_hash_data(self):
        h1 = SandboxExecutor.hash_data("test")
        h2 = SandboxExecutor.hash_data("test")
        h3 = SandboxExecutor.hash_data("different")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_log_append(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        event = audit.log("tool_call", "user", "vasp_relax", input_data="incar")
        assert event.event_type == "tool_call"
        assert event.action == "vasp_relax"
        assert event.input_hash is not None
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["actor"] == "user"

    def test_log_chain(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        e1 = audit.log("a", "u", "x")
        e2 = audit.log("b", "u", "y")
        assert e2.prev_hash == audit._compute_hash(e1)

    def test_multiple_events(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        for i in range(5):
            audit.log("test", "user", f"action_{i}")
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# safe_eval
# ---------------------------------------------------------------------------

class TestSafeEval:
    def test_basic_math(self):
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20
        assert safe_eval("10 / 2") == 5.0

    def test_comparisons(self):
        assert safe_eval("5 > 3") is True
        assert safe_eval("5 == 3") is False
        assert safe_eval("3 <= 3") is True

    def test_boolean_logic(self):
        assert safe_eval("True and False") is False
        assert safe_eval("True or False") is True
        assert safe_eval("not False") is True

    def test_undefined_name(self):
        with pytest.raises(SafeEvalError, match="Undefined name"):
            safe_eval("undefined_var + 1")

    def test_forbidden_call(self):
        with pytest.raises(SafeEvalError, match="Function calls are forbidden"):
            safe_eval("__import__('os').system('rm -rf /')")

    def test_forbidden_attribute(self):
        with pytest.raises(SafeEvalError, match="Attribute access is forbidden"):
            safe_eval("(1).__class__")

    def test_locals(self):
        assert safe_eval("x + y", {"x": 10, "y": 20}) == 30

    def test_list_dict(self):
        assert safe_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_eval("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    def test_if_expression(self):
        assert safe_eval("10 if 5 > 3 else 0") == 10
        assert safe_eval("10 if 5 < 3 else 0") == 0


# ---------------------------------------------------------------------------
# Config key resolution
# ---------------------------------------------------------------------------

class TestConfigKeyResolution:
    def test_plain_key(self):
        assert HuginnConfig.resolve_key("secret123") == "secret123"

    def test_none_key(self):
        assert HuginnConfig.resolve_key(None) is None

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("HUGINN_TEST_KEY", "resolved_value")
        assert HuginnConfig.resolve_key("env:HUGINN_TEST_KEY") == "resolved_value"

    def test_env_prefix_missing(self):
        assert HuginnConfig.resolve_key("env:NONEXISTENT_VAR_XYZ") is None

    def test_resolved_api_key_property(self):
        cfg = HuginnConfig(api_key="env:TEST_KEY")
        os.environ["TEST_KEY"] = "hidden"
        assert cfg.resolved_api_key == "hidden"
        del os.environ["TEST_KEY"]

    def test_to_dict_mask_key(self):
        cfg = HuginnConfig(api_key="secret")
        d = cfg.to_dict(mask_key=True)
        assert d["api_key"] == "***"
        d2 = cfg.to_dict(mask_key=False)
        assert d2["api_key"] == "secret"

    def test_save_load_json(self, tmp_path: Path):
        cfg = HuginnConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        path = tmp_path / "cfg.json"
        cfg.save(str(path), format="json")
        loaded = HuginnConfig.load(str(path), format="json")
        assert loaded.provider == "openai"
        assert loaded.model == "gpt-4o"
        assert loaded.api_key == "sk-test"

    def test_from_dict_ignores_unknown(self):
        d = {"provider": "ollama", "unknown_field": 123}
        cfg = HuginnConfig.from_dict(d)
        assert cfg.provider == "ollama"


# ---------------------------------------------------------------------------
# HPC client input sanitization
# ---------------------------------------------------------------------------

class TestHPCSanitization:
    def test_sanitize_job_name_removes_special_chars(self):
        from huginn.hpc.client import _sanitize_job_name
        assert _sanitize_job_name("my-job_1") == "my-job_1"
        assert _sanitize_job_name("job; rm -rf /") == "job__rm_-rf__"
        assert _sanitize_job_name("job|cat /etc/passwd") == "job_cat__etc_passwd"

    def test_sanitize_job_name_truncation(self):
        from huginn.hpc.client import _sanitize_job_name
        long_name = "a" * 100
        assert len(_sanitize_job_name(long_name)) == 64

    def test_sanitize_job_name_invalid(self):
        from huginn.hpc.client import _sanitize_job_name
        with pytest.raises(ValueError):
            _sanitize_job_name("|")  # becomes "_" which is rejected

    def test_validate_path_component_blocks_metacharacters(self):
        from huginn.hpc.client import _validate_path_component
        _validate_path_component("/home/user/work")  # OK
        with pytest.raises(ValueError, match="forbidden characters"):
            _validate_path_component("/home/user; rm -rf /")
        with pytest.raises(ValueError, match="forbidden characters"):
            _validate_path_component("/home/user|cat /etc/passwd")
        with pytest.raises(ValueError, match="forbidden characters"):
            _validate_path_component("/home/user`whoami`")
        with pytest.raises(ValueError, match="forbidden characters"):
            _validate_path_component("$HOME/malicious")

    def test_shlex_join_prevents_injection(self):
        import shlex
        # Simulate how _exec quotes list arguments
        malicious_job_id = "12345; cat /etc/passwd"
        cmd = ["sacct", "-j", malicious_job_id, "--format=JobID,State"]
        joined = shlex.join(cmd)
        assert ";" not in joined.split("12345")[1] or "'12345; cat /etc/passwd'" in joined
        # Ensure the malicious string is treated as a single argument
        assert shlex.split(joined)[2] == malicious_job_id


# ---------------------------------------------------------------------------
# Restricted Python validator
# ---------------------------------------------------------------------------

class TestRestrictedPython:
    def test_valid_code_passes(self):
        from huginn.security import validate_code
        validate_code("x = 2 + 3\nprint(x)")
        validate_code("import math\nprint(math.sin(0))")

    def test_forbidden_import_os(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Forbidden import: os"):
            validate_code("import os\nos.system('ls')")

    def test_forbidden_import_subprocess(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Forbidden import: subprocess"):
            validate_code("from subprocess import run\nrun([''])")

    def test_forbidden_builtin_eval(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Forbidden builtin call: eval"):
            validate_code("eval('1+1')")

    def test_forbidden_builtin_exec(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Forbidden builtin call: exec"):
            validate_code("exec('print(1)')")

    def test_forbidden_attribute_subclasses(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Forbidden attribute access"):
            validate_code("(1).__class__.__subclasses__()")

    def test_empty_code_rejected(self):
        from huginn.security import validate_code, RestrictedPythonError
        with pytest.raises(RestrictedPythonError, match="Empty code"):
            validate_code("")
        with pytest.raises(RestrictedPythonError, match="Empty code"):
            validate_code("   ")
