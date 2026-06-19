from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huginn.config import HuginnConfig
from huginn.security import (
    AuditLogger,
    SafeEvalError,
    SandboxConfig,
    SandboxError,
    SandboxExecutor,
    SandboxResult,
    safe_eval,
)

# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------


class TestSandboxExecutor:
    def test_dry_run(self):
        cfg = SandboxConfig(dry_run=True, allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        import sys
        result = sandbox.run([sys.executable, "--version"])
        assert result.dry_run is True
        assert "--version" in result.stdout
        assert result.returncode == 0

    def test_whitelist_blocks_unknown_executable(self):
        # Use an executable that exists but is not in the whitelist
        import sys
        cfg = SandboxConfig(allowed_executables={"lake", "lean"})
        sandbox = SandboxExecutor(cfg)
        with pytest.raises(SandboxError, match="not in sandbox whitelist"):
            sandbox.run([sys.executable, "-c", "print('blocked')"])

    def test_string_command_forbidden(self):
        sandbox = SandboxExecutor()
        with pytest.raises(SandboxError, match="String commands are forbidden"):
            sandbox.run("python script.py")  # type: ignore[arg-type]

    def test_empty_command_forbidden(self):
        sandbox = SandboxExecutor()
        with pytest.raises(SandboxError):
            sandbox.run([])

    def test_timeout_clamping(self):
        import sys
        cfg = SandboxConfig(max_timeout=5.0, allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        # Command that sleeps longer than allowed
        result = sandbox.run(
            [sys.executable, "-c", "import time; time.sleep(10)"], timeout=2.0
        )
        assert result.success is False
        assert result.returncode == -1  # timeout marker

    def test_real_execution_success(self):
        import sys
        cfg = SandboxConfig(allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        result = sandbox.run([sys.executable, "-c", "print('hello sandbox')"])
        assert result.success is True
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout

    def test_cwd_validation_strict(self):
        cfg = SandboxConfig(
            strict_work_dir=True,
            allowed_work_dirs={Path("/tmp")},
            allowed_executables={"python", "python3"},
        )
        sandbox = SandboxExecutor(cfg)
        import sys
        with pytest.raises(SandboxError, match="outside allowed roots"):
            sandbox.run([sys.executable, "-c", "pass"], cwd="/etc")

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
        import hashlib

        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        e1 = audit.log("a", "u", "x")
        e2 = audit.log("b", "u", "y")
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        expected = hashlib.sha256(lines[0].encode("utf-8")).hexdigest()[:32]
        assert e2.prev_hash == expected

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
        assert (
            ";" not in joined.split("12345")[1] or "'12345; cat /etc/passwd'" in joined
        )
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
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden import: os"):
            validate_code("import os\nos.system('ls')")

    def test_forbidden_import_subprocess(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden import: subprocess"):
            validate_code("from subprocess import run\nrun([''])")

    def test_forbidden_builtin_eval(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden builtin call: eval"):
            validate_code("eval('1+1')")

    def test_forbidden_builtin_exec(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden builtin call: exec"):
            validate_code("exec('print(1)')")

    def test_forbidden_attribute_subclasses(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden attribute access"):
            validate_code("(1).__class__.__subclasses__()")

    def test_empty_code_rejected(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Empty code"):
            validate_code("")
        with pytest.raises(RestrictedPythonError, match="Empty code"):
            validate_code("   ")
    def test_forbidden_attribute_method_call(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Forbidden method call"):
            validate_code("obj.__import__('os')")

    def test_syntax_error(self):
        from huginn.security import RestrictedPythonError, validate_code

        with pytest.raises(RestrictedPythonError, match="Syntax error"):
            validate_code("if x ==")  # incomplete syntax


# ---------------------------------------------------------------------------
# ContainerExecutor
# ---------------------------------------------------------------------------

class TestContainerExecutor:
    def test_invalid_runtime(self):
        from huginn.security.container_executor import ContainerExecutor

        with pytest.raises(ValueError, match="Unsupported container runtime"):
            ContainerExecutor("invalid", "image")

    def test_build_command_docker(self):
        from huginn.security.container_executor import ContainerExecutor

        ce = ContainerExecutor("docker", "test-image")
        cmd = ce._build_command("/usr/bin/docker", ["python", "script.py"], Path("/work"), {"FOO": "bar"})
        assert "docker" in cmd[0]
        assert "run" in cmd
        assert "--rm" in cmd
        assert "test-image" in cmd
        assert any("FOO=bar" in a for a in cmd)

    def test_build_command_podman(self):
        from huginn.security.container_executor import ContainerExecutor

        ce = ContainerExecutor("podman", "test-image")
        cmd = ce._build_command("/usr/bin/podman", ["python", "script.py"], Path("/work"), {})
        assert "podman" in cmd[0]
        assert "run" in cmd
        assert "--rm" in cmd

    def test_build_command_apptainer(self):
        from huginn.security.container_executor import ContainerExecutor

        ce = ContainerExecutor("apptainer", "test-image.sif")
        cmd = ce._build_command("/usr/bin/apptainer", ["python", "script.py"], Path("/work"), {"FOO": "bar"})
        assert "apptainer" in cmd[0]
        assert "exec" in cmd
        assert "test-image.sif" in cmd
        assert any("FOO=bar" in a for a in cmd)

    def test_dry_run(self):
        from huginn.security.container_executor import ContainerExecutor

        ce = ContainerExecutor("docker", "test-image", sandbox_config=SandboxConfig(dry_run=True))
        import sys
        result = ce.run([sys.executable, "-c", "print(1)"])
        assert result.dry_run is True
        assert result.returncode == 0

    def test_runtime_not_found(self):
        from huginn.security.container_executor import ContainerExecutor

        with patch("shutil.which", return_value=None):
            ce = ContainerExecutor("docker", "test-image")
            import sys
            result = ce.run([sys.executable, "-c", "print(1)"])
        assert result.success is False
        assert "not found in PATH" in result.stderr

    def test_timeout(self):
        from huginn.security.container_executor import ContainerExecutor

        with patch("shutil.which", return_value="/fake/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(
                    cmd=["docker"], timeout=1
                )
                ce = ContainerExecutor("docker", "test-image")
                import sys
                result = ce.run([sys.executable, "-c", "pass"], timeout=0.5)
        assert result.success is False
        assert result.returncode == -1


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

class TestSecrets:
    def test_env_backend_get(self, monkeypatch):
        from huginn.security.secrets import EnvSecretBackend

        monkeypatch.setenv("TEST_SECRET", "shh")
        backend = EnvSecretBackend()
        assert backend.get("TEST_SECRET") == "shh"
        assert backend.get("MISSING") is None

    def test_env_backend_set(self, monkeypatch):
        from huginn.security.secrets import EnvSecretBackend

        backend = EnvSecretBackend()
        backend.set("NEW_SECRET", "value")
        assert os.environ.get("NEW_SECRET") == "value"
        del os.environ["NEW_SECRET"]

    def test_vault_backend_missing_hvac(self, monkeypatch):
        from huginn.security.secrets import VaultSecretBackend

        monkeypatch.delenv("HUGINN_VAULT_ADDR", raising=False)
        monkeypatch.delenv("HUGINN_VAULT_TOKEN", raising=False)
        backend = VaultSecretBackend()
        with pytest.raises(ImportError, match="pip install hvac"):
            backend.get("secret")

    def test_vault_backend_mocked(self, monkeypatch):
        from huginn.security.secrets import VaultSecretBackend

        vault = MagicMock()
        vault.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"value": "secret_value"}}
        }
        with patch.dict("sys.modules", {"hvac": MagicMock(Client=MagicMock(return_value=vault))}):
            backend = VaultSecretBackend("http://vault:8200", "token")
            assert backend.get("mysecret:value") == "secret_value"
            # Without colon, key defaults to "value" and the mock returns it for all paths

    def test_get_secret_backend_default(self):
        from huginn.security.secrets import EnvSecretBackend, get_secret_backend

        backend = get_secret_backend()
        assert isinstance(backend, EnvSecretBackend)

    def test_get_secret_backend_vault(self, monkeypatch):
        from huginn.security.secrets import VaultSecretBackend, get_secret_backend

        monkeypatch.setenv("HUGINN_SECRET_BACKEND", "vault")
        with patch("huginn.security.secrets.VaultSecretBackend", return_value=MagicMock(spec=VaultSecretBackend)):
            backend = get_secret_backend()
            assert backend is not None


# ---------------------------------------------------------------------------
# SafeEval extended
# ---------------------------------------------------------------------------

class TestSafeEvalExtended:
    def test_bool_op(self):
        assert safe_eval("True and False") is False
        assert safe_eval("True or False") is True
        assert safe_eval("True and True and False") is False

    def test_tuple_list_dict_set(self):
        assert safe_eval("(1, 2, 3)") == (1, 2, 3)
        assert safe_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_eval("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}
        assert safe_eval("{1, 2, 3}") == {1, 2, 3}

    def test_subscript(self):
        assert safe_eval("[1, 2, 3][1]") == 2
        assert safe_eval("{'a': 1}['a']") == 1

    def test_unsupported_node(self):
        with pytest.raises(SafeEvalError, match="Forbidden expression construct"):
            safe_eval("lambda x: x + 1")

    def test_forbidden_call(self):
        with pytest.raises(SafeEvalError, match="Function calls are forbidden"):
            safe_eval("abs(-5)")

    def test_forbidden_attribute(self):
        with pytest.raises(SafeEvalError, match="Attribute access is forbidden"):
            safe_eval("(1).__class__")

    def test_invalid_syntax(self):
        with pytest.raises(SafeEvalError, match="Invalid syntax"):
            safe_eval("1 +")

    def test_undefined_name(self):
        with pytest.raises(SafeEvalError, match="Undefined name"):
            safe_eval("unknown")

    def test_compare_chains(self):
        assert safe_eval("1 < 2 < 3") is True
        assert safe_eval("1 < 2 > 3") is False
        assert safe_eval("1 == 1 == 1") is True

    def test_not_in(self):
        assert safe_eval("1 not in [2, 3]") is True
        assert safe_eval("1 not in [1, 2]") is False

    def test_is_isnot(self):
        assert safe_eval("None is None") is True
        assert safe_eval("1 is not None") is True

    def test_floor_div_mod(self):
        assert safe_eval("7 // 2") == 3
        assert safe_eval("7 % 2") == 1

    def test_unary_ops(self):
        assert safe_eval("-5") == -5
        assert safe_eval("+5") == 5
        assert safe_eval("~5") == -6
        assert safe_eval("not True") is False

    def test_if_expression(self):
        assert safe_eval("10 if 5 > 3 else 0") == 10
        assert safe_eval("10 if 5 < 3 else 0") == 0

    def test_in_operator(self):
        assert safe_eval("1 in [1, 2, 3]") is True
        assert safe_eval("4 in [1, 2, 3]") is False

    def test_complex_comparison(self):
        assert safe_eval("5 > 3 and 2 < 4") is True
        assert safe_eval("5 > 3 or 2 > 4") is True

    def test_dict_with_expr(self):
        assert safe_eval("{'a': 1 + 2, 'b': 3 * 4}") == {"a": 3, "b": 12}

    def test_list_with_expr(self):
        assert safe_eval("[1 + 2, 3 * 4, 5 - 1]") == [3, 12, 4]

    def test_tuple_with_expr(self):
        assert safe_eval("(1 + 2, 3 * 4)") == (3, 12)

    def test_set_with_expr(self):
        assert safe_eval("{1 + 2, 3 * 4}") == {3, 12}

    def test_subscript_dict_expr(self):
        assert safe_eval("{'a': 1 + 2}['a']") == 3

    def test_subscript_list_expr(self):
        assert safe_eval("[1, 2, 3][1 + 1]") == 3

    def test_subscript_tuple_expr(self):
        assert safe_eval("(1, 2, 3)[0]") == 1

    def test_unsupported_binop(self):
        with pytest.raises(SafeEvalError, match="Unsupported binary operator"):
            safe_eval("1 << 2")  # Bit shift is not allowed


# ---------------------------------------------------------------------------
# Audit extended
# ---------------------------------------------------------------------------

class TestAuditExtended:
    def test_verify_chain_empty(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        assert audit.verify_chain() == []

    def test_verify_chain_valid(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        audit.log("a", "u", "x")
        audit.log("b", "u", "y")
        assert audit.verify_chain() == []

    def test_verify_chain_bad_json(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("not json\n", encoding="utf-8")
        audit = AuditLogger(str(log_file))
        mismatches = audit.verify_chain()
        assert len(mismatches) == 1
        assert mismatches[0][1] == "valid_json"

    def test_verify_chain_empty_lines(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        audit.log("a", "u", "x")
        with open(log_file, "a") as f:
            f.write("\n\n")
        audit.log("b", "u", "y")
        assert audit.verify_chain() == []

    def test_log_with_output_hash(self, tmp_path: Path):
        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(str(log_file))
        event = audit.log("tool_call", "user", "test", input_data="input", output_data="output")
        assert event.input_hash is not None
        assert event.output_hash is not None
        assert event.input_hash != event.output_hash

    def test_log_none_path(self, tmp_path: Path):
        import os
        orig_dir = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            audit = AuditLogger()
            assert audit.log_path.name == "huginn_audit.jsonl"
            audit.log("test", "u", "a")
            assert audit.log_path.exists()
        finally:
            os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# Sandbox extended
# ---------------------------------------------------------------------------

class TestSandboxExtended:
    def test_resolve_executable_empty(self):
        sandbox = SandboxExecutor()
        with pytest.raises(SandboxError, match="Empty command"):
            sandbox._resolve_executable([])

    def test_validate_cwd_strict_empty_dirs(self):
        cfg = SandboxConfig(strict_work_dir=True, allowed_work_dirs={Path("/nonexistent")})
        sandbox = SandboxExecutor(cfg)
        # Only /nonexistent is allowed, so /tmp should be rejected
        import sys
        with pytest.raises(SandboxError, match="outside allowed roots"):
            sandbox.run([sys.executable, "-c", "pass"], cwd="/tmp")

    def test_string_truncation(self):
        cfg = SandboxConfig(allowed_executables={"python", "python3"}, max_output_bytes=10)
        sandbox = SandboxExecutor(cfg)
        import sys
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "01234567890123456789"
            mock_run.return_value.stderr = "abcdefghijklmnopqrstuvwxyz"
            result = sandbox.run([sys.executable, "-c", "print(1)"])
            assert "truncated" in result.stdout
            assert "truncated" in result.stderr

    def test_remote_kwargs_filtered(self):
        cfg = SandboxConfig(allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        import sys
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            sandbox.run([sys.executable, "-c", "print(1)"], queue="normal", walltime="1:00:00")
            # Ensure remote kwargs are not passed to subprocess.run
            call_kwargs = mock_run.call_args[1]
            assert "queue" not in call_kwargs
            assert "walltime" not in call_kwargs

    def test_env_passed(self):
        cfg = SandboxConfig(allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        import sys
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            sandbox.run([sys.executable, "-c", "print(1)"], env={"FOO": "bar"})
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs.get("env") == {"FOO": "bar"}

    def test_dry_run_with_custom_config(self):
        cfg = SandboxConfig(dry_run=True, allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        import sys
        result = sandbox.run([sys.executable, "-c", "print(1)"])
        assert result.dry_run is True
        assert result.returncode == 0

    def test_sandbox_result_defaults(self):
        result = SandboxResult(success=True, returncode=0, stdout="", stderr="", command=[], dry_run=False)
        assert result.blocked is False
        assert result.block_reason is None
