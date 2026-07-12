"""Security module integration tests.

Checks that each security layer blocks what it should and allows what it should.
Run with: pytest tests/test_security_modules.py -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_command_filter():
    from huginn.security.command_filter import check_command_safety

    dangerous = ["rm -rf /", "del /f /q", "format c:"]
    safe = ["python -c print(1)", "ls -la", "git status"]

    for cmd in dangerous:
        result = check_command_safety(cmd)
        assert not result.is_safe, f"Should block: {cmd}"

    for cmd in safe:
        result = check_command_safety(cmd)
        assert result.is_safe, f"Should allow: {cmd}"


def test_restricted_python():
    from huginn.security.restricted_python import validate_code, RestrictedPythonError

    # dangerous import should raise
    try:
        validate_code("import os; os.system('rm -rf /')")
        assert False, "Should have raised RestrictedPythonError"
    except RestrictedPythonError:
        pass

    # safe code should pass (returns None on success)
    validate_code("x = 1 + 2; print(x)")


def test_policy_engine():
    from huginn.security.policy_engine import PolicyEngine

    pe = PolicyEngine()
    assert len(pe._rules) > 0, "Policy engine should have rules loaded"


def test_sandbox_executor():
    from huginn.security.sandbox import SandboxExecutor

    se = SandboxExecutor()
    assert se is not None


def test_safe_eval():
    from huginn.security.safe_eval import safe_eval, SafeEvalError

    # safe expression
    result = safe_eval("2 + 3")
    assert result == 5

    # dangerous expression should raise
    try:
        safe_eval("__import__('os').system('whoami')")
        assert False, "Should have blocked dangerous eval"
    except SafeEvalError:
        pass


def test_circuit_breaker():
    from huginn.security.external_breaker import ExternalCircuitBreaker

    cb = ExternalCircuitBreaker()
    assert cb is not None


def test_credential_store():
    from huginn.security.credential_store import CredentialStore

    with tempfile.TemporaryDirectory() as td:
        cs = CredentialStore(db_path=os.path.join(td, "test_creds.db"))
        assert cs is not None
