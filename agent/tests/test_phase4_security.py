"""Phase 4 tests — RBAC, container hardening, audit signing, secret manager."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
#  RBAC tests  (rbac.py)
# =========================================================================

class TestRole:
    def test_role_values(self):
        from huginn.security.rbac import Role
        assert Role.VIEWER.value == "viewer"
        assert Role.OPERATOR.value == "operator"
        assert Role.ADMIN.value == "admin"

    def test_role_levels(self):
        from huginn.security.rbac import Role
        assert Role.VIEWER.level < Role.OPERATOR.level < Role.ADMIN.level

    def test_role_from_string(self):
        from huginn.security.rbac import Role
        assert Role("viewer") is Role.VIEWER
        assert Role("admin") is Role.ADMIN


class TestUser:
    def test_create_user(self):
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="alice", role=Role.OPERATOR)
        assert user.user_id == "u1"
        assert user.username == "alice"
        assert user.role is Role.OPERATOR
        assert user.active is True

    def test_can_viewer(self):
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="bob", role=Role.VIEWER)
        assert user.can("read")
        assert user.can("query")
        assert not user.can("execute")
        assert not user.can("admin")

    def test_can_operator(self):
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="carol", role=Role.OPERATOR)
        assert user.can("read")
        assert user.can("execute")
        assert user.can("write")
        assert not user.can("admin")

    def test_can_admin(self):
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="dave", role=Role.ADMIN)
        assert user.can("admin")
        assert user.can("config")
        assert user.can("user_manage")
        assert user.can("execute")

    def test_has_role_or_higher(self):
        from huginn.security.rbac import Role, User
        admin = User(user_id="u1", username="admin", role=Role.ADMIN)
        assert admin.has_role_or_higher(Role.VIEWER)
        assert admin.has_role_or_higher(Role.OPERATOR)
        assert admin.has_role_or_higher(Role.ADMIN)

        viewer = User(user_id="u2", username="viewer", role=Role.VIEWER)
        assert viewer.has_role_or_higher(Role.VIEWER)
        assert not viewer.has_role_or_higher(Role.OPERATOR)

    def test_to_dict_from_dict(self):
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="eve", role=Role.OPERATOR, metadata={"team": "matsci"})
        d = user.to_dict()
        assert d["role"] == "operator"
        assert d["metadata"]["team"] == "matsci"

        restored = User.from_dict(d)
        assert restored.user_id == user.user_id
        assert restored.role is Role.OPERATOR
        assert restored.metadata["team"] == "matsci"


class TestJWT:
    def test_encode_decode(self):
        from huginn.security.rbac import jwt_decode, jwt_encode
        secret = "test-secret-key-12345"
        payload = {"sub": "u1", "username": "alice", "role": "admin"}
        token = jwt_encode(payload, secret, expires_in=3600)
        assert isinstance(token, str)
        assert token.count(".") == 2

        decoded = jwt_decode(token, secret)
        assert decoded["sub"] == "u1"
        assert decoded["username"] == "alice"
        assert decoded["role"] == "admin"
        assert "iat" in decoded
        assert "exp" in decoded

    def test_wrong_secret(self):
        from huginn.security.rbac import jwt_decode, jwt_encode
        token = jwt_encode({"sub": "u1"}, "secret1")
        with pytest.raises(ValueError, match="signature mismatch"):
            jwt_decode(token, "wrong-secret")

    def test_expired_token(self):
        from huginn.security.rbac import jwt_decode, jwt_encode
        token = jwt_encode({"sub": "u1"}, "secret", expires_in=-1)
        with pytest.raises(ValueError, match="expired"):
            jwt_decode(token, "secret")

    def test_malformed_token(self):
        from huginn.security.rbac import jwt_decode
        with pytest.raises(ValueError, match="Malformed"):
            jwt_decode("not.a.valid.jwt", "secret")

    def test_bytes_secret(self):
        from huginn.security.rbac import jwt_decode, jwt_encode
        token = jwt_encode({"sub": "u1"}, b"byte-secret")
        decoded = jwt_decode(token, b"byte-secret")
        assert decoded["sub"] == "u1"


class TestAPIKeyGeneration:
    def test_generate_api_key(self):
        from huginn.security.rbac import generate_api_key
        key = generate_api_key()
        assert len(key) >= 48  # 48 bytes → ~64 urlsafe chars
        assert isinstance(key, str)

    def test_hash_api_key(self):
        from huginn.security.rbac import hash_api_key
        h = hash_api_key("test-key")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest
        # Deterministic
        assert hash_api_key("test-key") == h


class TestSessionManager:
    def test_create_and_get(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager(max_idle=60)
        session = mgr.create("u1")
        assert session.user_id == "u1"
        assert session.session_id

        retrieved = mgr.get(session.session_id)
        assert retrieved is session

    def test_destroy(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager()
        session = mgr.create("u1")
        mgr.destroy(session.session_id)
        assert mgr.get(session.session_id) is None

    def test_destroy_user_sessions(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager()
        s1 = mgr.create("u1")
        s2 = mgr.create("u1")
        s3 = mgr.create("u2")
        count = mgr.destroy_user_sessions("u1")
        assert count == 2
        assert mgr.get(s1.session_id) is None
        assert mgr.get(s2.session_id) is None
        assert mgr.get(s3.session_id) is not None

    def test_expired_session(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager(max_idle=0.01)  # 10ms
        session = mgr.create("u1")
        time.sleep(0.05)
        assert mgr.get(session.session_id) is None

    def test_cleanup_expired(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager(max_idle=0.01)
        mgr.create("u1")
        mgr.create("u2")
        time.sleep(0.05)
        cleaned = mgr.cleanup_expired()
        assert cleaned == 2
        assert mgr.active_count() == 0

    def test_active_count(self):
        from huginn.security.rbac import SessionManager
        mgr = SessionManager()
        mgr.create("u1")
        mgr.create("u2")
        assert mgr.active_count() == 2


# =========================================================================
#  User store tests  (user_store.py)
# =========================================================================

class TestUserStore:
    def _make_store(self, tmp_path: Path):
        from huginn.security.user_store import UserStore
        return UserStore(store_path=tmp_path / "users.json")

    def test_create_user(self, tmp_path):
        store = self._make_store(tmp_path)
        user, api_key = store.create_user("alice", role=__import__("huginn.security.rbac", fromlist=["Role"]).Role.ADMIN)
        assert user.username == "alice"
        assert len(api_key) > 20

    def test_get_user_by_username(self, tmp_path):
        store = self._make_store(tmp_path)
        store.create_user("bob")
        user = store.get_user_by_username("bob")
        assert user is not None
        assert user.username == "bob"

    def test_get_user_by_api_key(self, tmp_path):
        from huginn.security.rbac import Role
        store = self._make_store(tmp_path)
        user, api_key = store.create_user("carol", role=Role.OPERATOR)
        found = store.get_user_by_api_key(api_key)
        assert found is not None
        assert found.user_id == user.user_id

    def test_duplicate_username(self, tmp_path):
        store = self._make_store(tmp_path)
        store.create_user("dave")
        with pytest.raises(ValueError, match="already exists"):
            store.create_user("dave")

    def test_update_role(self, tmp_path):
        from huginn.security.rbac import Role
        store = self._make_store(tmp_path)
        user, _ = store.create_user("eve")
        assert user.role is Role.VIEWER
        store.update_role(user.user_id, Role.ADMIN)
        updated = store.get_user(user.user_id)
        assert updated.role is Role.ADMIN

    def test_deactivate_user(self, tmp_path):
        store = self._make_store(tmp_path)
        user, _ = store.create_user("frank")
        store.deactivate_user(user.user_id)
        u = store.get_user(user.user_id)
        assert u.active is False

    def test_delete_user(self, tmp_path):
        store = self._make_store(tmp_path)
        user, _ = store.create_user("grace")
        store.delete_user(user.user_id)
        assert store.get_user(user.user_id) is None
        assert store.get_user_by_username("grace") is None

    def test_list_users(self, tmp_path):
        from huginn.security.rbac import Role
        store = self._make_store(tmp_path)
        store.create_user("hank")
        store.create_user("iris")
        user3, _ = store.create_user("jack")
        store.deactivate_user(user3.user_id)

        all_users = store.list_users()
        assert len(all_users) == 3

        active_only = store.list_users(active_only=True)
        assert len(active_only) == 2

    def test_rotate_api_key(self, tmp_path):
        store = self._make_store(tmp_path)
        user, old_key = store.create_user("kate")
        new_key = store.rotate_api_key(user.user_id)
        assert new_key != old_key
        assert store.get_user_by_api_key(new_key) is not None
        assert store.get_user_by_api_key(old_key) is None

    def test_persistence(self, tmp_path):
        from huginn.security.rbac import Role
        from huginn.security.user_store import UserStore

        path = tmp_path / "users.json"
        store1 = UserStore(store_path=path)
        user, key = store1.create_user("leo", role=Role.OPERATOR)

        # New store instance should load from file
        store2 = UserStore(store_path=path)
        loaded = store2.get_user_by_username("leo")
        assert loaded is not None
        assert loaded.role is Role.OPERATOR
        assert store2.get_user_by_api_key(key) is not None


# =========================================================================
#  Container hardening tests  (container_executor.py)
# =========================================================================

class TestContainerSecurityConfig:
    def test_defaults(self):
        from huginn.security.container_executor import ContainerSecurityConfig
        cfg = ContainerSecurityConfig()
        assert cfg.network_none is True
        assert cfg.read_only_root is True
        assert cfg.no_new_privileges is True
        assert cfg.drop_all_capabilities is True
        assert cfg.run_as_user == "1000"
        assert cfg.memory_limit is None
        assert cfg.cpu_limit is None

    def test_custom_values(self):
        from huginn.security.container_executor import ContainerSecurityConfig
        cfg = ContainerSecurityConfig(
            network_none=False,
            memory_limit="2g",
            cpu_limit=4.0,
            pids_limit=100,
            require_digest=True,
            allowed_images={"python:3.11"},
        )
        assert cfg.network_none is False
        assert cfg.memory_limit == "2g"
        assert cfg.cpu_limit == 4.0
        assert cfg.pids_limit == 100
        assert cfg.require_digest is True


class TestDigestPinning:
    def test_is_digest_pinned_valid(self):
        from huginn.security.container_executor import _is_digest_pinned
        assert _is_digest_pinned(
            "python@sha256:" + "a" * 64
        )
        assert _is_digest_pinned(
            "ghcr.io/myorg/myimage@sha256:" + "b" * 64
        )

    def test_is_digest_pinned_invalid(self):
        from huginn.security.container_executor import _is_digest_pinned
        assert not _is_digest_pinned("python:3.11")
        assert not _is_digest_pinned("python:latest")
        assert not _is_digest_pinned("ubuntu")


class TestContainerHardening:
    def test_build_command_includes_security_flags(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(
            network_none=True,
            memory_limit="512m",
            cpu_limit=2.0,
            pids_limit=50,
            read_only_root=True,
            run_as_user="1000",
            no_new_privileges=True,
            drop_all_capabilities=True,
        )
        sandbox = SandboxConfig(dry_run=True)
        executor = ContainerExecutor(
            "docker",
            "python:3.11",
            sandbox_config=sandbox,
            security_config=sec,
        )
        cmd = executor._build_command(
            "/usr/bin/docker", ["python", "-c", "print(1)"],
            Path("/tmp"), {}
        )
        assert "--network" in cmd
        assert "none" in cmd
        assert "--memory" in cmd
        assert "512m" in cmd
        assert "--cpus" in cmd
        assert "2.0" in cmd
        assert "--pids-limit" in cmd
        assert "50" in cmd
        assert "--read-only" in cmd
        assert "--user" in cmd
        assert "1000" in cmd
        assert "--security-opt=no-new-privileges" in cmd
        assert "--cap-drop=ALL" in cmd

    def test_build_command_no_security(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(
            network_none=False,
            read_only_root=False,
            run_as_user=None,
            no_new_privileges=False,
            drop_all_capabilities=False,
        )
        sandbox = SandboxConfig(dry_run=True)
        executor = ContainerExecutor(
            "docker",
            "python:3.11",
            sandbox_config=sandbox,
            security_config=sec,
        )
        cmd = executor._build_command(
            "/usr/bin/docker", ["echo", "hi"],
            Path("/tmp"), {}
        )
        assert "--network" not in cmd
        assert "--read-only" not in cmd
        assert "--cap-drop=ALL" not in cmd

    def test_build_command_apptainer_network(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(network_none=True)
        sandbox = SandboxConfig(dry_run=True)
        executor = ContainerExecutor(
            "apptainer",
            "image.sif",
            sandbox_config=sandbox,
            security_config=sec,
        )
        cmd = executor._build_command(
            "/usr/bin/apptainer", ["ls"],
            Path("/tmp"), {}
        )
        assert "--net" in cmd
        assert "none" in cmd

    def test_require_digest_blocks_tag(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(require_digest=True)
        executor = ContainerExecutor(
            "docker",
            "python:3.11",  # tag, not digest
            sandbox_config=SandboxConfig(),
            security_config=sec,
        )
        result = executor.run(["echo", "hi"])
        assert result.success is False
        assert result.blocked is True
        assert result.block_reason == "image_not_pinned"

    def test_require_digest_allows_pinned(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        digest_image = "python@sha256:" + "a" * 64
        sec = ContainerSecurityConfig(require_digest=True)
        executor = ContainerExecutor(
            "docker",
            digest_image,
            sandbox_config=SandboxConfig(dry_run=True),
            security_config=sec,
        )
        result = executor.run(["echo", "hi"])
        assert result.success is True
        assert result.dry_run is True

    def test_allowed_images_blocks(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(
            require_digest=False,
            allowed_images={"alpine:3.18"},
        )
        executor = ContainerExecutor(
            "docker",
            "python:3.11",
            sandbox_config=SandboxConfig(),
            security_config=sec,
        )
        result = executor.run(["echo", "hi"])
        assert result.blocked is True
        assert result.block_reason == "image_not_allowed"

    def test_extra_mounts(self):
        from huginn.security.container_executor import (
            ContainerExecutor,
            ContainerSecurityConfig,
        )
        from huginn.security.sandbox import SandboxConfig

        sec = ContainerSecurityConfig(
            network_none=False,
            extra_mounts=["/data:/data:ro"],
        )
        executor = ContainerExecutor(
            "docker",
            "python:3.11",
            sandbox_config=SandboxConfig(dry_run=True),
            security_config=sec,
        )
        cmd = executor._build_command(
            "/usr/bin/docker", ["ls"],
            Path("/tmp"), {}
        )
        assert "/data:/data:ro" in cmd


# =========================================================================
#  Audit signing tests  (audit.py)
# =========================================================================

class TestAuditSigning:
    def test_log_with_signing_key(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path, signing_key=b"my-secret-key")
        event = logger.log("tool_call", "user", "test_action")
        assert event.signature is not None
        assert len(event.signature) == 32

    def test_log_without_signing_key(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        event = logger.log("tool_call", "user", "test_action")
        assert event.signature is None

    def test_verify_signatures_valid(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path, signing_key=b"secret")
        logger.log("tool_call", "user", "action1")
        logger.log("tool_call", "user", "action2")
        logger.log("tool_call", "user", "action3")

        issues = logger.verify_signatures()
        assert len(issues) == 0

    def test_verify_signatures_tampered(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path, signing_key=b"secret")
        logger.log("tool_call", "user", "action1")
        logger.log("tool_call", "user", "action2")

        # Tamper with line 1
        lines = log_path.read_text().splitlines()
        record = json.loads(lines[0])
        record["action"] = "tampered"
        lines[0] = json.dumps(record, sort_keys=True)
        log_path.write_text("\n".join(lines) + "\n")

        logger2 = AuditLogger(log_path, signing_key=b"secret")
        issues = logger2.verify_signatures()
        assert len(issues) > 0
        assert issues[0][1] == "signature_mismatch"

    def test_verify_signatures_no_key(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")
        # No signing key → no issues (cannot verify)
        assert logger.verify_signatures() == []


class TestAuditQuery:
    def _populate(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "vasp_relaxation")
        logger.log("subprocess_exec", "agent", "compile_code")
        logger.log("tool_call", "user", "lammps_run")
        logger.log("llm_invoke", "agent", "generate_report")
        logger.log("tool_call", "skill", "vasp_analysis")
        return logger

    def test_query_all(self, tmp_path):
        logger = self._populate(tmp_path)
        results = logger.query()
        assert len(results) == 5

    def test_query_by_event_type(self, tmp_path):
        logger = self._populate(tmp_path)
        results = logger.query(event_type="tool_call")
        assert len(results) == 3

    def test_query_by_actor(self, tmp_path):
        logger = self._populate(tmp_path)
        results = logger.query(actor="agent")
        assert len(results) == 2

    def test_query_by_action_substring(self, tmp_path):
        logger = self._populate(tmp_path)
        results = logger.query(action="vasp")
        assert len(results) == 2

    def test_query_limit(self, tmp_path):
        logger = self._populate(tmp_path)
        results = logger.query(limit=2)
        assert len(results) == 2

    def test_query_empty_log(self, tmp_path):
        from huginn.security.audit import AuditLogger
        logger = AuditLogger(tmp_path / "empty.jsonl")
        assert logger.query() == []


class TestAuditChainReplay:
    def test_chain_continuity_after_reload(self, tmp_path):
        from huginn.security.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"

        # Write 3 events
        logger1 = AuditLogger(log_path)
        logger1.log("tool_call", "user", "action1")
        logger1.log("tool_call", "user", "action2")
        logger1.log("tool_call", "user", "action3")

        # Create new logger instance (simulates restart)
        logger2 = AuditLogger(log_path)
        logger2.log("tool_call", "user", "action4")

        # Chain should be intact
        mismatches = logger2.verify_chain()
        assert len(mismatches) == 0


# =========================================================================
#  Audit rotation tests  (audit.py — AuditLogRotator)
# =========================================================================

class TestAuditLogRotator:
    def test_should_rotate_by_size(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        # Write some data
        for i in range(20):
            logger.log("tool_call", "user", f"action_{i}", details={"data": "x" * 100})

        rotator = AuditLogRotator(logger, max_size=100)  # Very small threshold
        assert rotator.should_rotate() is True

    def test_should_not_rotate_small_file(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")

        rotator = AuditLogRotator(logger, max_size=10_000_000)
        assert rotator.should_rotate() is False

    def test_rotate_creates_archive(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")
        logger.log("tool_call", "user", "action2")

        rotator = AuditLogRotator(logger, compress=False)
        archive = rotator.rotate()
        assert archive is not None
        assert archive.exists()
        assert archive.suffix == ".jsonl"
        # Original log should be gone (moved to archive)
        assert not log_path.exists()

    def test_rotate_with_compression(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")

        rotator = AuditLogRotator(logger, compress=True)
        archive = rotator.rotate()
        assert archive is not None
        assert archive.suffix == ".gz"
        assert archive.exists()

    def test_rotate_empty_returns_none(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        logger = AuditLogger(tmp_path / "audit.jsonl")
        rotator = AuditLogRotator(logger)
        assert rotator.rotate() is None

    def test_rotate_if_needed(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        for i in range(10):
            logger.log("tool_call", "user", f"action_{i}", details={"data": "x" * 200})

        rotator = AuditLogRotator(logger, max_size=50)
        archive = rotator.rotate_if_needed()
        assert archive is not None

    def test_prune_archives(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        rotator = AuditLogRotator(
            AuditLogger(log_path),
            compress=False,
            max_archives=2,
        )

        # Create 4 archives
        for i in range(4):
            logger = AuditLogger(log_path)
            logger.log("tool_call", "user", f"action_{i}")
            rotator.logger = logger
            rotator.rotate()
            time.sleep(0.05)  # Ensure different timestamps

        archives = rotator.list_archives()
        assert len(archives) <= 2

    def test_list_archives(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")

        rotator = AuditLogRotator(logger, compress=False)
        rotator.rotate()
        archives = rotator.list_archives()
        assert len(archives) == 1

    def test_new_log_after_rotation(self, tmp_path):
        from huginn.security.audit import AuditLogger, AuditLogRotator
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log("tool_call", "user", "action1")

        rotator = AuditLogRotator(logger, compress=False)
        rotator.rotate()

        # New events should go to a fresh file
        logger.log("tool_call", "user", "action2")
        assert log_path.exists()
        mismatches = logger.verify_chain()
        assert len(mismatches) == 0


# =========================================================================
#  Secret manager tests  (secrets.py)
# =========================================================================

class TestEnvSecretBackend:
    def test_get_set(self):
        from huginn.security.secrets import EnvSecretBackend
        backend = EnvSecretBackend()
        backend.set("TEST_SECRET_P4", "value123")
        assert backend.get("TEST_SECRET_P4") == "value123"
        # Cleanup
        os.environ.pop("TEST_SECRET_P4", None)

    def test_get_nonexistent(self):
        from huginn.security.secrets import EnvSecretBackend
        backend = EnvSecretBackend()
        assert backend.get("NONEXISTENT_KEY_XYZ_P4") is None

    def test_delete(self):
        from huginn.security.secrets import EnvSecretBackend
        backend = EnvSecretBackend()
        backend.set("TEST_DEL_P4", "value")
        assert backend.delete("TEST_DEL_P4") is True
        assert backend.get("TEST_DEL_P4") is None
        assert backend.delete("TEST_DEL_P4") is False

    def test_exists(self):
        from huginn.security.secrets import EnvSecretBackend
        backend = EnvSecretBackend()
        backend.set("TEST_EXISTS_P4", "val")
        assert backend.exists("TEST_EXISTS_P4") is True
        assert backend.exists("NOPE_P4") is False
        os.environ.pop("TEST_EXISTS_P4", None)

    def test_list_keys(self):
        from huginn.security.secrets import EnvSecretBackend
        backend = EnvSecretBackend()
        backend.set("P4_PREFIX_A", "1")
        backend.set("P4_PREFIX_B", "2")
        keys = backend.list_keys(prefix="P4_PREFIX_")
        assert "P4_PREFIX_A" in keys
        assert "P4_PREFIX_B" in keys
        os.environ.pop("P4_PREFIX_A", None)
        os.environ.pop("P4_PREFIX_B", None)

    def test_persist_file(self, tmp_path):
        from huginn.security.secrets import EnvSecretBackend
        persist = str(tmp_path / "secrets.json")
        backend1 = EnvSecretBackend(persist_file=persist)
        backend1.set("P4_PERSIST", "secret_val")

        # New backend should load from file
        os.environ.pop("P4_PERSIST", None)
        backend2 = EnvSecretBackend(persist_file=persist)
        assert backend2.get("P4_PERSIST") == "secret_val"
        os.environ.pop("P4_PERSIST", None)


class TestVaultSecretBackend:
    def test_parse_name(self):
        from huginn.security.secrets import VaultSecretBackend
        assert VaultSecretBackend._parse_name("path/to/secret:key") == ("path/to/secret", "key")
        assert VaultSecretBackend._parse_name("simple/path") == ("simple/path", "value")

    @patch("huginn.security.secrets.VaultSecretBackend._client")
    def test_get(self, mock_client_fn):
        from huginn.security.secrets import VaultSecretBackend
        mock_client = MagicMock()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"api_key": "abc123"}}
        }
        mock_client_fn.return_value = mock_client

        backend = VaultSecretBackend()
        result = backend.get("myapp:api_key")
        assert result == "abc123"
        mock_client.secrets.kv.v2.read_secret_version.assert_called_once_with(
            path="myapp", mount_point="secret"
        )

    @patch("huginn.security.secrets.VaultSecretBackend._client")
    def test_set_new(self, mock_client_fn):
        from huginn.security.secrets import VaultSecretBackend
        mock_client = MagicMock()
        # Simulate path not found on read
        mock_client.secrets.kv.v2.read_secret_version.side_effect = Exception("not found")
        mock_client_fn.return_value = mock_client

        backend = VaultSecretBackend()
        backend.set("newpath:mykey", "newval")
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path="newpath",
            secret={"mykey": "newval"},
            mount_point="secret",
        )

    @patch("huginn.security.secrets.VaultSecretBackend._client")
    def test_set_merge(self, mock_client_fn):
        from huginn.security.secrets import VaultSecretBackend
        mock_client = MagicMock()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"existing_key": "old_val"}}
        }
        mock_client_fn.return_value = mock_client

        backend = VaultSecretBackend()
        backend.set("myapp:new_key", "new_val")
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path="myapp",
            secret={"existing_key": "old_val", "new_key": "new_val"},
            mount_point="secret",
        )

    @patch("huginn.security.secrets.VaultSecretBackend._client")
    def test_delete(self, mock_client_fn):
        from huginn.security.secrets import VaultSecretBackend
        mock_client = MagicMock()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"key1": "val1", "key2": "val2"}}
        }
        mock_client_fn.return_value = mock_client

        backend = VaultSecretBackend()
        result = backend.delete("myapp:key1")
        assert result is True
        # Should write back remaining keys
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path="myapp",
            secret={"key2": "val2"},
            mount_point="secret",
        )

    @patch("huginn.security.secrets.VaultSecretBackend._client")
    def test_delete_last_key(self, mock_client_fn):
        from huginn.security.secrets import VaultSecretBackend
        mock_client = MagicMock()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"only_key": "val"}}
        }
        mock_client_fn.return_value = mock_client

        backend = VaultSecretBackend()
        result = backend.delete("myapp:only_key")
        assert result is True
        mock_client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once()


class TestSecretRotationHook:
    def test_register_and_rotate(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        backend.set("ROTATE_TEST_P4", "old_value")

        hooks = SecretRotationHook(backend)
        hooks.register("ROTATE_TEST_P4", lambda old: "new_value")
        new_val = hooks.rotate("ROTATE_TEST_P4")
        assert new_val == "new_value"
        assert backend.get("ROTATE_TEST_P4") == "new_value"
        os.environ.pop("ROTATE_TEST_P4", None)

    def test_rotate_all(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        backend.set("ROT_A_P4", "old_a")
        backend.set("ROT_B_P4", "old_b")

        hooks = SecretRotationHook(backend)
        hooks.register("ROT_A_P4", lambda old: "new_a")
        hooks.register("ROT_B_P4", lambda old: "new_b")
        results = hooks.rotate_all()
        assert results == {"ROT_A_P4": True, "ROT_B_P4": True}
        assert backend.get("ROT_A_P4") == "new_a"
        assert backend.get("ROT_B_P4") == "new_b"
        os.environ.pop("ROT_A_P4", None)
        os.environ.pop("ROT_B_P4", None)

    def test_rotate_unregistered(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        hooks = SecretRotationHook(backend)
        with pytest.raises(KeyError):
            hooks.rotate("NOPE")

    def test_needs_rotation(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        backend.set("NEEDS_ROT_P4", "val")
        hooks = SecretRotationHook(backend)
        # Never rotated → needs rotation
        assert hooks.needs_rotation("NEEDS_ROT_P4") is True

        hooks.register("NEEDS_ROT_P4", lambda old: "new")
        hooks.rotate("NEEDS_ROT_P4")
        # Just rotated → no rotation needed
        assert hooks.needs_rotation("NEEDS_ROT_P4", max_age_seconds=3600) is False
        os.environ.pop("NEEDS_ROT_P4", None)

    def test_last_rotated(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        hooks = SecretRotationHook(backend)
        assert hooks.last_rotated("NOPE") is None

        hooks.register("LAST_ROT_P4", lambda old: "new")
        backend.set("LAST_ROT_P4", "old")
        hooks.rotate("LAST_ROT_P4")
        assert hooks.last_rotated("LAST_ROT_P4") is not None
        os.environ.pop("LAST_ROT_P4", None)

    def test_registered_names(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        hooks = SecretRotationHook(backend)
        hooks.register("B_KEY", lambda old: "new")
        hooks.register("A_KEY", lambda old: "new")
        assert hooks.registered_names == ["A_KEY", "B_KEY"]

    def test_unregister(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        hooks = SecretRotationHook(backend)
        hooks.register("UNREG_P4", lambda old: "new")
        hooks.unregister("UNREG_P4")
        assert "UNREG_P4" not in hooks.registered_names

    def test_rotate_all_with_failure(self):
        from huginn.security.secrets import EnvSecretBackend, SecretRotationHook
        backend = EnvSecretBackend()
        hooks = SecretRotationHook(backend)
        hooks.register("FAIL_P4", lambda old: 1 / 0)  # Will raise
        results = hooks.rotate_all()
        assert results["FAIL_P4"] is False


class TestBackendRegistry:
    def test_get_env_backend(self):
        from huginn.security.secrets import EnvSecretBackend, get_secret_backend
        backend = get_secret_backend("env")
        assert isinstance(backend, EnvSecretBackend)

    def test_get_vault_backend(self):
        from huginn.security.secrets import VaultSecretBackend, get_secret_backend
        backend = get_secret_backend("vault")
        assert isinstance(backend, VaultSecretBackend)

    def test_get_aws_backend(self):
        from huginn.security.secrets import AWSSecretsManagerBackend, get_secret_backend
        backend = get_secret_backend("aws")
        assert isinstance(backend, AWSSecretsManagerBackend)

    def test_unknown_backend(self):
        from huginn.security.secrets import get_secret_backend
        with pytest.raises(ValueError, match="Unknown"):
            get_secret_backend("nonexistent")

    def test_register_custom_backend(self):
        from huginn.security.secrets import (
            SecretBackend,
            get_secret_backend,
            register_backend,
        )

        class CustomBackend(SecretBackend):
            def get(self, name):
                return "custom"

            def set(self, name, value):
                pass

        register_backend("custom", CustomBackend)
        backend = get_secret_backend("custom")
        assert isinstance(backend, CustomBackend)

    def test_default_from_env(self):
        from huginn.security.secrets import EnvSecretBackend, get_secret_backend
        os.environ["HUGINN_SECRET_BACKEND"] = "env"
        backend = get_secret_backend()
        assert isinstance(backend, EnvSecretBackend)
        os.environ.pop("HUGINN_SECRET_BACKEND", None)


# =========================================================================
#  Auth integration tests  (auth.py)
# =========================================================================

class TestAuthIntegration:
    def test_create_token(self):
        from huginn.security.auth import create_token
        from huginn.security.rbac import Role, User
        os.environ["HUGINN_JWT_SECRET"] = "test-jwt-secret-p4"
        try:
            user = User(user_id="u1", username="alice", role=Role.ADMIN)
            token = create_token(user, expires_in=3600)
            assert isinstance(token, str)
            assert token.count(".") == 2
        finally:
            os.environ.pop("HUGINN_JWT_SECRET", None)

    def test_create_token_no_secret(self):
        from huginn.security.auth import create_token
        from huginn.security.rbac import Role, User
        # Ensure no secrets are set
        os.environ.pop("HUGINN_JWT_SECRET", None)
        os.environ.pop("HUGINN_API_KEY", None)
        user = User(user_id="u1", username="alice", role=Role.ADMIN)
        with pytest.raises(RuntimeError, match="No JWT secret"):
            create_token(user)

    def test_request_context(self):
        from huginn.security.auth import RequestContext
        from huginn.security.rbac import Role, User
        user = User(user_id="u1", username="test", role=Role.OPERATOR)
        ctx = RequestContext(user=user, token="jwt-token", auth_mode="jwt")
        assert ctx.user is user
        assert ctx.auth_mode == "jwt"
        assert ctx.token == "jwt-token"
