"""Tests for HPC SSH host key verification."""

from __future__ import annotations

import pytest

from huginn.hpc.client import HPCClient, HPCConfig

paramiko = pytest.importorskip("paramiko")


def test_strict_mode_uses_reject_policy(monkeypatch):
    """Strict host-key checking should set RejectPolicy and load host keys."""
    cfg = HPCConfig(
        host="hpc.example.com",
        username="user",
        strict_host_key_checking=True,
        known_hosts_path="/fake/known_hosts",
    )
    client = HPCClient(cfg)

    calls = {"policy": None, "loaded": []}

    class FakeTransport:
        active = True

        def get_security_options(self):
            return self

    class FakeSSH:
        def __init__(self):
            self._host_keys = {"hpc.example.com": {}}

        def set_missing_host_key_policy(self, policy):
            calls["policy"] = policy

        def load_host_keys(self, path):
            calls["loaded"].append(path)

        def load_system_host_keys(self):
            calls["loaded"].append("__system__")

        def get_host_keys(self):
            return self._host_keys

        def connect(self, **kwargs):
            calls["connect"] = kwargs

        def open_sftp(self):
            return object()

    fake = FakeSSH()
    monkeypatch.setattr(paramiko, "SSHClient", lambda: fake)

    client.connect()

    assert isinstance(calls["policy"], paramiko.RejectPolicy)
    assert "/fake/known_hosts" in calls["loaded"]
    assert client._ssh is fake


def test_strict_mode_without_known_hosts_raises(monkeypatch):
    """If no host keys are available, connect must refuse in strict mode."""
    cfg = HPCConfig(
        host="hpc.example.com",
        username="user",
        strict_host_key_checking=True,
        known_hosts_path=None,
    )
    client = HPCClient(cfg)

    class FakeSSH:
        _host_keys = {}

        def set_missing_host_key_policy(self, policy):
            pass

        def load_system_host_keys(self):
            pass

        def get_host_keys(self):
            return self._host_keys

    monkeypatch.setattr(paramiko, "SSHClient", lambda: FakeSSH())

    with pytest.raises(paramiko.SSHException):
        client.connect()


def test_non_strict_mode_uses_auto_add_policy(monkeypatch):
    """When strict host-key checking is disabled, AutoAddPolicy is used."""
    cfg = HPCConfig(
        host="hpc.example.com",
        username="user",
        strict_host_key_checking=False,
    )
    client = HPCClient(cfg)

    calls = {"policy": None}

    class FakeSSH:
        _host_keys = {}

        def set_missing_host_key_policy(self, policy):
            calls["policy"] = policy

        def connect(self, **kwargs):
            pass

        def open_sftp(self):
            return object()

    monkeypatch.setattr(paramiko, "SSHClient", lambda: FakeSSH())

    client.connect()

    assert isinstance(calls["policy"], paramiko.AutoAddPolicy)
