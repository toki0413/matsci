"""T-BCSE-13: HPC 提权 / 容器逃逸 pattern 拦截测试.

验证 `huginn.hpc.client._validate_command` 拒绝常见提权/逃逸 pattern,
同时仍接受正常 HPC 命令.

require_digest 强制 (容器端) 已在 test_phase4_security.py 覆盖.
"""

from __future__ import annotations

import pytest

from huginn.hpc.client import _validate_command


def test_accepts_normal_command() -> None:
    _validate_command("vasp_std > vasp.out 2>&1")


def test_blocks_privileged_pattern() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("docker run --privileged image /bin/bash")


def test_blocks_sudo() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("sudo rm -rf /tmp/x")


def test_blocks_cap_add() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("docker run --cap-add=ALL image run.sh")


def test_blocks_user_root() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("docker run --user root image run.sh")


def test_blocks_setuid() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("chmod 4777 /usr/bin/something")


def test_blocks_mount_escape() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("docker run --mount source=/,target=/host image run.sh")


def test_blocks_docker_device_escape() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("docker run --device=/dev/sda image run.sh")


def test_case_insensitive() -> None:
    with pytest.raises(ValueError, match="privilege pattern"):
        _validate_command("DOCKER RUN --PRIVILEGED image /bin/bash")
