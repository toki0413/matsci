"""行为/策略制品生命周期 — 自包含 + 版本握手 + 健康门控回滚.

对应 microduck ``updaterd`` 的核心模式: "releases are swapped, not patched".
接真实 ``executor_backend`` 时, 每个物理策略后端应以**自包含制品**交付
(归一化 / 配置 / 契约 / 参数全部烘焙进制品), 而不是散落在运行时各处的参数.
本模块负责制品的:

- **诞生**   : ``BehaviorArtifact`` 自包含 (config + data + contract baked), 含指纹.
- **握手**   : 全局 ``ARTIFACT_CONTRACT_VERSION`` 整数, 版本不符拒绝 (microduck ``model_api``).
- **安装**   : 整体换目录 ``releases/<version>/``, ``current`` 指针原子切换 (temp+rename).
- **健康门控**: 安装后跑 ``health_check(current)``, 不健康自动回滚到前一版本.
- **崩溃兜底**: boot counter 安装时自增, 成功自愈清零 (microduck boot counter, 防
  "启动即崩但健康检查没抓到").

本模块纯 stdlib, 是独立的基础设施 — 照 microduck "updaterd 最先建" 的原则,
先于真实后端存在. 由 workspace / catalog 在接 executor 时消费. 不依赖 executor.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 制品契约版本 — SDK/daemon 版本会漂移, 用一个整数做握手, 不符即拒绝 (microduck model_api).
ARTIFACT_CONTRACT_VERSION = 1


@dataclass
class BehaviorArtifact:
    """一个自包含的行为 / 策略制品. 归一化与配置烘焙进 ``config``/``data``,
    部署侧不依赖外部步骤补上下文 — 是 microduck "ONNX 归一化烘焙进图" 的软件版.
    """

    name: str
    version: int
    contract_version: int = ARTIFACT_CONTRACT_VERSION
    # self-contained: 部署期所需的全部上下文 (归一化参数 / 超参 / 策略 broker 等).
    config: dict[str, Any] = field(default_factory=dict)
    # opaque payload: 如 onnx 路径 / 工具后端标识 / 协议 spec. 不解释, 只携带.
    data: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "v": self.version,
                "c": self.contract_version,
                "cfg": self.config,
                "data": self.data,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def contract_ok(self, expected: int = ARTIFACT_CONTRACT_VERSION) -> bool:
        return self.contract_version == expected


@dataclass
class InstallResult:
    installed_version: int
    rolled_back_to: int | None
    healthy: bool
    reason: str = ""


class BehaviorLifecycle:
    """On-disk 制品生命周期: ``releases/<ver>/`` + ``current`` 指针 + 健康门控回滚."""

    def __init__(self, state_root: str | Path) -> None:
        self.root = Path(state_root)
        self.releases = self.root / "releases"
        self.current_link = self.root / "current"
        self.boot_counter = self.root / "boot_counter"
        self.root.mkdir(parents=True, exist_ok=True)
        self.releases.mkdir(parents=True, exist_ok=True)

    # ── 查询 ───────────────────────────────────────────────────
    def current_version(self) -> int | None:
        if self.current_link.is_symlink():
            try:
                return int(self.current_link.resolve().name)
            except (ValueError, OSError):
                return None
        return None

    def boot_count(self) -> int:
        try:
            return int(self.boot_counter.read_text().strip() or "0")
        except (OSError, ValueError):
            return 0

    # ── 安装 / 健康门控 / 回滚 ──────────────────────────────────
    def install(
        self,
        artifact: BehaviorArtifact,
        health_check: Callable[[int], bool] | None = None,
    ) -> InstallResult:
        if not artifact.contract_ok():
            return InstallResult(
                artifact.version, None, False, "contract-version-mismatch"
            )
        prior = self.current_version()
        self._write_release(artifact)
        self._swap(artifact.version)
        self._bump_boot()
        healthy = health_check(artifact.version) if health_check else True
        if healthy:
            self._reset_boot()
            return InstallResult(artifact.version, None, True)
        rollback = self._rollback_to(prior)
        return InstallResult(
            artifact.version,
            rollback,
            False,
            f"health-gate-failed, rolled back to {rollback}",
        )

    # ── 内部 ───────────────────────────────────────────────────
    def _write_release(self, artifact: BehaviorArtifact) -> Path:
        d = self.releases / str(artifact.version)
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "name": artifact.name,
                    "version": artifact.version,
                    "contract_version": artifact.contract_version,
                    "config": artifact.config,
                    "data": artifact.data,
                    "fingerprint": artifact.fingerprint(),
                },
                indent=2,
            )
        )
        return d

    def _swap(self, version: int) -> None:
        tgt = self.releases / str(version)
        tmp = self.root / f"current.tmp.{version}"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink(missing_ok=True)
        tmp.symlink_to(tgt.name)
        os.replace(tmp, self.current_link)  # 原子: 整体指针切换, 不就地打补丁.

    def _rollback_to(self, version: int | None) -> int | None:
        if version is None:
            return None
        self._swap(version)
        return version

    def _bump_boot(self) -> int:
        n = self.boot_count() + 1
        self.boot_counter.write_text(str(n))
        return n

    def _reset_boot(self) -> None:
        self.boot_counter.write_text("0")
