"""Persistent credential store for SSH connections and LLM API keys.

Secrets (SSH passwords, LLM API keys) are encrypted at rest with Fernet
(AES-128-CBC + HMAC-SHA256). The encryption key is either unlocked by the
Huginn master password (``HUGINN_ENCRYPTION_PASSWORD``) or, for local
single-user desktop setups that run without a master password, auto-generated
and stored in ``~/.huginn/cred.key``. Either way the secret never lands on
disk in plaintext.

The store backs onto a small SQLite database (``~/.huginn/credentials.sqlite``
by default) and supports multiple named entries per kind, so a user can keep
several HPC clusters and several LLM providers side by side and pick a
default per kind.

借鉴 AstrBot 的三层 CRUD 思路: 数据层 (本模块) 只管加密存储 +
脱敏读取, 路由层负责 HTTP, 前端负责可视化. 与 AstrBot 不同的是, 这里
的凭据必须可逆解密 (要拿明文去连 SSH / 调 LLM), 所以用 Fernet 而不是
PBKDF2 哈希.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from huginn.crypto import KeyManager

logger = logging.getLogger(__name__)

# 凭据类型常量 — 用字符串而不是枚举, 方便 JSON 序列化与跨进程传递
CRED_KIND_SSH = "ssh"
CRED_KIND_LLM = "llm"
_VALID_KINDS = (CRED_KIND_SSH, CRED_KIND_LLM)

# 模块级单例 — get_credential_store() 懒加载, 测试可直接构造 CredentialStore
_store_lock = threading.Lock()
_store_singleton: CredentialStore | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    # 8 字节 hex = 16 字符, 对凭据条目数量绰绰有余且 URL 安全
    return secrets.token_hex(8)


def mask_secret(raw: str | None) -> str | None:
    """脱敏: 前4 + **** + 后4; 不足8位全掩; None 透传。

    和 routes/config.py 的 _mask_api_key 保持一致的展示风格, 让前端两处
    密钥展示视觉统一。
    """
    if raw is None:
        return None
    if not raw:
        return ""
    if len(raw) < 8:
        return "********"
    return f"{raw[:4]}****{raw[-4:]}"


def _cred_dir() -> Path:
    """凭据相关文件的落地目录。

    优先 HUGINN_CACHE_DIR (与项目其它模块的隔离约定一致, 测试也靠它
    把写入重定向到临时目录), 否则退回 ~/.huginn。
    """
    cache = os.environ.get("HUGINN_CACHE_DIR")
    if cache:
        return Path(cache)
    return Path.home() / ".huginn"


def _get_fernet() -> Fernet:
    """Return a Fernet instance for encrypting credentials.

    优先级:
    1. ``HUGINN_ENCRYPTION_PASSWORD`` + 加密密钥文件 (主密码保护, 与配置
       加密共用同一主密码, 用户只需记一个密码)
    2. 本地自动生成的密钥文件 ``<cred_dir>/cred.key`` (桌面端免主密码场景;
       文件本身明文, 靠文件系统权限保护, 换机需迁移)

    直接用 Fernet 而不走 CryptoVault, 因为 CryptoVault 每次 encrypt 都做
    480k 次 PBKDF2 迭代, 对凭据这种低频但可能批量读取的场景太慢; Fernet
    key 本身已是 256-bit 高熵随机量, 不需要再派生。
    """
    key_file = Path(
        os.environ.get("HUGINN_CREDENTIAL_KEY_FILE")
        or (_cred_dir() / "cred.key")
    )
    key_file.parent.mkdir(parents=True, exist_ok=True)

    password = os.environ.get("HUGINN_ENCRYPTION_PASSWORD")
    if password:
        km = KeyManager(key_file)
        if key_file.exists():
            master_key = km.load_key_file(password)
        else:
            master_key = km.create_key_file(password)
        return Fernet(master_key)

    # 无主密码 — 落地一个随机 Fernet key 文件, 靠文件权限兜底
    if not key_file.exists():
        key_file.write_bytes(Fernet.generate_key())
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            # Windows 上 chmod 语义不同, 忽略即可; 文件仍在用户目录下
            pass
    return Fernet(key_file.read_bytes())


@dataclass
class CredentialRecord:
    """一条凭据的完整记录 (含解密后的明文 secret)。

    仅供内部 / 受信任的调用方使用; 对外返回时必须走 to_masked_dict()。
    """

    id: str
    kind: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    secret: str = ""  # 解密后的明文, 切勿直接序列化给前端
    is_default: bool = False
    created_at: str = ""
    updated_at: str = ""

    def to_masked_dict(self) -> dict[str, Any]:
        """对外展示用: secret 脱敏, 附 has_secret 方便前端显示状态灯。"""
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "metadata": dict(self.metadata),
            "secret_masked": mask_secret(self.secret) if self.secret else None,
            "has_secret": bool(self.secret),
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CredentialStore:
    """加密凭据的持久化存储, 支持 SSH 与 LLM 两类、多套、设默认。

    线程安全: 每次操作新建 sqlite3 连接 (with 块自动关闭), Fernet 实例
    本身线程安全, 因此 CredentialStore 对象可跨线程共享。
    """

    def __init__(self, db_path: str | Path, fernet: Fernet | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # fernet 可为 None (测试时不想真加密可注入假实现); 默认按规则取
        self._fernet = fernet or _get_fernet()
        self._ensure_schema()

    # ── 内部: 数据库连接与 schema ───────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False: FastAPI 线程池里可能跨线程复用本对象,
        # 但每次 with 块内连接是独立的, 不会真跨线程用同一连接
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS credentials (
                    id            TEXT PRIMARY KEY,
                    kind          TEXT NOT NULL,
                    name          TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    secret_enc    TEXT NOT NULL DEFAULT '',
                    is_default    INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            # 按种类 + 名称查重用得上, 走个普通索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_credentials_kind "
                "ON credentials(kind)"
            )
            conn.commit()

    def _encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def _decrypt(self, token: str) -> str:
        if not token:
            return ""
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            # 密钥换了 / 文件损坏 — 不要把密文抛出去, 给个安全提示
            raise RuntimeError(
                "无法解密凭据: 加密密钥可能已变更。若重置过主密码或迁移过"
                "cred.key, 旧凭据需重新录入。"
            ) from exc

    def _row_to_record(self, row: sqlite3.Row) -> CredentialRecord:
        return CredentialRecord(
            id=row["id"],
            kind=row["kind"],
            name=row["name"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            secret=self._decrypt(row["secret_enc"]),
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── 公开 API: CRUD ───────────────────────────────────────────

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """列出凭据 (脱敏)。kind 为 None 时返回全部。"""
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM credentials WHERE kind=? ORDER BY "
                    "is_default DESC, created_at ASC",
                    (kind,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM credentials ORDER BY "
                    "is_default DESC, created_at ASC"
                ).fetchall()
        return [self._row_to_record(r).to_masked_dict() for r in rows]

    def get(self, cid: str) -> dict[str, Any] | None:
        """单条脱敏详情。不存在返回 None。"""
        rec = self.get_record(cid)
        return rec.to_masked_dict() if rec else None

    def get_record(self, cid: str) -> CredentialRecord | None:
        """单条完整记录 (含明文 secret)。仅限内部受信调用方。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM credentials WHERE id=?", (cid,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_secret(self, cid: str) -> str:
        """直接取明文 secret。供 HPC / LLM 客户端集成用。"""
        rec = self.get_record(cid)
        return rec.secret if rec else ""

    def create(
        self,
        kind: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        secret: str = "",
        is_default: bool | None = None,
    ) -> dict[str, Any]:
        """新建一条凭据。

        is_default=None 时: 若该 kind 当前没有任何条目, 自动设为默认;
        否则不设默认。显式传 True 则强制设默认 (并取消同 kind 其他默认)。
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind 必须是 {_VALID_KINDS} 之一, 得到 {kind!r}")
        if not name or not name.strip():
            raise ValueError("name 不能为空")

        cid = _new_id()
        now = _now_iso()
        metadata = metadata or {}

        # 决定是否设默认
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM credentials WHERE kind=?", (kind,)
            ).fetchone()[0]
            should_be_default = is_default if is_default is not None else (count == 0)

            if should_be_default:
                conn.execute(
                    "UPDATE credentials SET is_default=0 WHERE kind=?", (kind,)
                )

            conn.execute(
                "INSERT INTO credentials "
                "(id, kind, name, metadata_json, secret_enc, is_default, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    cid,
                    kind,
                    name.strip(),
                    json.dumps(metadata, ensure_ascii=False),
                    self._encrypt(secret),
                    1 if should_be_default else 0,
                    now,
                    now,
                ),
            )
            conn.commit()

        return self.get(cid)  # type: ignore[return-value]

    def update(
        self,
        cid: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        secret: str | None = None,
    ) -> dict[str, Any] | None:
        """部分更新。只更新传入的字段; secret 传 None 表示不改密钥。

        secret="" (空串) 表示清空密钥 — 与 None 语义区分开。
        """
        rec = self.get_record(cid)
        if rec is None:
            return None

        new_name = name.strip() if name is not None else rec.name
        if not new_name:
            raise ValueError("name 不能为空")
        new_metadata = metadata if metadata is not None else rec.metadata
        # None=不改; 传了值(含空串)就覆盖
        new_secret = secret if secret is not None else rec.secret
        now = _now_iso()

        with self._connect() as conn:
            conn.execute(
                "UPDATE credentials SET name=?, metadata_json=?, secret_enc=?, "
                "updated_at=? WHERE id=?",
                (
                    new_name,
                    json.dumps(new_metadata, ensure_ascii=False),
                    self._encrypt(new_secret),
                    now,
                    cid,
                ),
            )
            conn.commit()
        return self.get(cid)

    def delete(self, cid: str) -> bool:
        """删除一条凭据。若删的是默认, 同 kind 自动提升最早一条为默认。"""
        rec = self.get_record(cid)
        if rec is None:
            return False

        with self._connect() as conn:
            conn.execute("DELETE FROM credentials WHERE id=?", (cid,))
            # 删了默认就补一个, 避免该 kind 出现"无默认"的尴尬状态
            if rec.is_default:
                survivor = conn.execute(
                    "SELECT id FROM credentials WHERE kind=? "
                    "ORDER BY created_at ASC LIMIT 1",
                    (rec.kind,),
                ).fetchone()
                if survivor:
                    conn.execute(
                        "UPDATE credentials SET is_default=1 WHERE id=?",
                        (survivor["id"],),
                    )
            conn.commit()
        return True

    def set_default(self, cid: str) -> bool:
        """把指定凭据设为同 kind 的默认, 取消其他默认。"""
        rec = self.get_record(cid)
        if rec is None:
            return False
        with self._connect() as conn:
            conn.execute(
                "UPDATE credentials SET is_default=0 WHERE kind=?", (rec.kind,)
            )
            conn.execute(
                "UPDATE credentials SET is_default=1, updated_at=? WHERE id=?",
                (_now_iso(), cid),
            )
            conn.commit()
        return True

    def get_default(self, kind: str) -> dict[str, Any] | None:
        """取某 kind 的默认凭据 (脱敏)。没有默认返回 None。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM credentials WHERE kind=? AND is_default=1 LIMIT 1",
                (kind,),
            ).fetchone()
        return self._row_to_record(row).to_masked_dict() if row else None

    def get_default_record(self, kind: str) -> CredentialRecord | None:
        """取某 kind 的默认凭据完整记录 (含明文)。内部用。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM credentials WHERE kind=? AND is_default=1 LIMIT 1",
                (kind,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    # ── 适配层: 把凭据转成现有系统的配置对象 ─────────────────────

    def to_hpc_config(self, cid: str):
        """从一条 SSH 凭据构造 HPCConfig (含解密后的 password)。

        供 /hpc/* 端点和远程执行器调用, 这样前端提交任务时只需传
        credential_id, 不用每次带 host/password。
        """
        rec = self.get_record(cid)
        if rec is None or rec.kind != CRED_KIND_SSH:
            return None
        # lazy import 避免与 hpc.client 的潜在循环导入
        from huginn.hpc.client import HPCConfig

        m = rec.metadata
        return HPCConfig(
            host=m.get("host", ""),
            username=m.get("username", ""),
            scheduler=m.get("scheduler", "slurm"),
            key_path=m.get("key_path") or None,
            password=rec.secret or None,
            port=int(m.get("port", 22)),
            remote_work_dir=m.get("remote_work_dir", "~/huginn_jobs"),
            default_queue=m.get("default_queue") or None,
            gpu_queue=m.get("gpu_queue") or None,
            strict_host_key_checking=bool(
                m.get("strict_host_key_checking", True)
            ),
            known_hosts_path=m.get("known_hosts_path") or None,
        )

    def to_llm_info(self, cid: str) -> dict[str, Any] | None:
        """从一条 LLM 凭据取出去建模型用的字段 (含明文 api_key)。

        返回 dict 而不是 ModelConfig, 是为了不绑死数据结构, 调用方
        (models/registry 或 agent factory) 按需取字段。
        """
        rec = self.get_record(cid)
        if rec is None or rec.kind != CRED_KIND_LLM:
            return None
        m = rec.metadata
        return {
            "alias": m.get("alias", rec.name),
            "provider": m.get("provider", ""),
            "model": m.get("model", ""),
            "base_url": m.get("base_url") or None,
            "api_key": rec.secret,
            "temperature": m.get("temperature"),
            "max_tokens": m.get("max_tokens"),
            "thinking": m.get("thinking"),
            "enabled": m.get("enabled", True),
        }

    def import_from_config(self, config) -> dict[str, str]:
        """Batch-import plain-text API keys from a HuginnConfig into the store.

        Walks ``config.models`` and creates an LLM credential for every entry
        whose ``api_key`` is a literal key (not an ``env:`` / ``keyring:``
        reference). Entries that already have a credential with the same name
        are skipped. Returns ``{alias: credential_id}`` for everything that was
        actually imported.
        """
        imported: dict[str, str] = {}
        existing = {c["name"] for c in self.list(CRED_KIND_LLM)}

        for m in config.models:
            key = m.api_key
            if not key or key.startswith("env:") or key.startswith("keyring:"):
                continue
            if m.alias in existing:
                logger.info("skip importing '%s' — credential already exists", m.alias)
                continue

            rec = self.create(
                kind=CRED_KIND_LLM,
                name=m.alias,
                metadata={
                    "alias": m.alias,
                    "provider": m.provider,
                    "model": m.model or "",
                    "base_url": m.base_url or "",
                    "temperature": m.temperature,
                },
                secret=key,
            )
            imported[m.alias] = rec["id"]
            logger.info("imported LLM credential '%s' (provider=%s)", m.alias, m.provider)

        return imported


def get_credential_store() -> CredentialStore:
    """返回全局 CredentialStore 单例 (线程安全懒加载)。

    路由层和集成层都走这个入口, 保证全进程共用同一份 DB 与 Fernet。
    测试不要用这个 — 直接 CredentialStore(tmp_path, fernet) 隔离。
    """
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                db_path = Path(
                    os.environ.get("HUGINN_CREDENTIAL_DB")
                    or (_cred_dir() / "credentials.sqlite")
                )
                _store_singleton = CredentialStore(db_path)
    return _store_singleton


def reset_credential_store() -> None:
    """清除单例。主要给测试用, 生产代码别调。"""
    global _store_singleton
    with _store_lock:
        _store_singleton = None
