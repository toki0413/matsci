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
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from huginn.crypto import KeyManager
from huginn.utils.common import now_iso

logger = logging.getLogger(__name__)

# 凭据类型常量 — 用字符串而不是枚举, 方便 JSON 序列化与跨进程传递
CRED_KIND_SSH = "ssh"
CRED_KIND_LLM = "llm"
_VALID_KINDS = (CRED_KIND_SSH, CRED_KIND_LLM)

# 模块级单例 — get_credential_store() 懒加载, 测试可直接构造 CredentialStore
_store_lock = threading.Lock()
_store_singleton: CredentialStore | None = None


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
        # WAL: 每个新连接都设上, 避免并发写时 "database is locked"
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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
        now = now_iso()
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
        now = now_iso()

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
                (now_iso(), cid),
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


# ════════════════════════════════════════════════════════════════════
# 外部服务 API Key 凭据存储 (service-keyed, 加密 JSON 落盘)
# ════════════════════════════════════════════════════════════════════
#
# 上面那个 CredentialStore 面向 SSH 连接 + LLM 推理 key: SQLite、以 cid
# 为主键、同 kind 可多套。下面这套 ServiceCredentialStore 面向"外部数据源 /
# 出版商 / LLM provider"的 API key: 一份 service 一份 key, 走加密 JSON,
# 主键就是 service 名。两套并行, 互不干扰 — 上面的 SSH/LLM CRUD 照常用,
# 这套给前端"选服务 → 填 key → 一键测试"用。
#
# 设计要点:
# - api_key 用 Fernet 加密后写进 JSON, metadata 明文 (非敏感);
# - 全程 RLock, load → 改 → save 原子化, 避免并发写丢更新;
# - 明文 key 只能经 get_credential() 拿到, list_services() 永不回密文/明文;
# - cryptography 惰性导入, 缺包时给清晰报错而不是模块加载期炸。

# 前端下拉用的预定义服务清单 — 不在这里面的服务名会被路由层拒掉,
# 避免任意字符串当 service 名写进存储。
SUPPORTED_SERVICES: list[str] = [
    "openai",
    "anthropic",
    "google_ai",
    "deepseek",
    "qwen",
    "materials_project",
    "wiley",
    "scopus",
    "springer_nature",
    "elsevier_science_direct",
    "arxiv",
    "semantic_scholar",
    "nist_webbook",
    "pubchem",
    "chemspider",
]

# 模块级单例锁 + 单例引用 — 与上面的 _store_lock 分开, 两套存储互不干扰
_svc_store_lock = threading.Lock()
_svc_store_singleton: "ServiceCredentialStore | None" = None


def _service_cred_file() -> Path:
    """服务 API key 加密 JSON 的落盘路径。

    优先 HUGINN_CACHE_DIR (与项目其它模块的隔离约定一致, 测试靠它把写入
    重定向到临时目录), 否则退回 ~/.huginn/credentials.enc.json。
    """
    cache = os.environ.get("HUGINN_CACHE_DIR")
    if cache:
        return Path(cache) / "credentials.enc.json"
    return Path.home() / ".huginn" / "credentials.enc.json"


def _master_key_file() -> Path:
    """主密钥文件路径。

    HUGINN_ENCRYPTION_KEY 未设时自动生成的 Fernet key 落到这里。生产环境
    应改用环境变量显式提供, 这个文件只是兜底。
    """
    cache = os.environ.get("HUGINN_CACHE_DIR")
    base = Path(cache) if cache else (Path.home() / ".huginn")
    return base / "master.key"


def _get_service_fernet():
    """构造 ServiceCredentialStore 用的 Fernet 实例。

    优先级:
    1. ``HUGINN_ENCRYPTION_KEY`` 环境变量 — 一个 base64 urlsafe Fernet key
    2. 自动生成并落到 ``master.key`` 文件 (同时打 WARNING, 提醒生产环境
       显式配置; 桌面单机场景靠文件权限兜底)

    惰性导入 cryptography: 缺包时抛 ImportError 而不是在模块加载期就炸,
    这样即使没装 cryptography, 上面的 CredentialStore 仍可正常 import
    (它顶部已硬 import 过, 这里单独可控方便后续按需拆分)。
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - cryptography 是硬依赖
        raise ImportError(
            "加密凭据存储需要 cryptography 库。请执行 "
            "`pip install cryptography` 后重试。"
        ) from exc

    env_key = os.environ.get("HUGINN_ENCRYPTION_KEY")
    if env_key:
        try:
            return Fernet(env_key.encode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "HUGINN_ENCRYPTION_KEY 不是合法的 Fernet key。"
                "请用 Fernet.generate_key() 生成一个 base64 urlsafe key。"
            ) from exc

    # 没设环境变量 — 自动生成并落盘, 同时告警
    key_file = _master_key_file()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if not key_file.exists():
        new_key = Fernet.generate_key()
        key_file.write_bytes(new_key)
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            # Windows 上 chmod 语义不同, 忽略即可; 文件仍在用户目录下
            pass
        logger.warning(
            "HUGINN_ENCRYPTION_KEY 未设置, 已自动生成主密钥并写入 %s。"
            "生产环境请通过环境变量显式提供 HUGINN_ENCRYPTION_KEY, "
            "并妥善备份该密钥文件 — 丢失后已存储的凭据将无法解密。",
            key_file,
        )
    return Fernet(key_file.read_bytes())


class ServiceCredentialStore:
    """以 service 名为主键的加密 API key 存储 (JSON 落盘)。

    一份 service 一份 key, 适合前端做"选服务 → 填 key → 测试"的流程。
    api_key 用 Fernet 加密后写进 JSON, metadata 明文存储 (非敏感参数)。
    全程加锁, load → 改 → save 原子化, 避免并发写丢更新。

    明文 key 只能通过 :meth:`get_credential` 取到;
    :meth:`list_services` 永远不返回密文或明文, 只回 service 名 + metadata +
    has_key 标记, 方便前端展示状态灯。

    与 :class:`CredentialStore` (SQLite / SSH+LLM / cid-keyed) 互补, 不共享
    存储文件, 单例也各自独立。
    """

    def __init__(
        self,
        file_path: str | Path,
        fernet: Any = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # fernet 可注入 (测试用假实现); 默认走 HUGINN_ENCRYPTION_KEY / master.key
        self._fernet = fernet if fernet is not None else _get_service_fernet()
        self._lock = threading.RLock()

    # ── 内部: 文件读写 ─────────────────────────────────────────

    def _load(self) -> dict[str, dict[str, Any]]:
        """读 JSON。文件不存在或损坏时返回空 dict, 不抛 — 让上层逻辑统一处理。"""
        if not self.file_path.exists():
            return {}
        try:
            text = self.file_path.read_text(encoding="utf-8")
            data = json.loads(text) if text.strip() else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "凭据文件 %s 读取失败, 当作空处理: %s", self.file_path, exc
            )
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        """原子写: 先写 .tmp 再 rename, 避免半写状态损坏凭据库。"""
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.file_path)

    def _encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def _decrypt(self, token: str) -> str:
        if not token:
            return ""
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except Exception as exc:
            # 密钥换了 / 文件损坏 — 不要把密文抛出去, 给个安全提示
            raise RuntimeError(
                "无法解密凭据: 主密钥可能已变更。若重置过 HUGINN_ENCRYPTION_KEY "
                "或迁移过 master.key, 旧凭据需重新录入。"
            ) from exc

    # ── 公开 API ───────────────────────────────────────────────

    def set_credential(
        self,
        service: str,
        api_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """新增 / 更新一份 service 凭据。

        metadata 走合并而非整体替换 — 便于增量补字段 (比如先存 key, 后补
        base_url)。updated_at / created_at 自动维护。
        """
        if not service or not service.strip():
            raise ValueError("service 不能为空")
        service = service.strip()
        with self._lock:
            data = self._load()
            entry = data.get(service, {})
            old_meta = entry.get("metadata") if isinstance(entry, dict) else None
            new_meta = dict(old_meta or {})
            if metadata:
                new_meta.update(metadata)
            now = now_iso()
            new_meta["updated_at"] = now
            if "created_at" not in new_meta:
                new_meta["created_at"] = now
            data[service] = {
                "api_key": self._encrypt(api_key or ""),
                "metadata": new_meta,
            }
            self._save(data)
        logger.info("service 凭据已写入: %s", service)

    def get_credential(self, service: str) -> str:
        """取明文 api_key。不存在返回空串 (调用方按需用 has_credential 判)。"""
        with self._lock:
            data = self._load()
            entry = data.get(service)
            if not isinstance(entry, dict):
                return ""
            return self._decrypt(entry.get("api_key", ""))

    def list_services(self) -> list[dict[str, Any]]:
        """列出已配置的 service (脱敏: 只回 service 名 + metadata + has_key)。

        顺序按 service 名排序, 方便前端稳定渲染。永远不回密文或明文 key。
        """
        with self._lock:
            data = self._load()
            result: list[dict[str, Any]] = []
            for name, entry in sorted(data.items()):
                if not isinstance(entry, dict):
                    continue
                meta = entry.get("metadata") or {}
                enc = entry.get("api_key", "")
                result.append(
                    {
                        "service": name,
                        "metadata": dict(meta),
                        "has_key": bool(enc),
                        "updated_at": meta.get("updated_at", ""),
                    }
                )
            return result

    def delete_credential(self, service: str) -> bool:
        """删除一份 service 凭据。不存在返回 False。"""
        with self._lock:
            data = self._load()
            if service not in data:
                return False
            del data[service]
            self._save(data)
        logger.info("service 凭据已删除: %s", service)
        return True

    def has_credential(self, service: str) -> bool:
        with self._lock:
            data = self._load()
            entry = data.get(service)
            return isinstance(entry, dict) and bool(entry.get("api_key"))

    def get_metadata(self, service: str) -> dict[str, Any]:
        """取 service 的 metadata (非敏感)。不存在返回空 dict。"""
        with self._lock:
            data = self._load()
            entry = data.get(service)
            if not isinstance(entry, dict):
                return {}
            return dict(entry.get("metadata") or {})


def get_service_credential_store() -> ServiceCredentialStore:
    """返回全局 ServiceCredentialStore 单例 (线程安全懒加载)。

    路由层和集成层都走这个入口, 保证全进程共用同一份 JSON 与 Fernet。
    测试不要用这个 — 直接 ServiceCredentialStore(tmp_path, fernet) 隔离。
    """
    global _svc_store_singleton
    if _svc_store_singleton is None:
        with _svc_store_lock:
            if _svc_store_singleton is None:
                _svc_store_singleton = ServiceCredentialStore(_service_cred_file())
    return _svc_store_singleton


def reset_service_credential_store() -> None:
    """清除 service 凭据单例。主要给测试用, 生产代码别调。"""
    global _svc_store_singleton
    with _svc_store_lock:
        _svc_store_singleton = None
