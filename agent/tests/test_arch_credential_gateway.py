"""制度化闭环：凭据归口 / 凭据网关门禁。

用户服务凭据（LLM key、MP、MinerU、WeCom、加密 password）允许经 `/config`
前端配置；但 **operator 级密钥**（后端自身鉴权/审计/主密钥/密钥后端基础设施）
必须留在 env / 密钥管理器，禁止通过 `/config` 通道被前端设置。

三条规则:
  1. operator 密钥不得出现在 `_apply_legacy_params_to_env` 映射里(即不可被前端写入);
  2. 用户服务凭据必须显式登记在映射里(防被误删掉线);
  3. 经 `/config` 读回的配置必须对凭据脱敏, 不泄露明文。
"""

from __future__ import annotations

from huginn.config import HuginnConfig
from huginn.routes.config import _apply_legacy_params_to_env

# 前端可配置的用户服务凭据 (都应在 /config 映射里)
USER_SERVICE_CREDENTIALS = {
    "api_key",           # LLM provider key (desktop 里同时兼作后端 HUGINN_API_KEY)
    "mp_api_key",        # Materials Project
    "mineru_api_keys",   # MinerU 文献解析
    "wecom_token",       # 企微机器人
    "encryption_password",  # 配置加密口令
}

# operator 级密钥: 只能 env/密钥管理器配, 严禁经 /config 前端写入
OPERATOR_ONLY_ENV = {
    "HUGINN_ADMIN_API_KEY",   # 后端管理鉴权
    "HUGINN_JWT_SECRET",      # JWT 签名密钥
    "HUGINN_AUDIT_SIGNING_KEY",  # 审计日志签名
    "HUGINN_ENCRYPTION_KEY",  # raw Fernet 主密钥 (区别于 HUGINN_ENCRYPTION_PASSWORD)
    "HUGINN_VAULT_ADDR",      # Vault 基础设施
    "HUGINN_VAULT_TOKEN",     # Vault 凭据
    # AWS_* 由 secrets.py 直接读, 也不应经 /config 写入
}


def _mapping() -> dict[str, str]:
    import inspect
    src = inspect.getsource(_apply_legacy_params_to_env)
    # 提取 mapping 字典字面量里的 key -> env 对 (简单可靠)
    mapping: dict[str, str] = {}
    for line in src.splitlines():
        line = line.strip()
        if '": "' in line and ":" in line:
            key, _, rest = line.partition(":")
            env = rest.strip().rstrip(",").strip('"')
            mapping[key.strip().strip('"')] = env
    return mapping


def test_operator_secrets_not_frontend_configurable():
    mapping = _mapping()
    leak = [env for env in OPERATOR_ONLY_ENV if env in mapping.values()]
    assert not leak, (
        "operator 级密钥不得经 /config 前端写入, 否则前端可改后端自身鉴权/审计密钥。"
        f" 泄漏: {leak}"
    )


def test_user_service_credentials_registered_in_mapping():
    mapping = _mapping()
    missing = USER_SERVICE_CREDENTIALS - set(mapping.keys())
    assert not missing, (
        "用户服务凭据必须显式登记在 /config 映射里(前端可配)。缺失: "
        f"{sorted(missing)}"
    )


def test_config_roundtrip_masks_credentials():
    import os as _os
    saved = {k: _os.environ.get(k) for k in (
        "MP_API_KEY", "MINERU_API_KEYS", "HUGINN_WECOM_TOKEN", "HUGINN_ENCRYPTION_PASSWORD",
    )}
    try:
        _apply_legacy_params_to_env({
            "mp_api_key": "mp-secret-abc",
            "mineru_api_keys": "mineru-secret",
            "wecom_token": "wecom-secret",
            "encryption_password": "enc-pass",
        })
        cfg = HuginnConfig.from_env()
        d = cfg.to_dict(mask_key=True)
        for secret in ("mp-secret-abc", "mineru-secret", "wecom-secret", "enc-pass"):
            assert secret not in str(d), f"凭据 {secret!r} 在 to_dict(mask_key=True) 中泄露明文"
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v