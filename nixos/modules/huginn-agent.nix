# modules/huginn-agent.nix
#
# 把 Huginn FastAPI 后端 (agent/huginn/server.py) 声明式跑成 hardened systemd 服务.
#
#     端口默认 8000 (server.py 的 --port), 监听 127.0.0.1.
#     运行时目录默认 /data/huginn (对应 get_runtime_home 的 $HUGINN_CACHE_DIR),
#     并补齐 deploy 脚本里用的 HUGINN_MEMORY_DIR / WORKSPACE_DIR.
#
# venv 自举约定: 首次启动 (ExecStartPre) 若 ${cfg.agentPath}/.venv 不存在, 用仓库自带
#   deploy_and_run_long_research.sh 同款命令建 venv + 装依赖, 幂等 (marker 文件短路).
#   systemd 单元本体是纯声明式 + hardened, 不持有密钥.

{ config, lib, pkgs, ... }:
let
  cfg = config.services.huginn-agent;
in
{
  options.services.huginn-agent = {
    enable = lib.mkEnableOption "the Huginn FastAPI backend service";

    agentPath = lib.mkOption {
      type = lib.types.path;
      default = "/opt/matsci/agent";
      description = "Agent 代码根目录 (含 pyproject.toml 与 huginn 包). 需预先 git clone.";
    };

    runtimeDir = lib.mkOption {
      type = lib.types.path;
      default = "/data/huginn";
      description = "runtime home (HUGINN_CACHE_DIR). 缺省生成, 持久化不被 /tmp 清理.";
    };

    workspaceDir = lib.mkOption {
      type = lib.types.path;
      default = "/data/huginn/workspace";
      description = "WORKSPACE_DIR, 长闭环/工具的执行落盘目录.";
    };

    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "绑定地址. 默认回环; 不要直接暴露公网.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "HTTP/WS 端口 (server.py --port).";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "额外注入的环境变量 (HUGINN_ENV, HUGINN_DEV_MODE, HUGINN_* 开关等).";
    };

    envFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "systemd EnvironmentFile (只放密钥, 如 DEEPSEEK_API_KEY). 权限建议 0600.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "huginn";
      description = "运行身份. 默认自动断言创建同名系统用户.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "huginn";
      description = "运行主组.";
    };
  };

  config = lib.mkIf cfg.enable {
    # 专用低权用户, 不 root
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.runtimeDir;
      createHome = true;
      description = "Huginn agent service user";
    };
    users.groups.${cfg.group} = { };

    systemd.services.huginn-agent = {
      description = "Huginn scientific autonomous agent backend";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      path = with pkgs; [ python311 git gcc ];

      # venv 自举 (only once): 复用仓库 deploy 脚本的同款依赖集, 幂等
      preStart = ''
        venv="${cfg.agentPath}/.venv"
        marker="${cfg.agentPath}/.venv/.huginn_ready"
        if [ ! -e "$marker" ]; then
          ${pkgs.python311}/bin/python -m venv "$venv"
          "$venv/bin/pip" install --no-input -q '.[dev]' \
            cryptography pydantic "langchain" "langchain-core" "langchain-openai" \
            langgraph scipy sympy z3-solver aiohttp fastapi uvicorn httpx \
            tenacity requests 2>&1 | tail -2 || true
          touch "$marker"
        fi
      '';

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.agentPath;
        ExecStart = "${cfg.agentPath}/.venv/bin/python -m huginn.server --port ${toString cfg.port}";
        Environment = [
          "PYTHONPATH=${cfg.agentPath}"
          "HUGINN_CACHE_DIR=${cfg.runtimeDir}"
          "HUGINN_MEMORY_DIR=${cfg.runtimeDir}/memory"
          "WORKSPACE_DIR=${cfg.workspaceDir}"
        ] ++ (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.environment);
        # 密钥只从 EnvironmentFile 来, 不进单元图 / 命令行
        EnvironmentFile = lib.mkIf (cfg.envFile != null) cfg.envFile;

        # ── hardening ──
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        NoNewPrivileges = true;
        RestrictSUIDSGID = true;
        PrivateDevices = true;
        MemoryDenyWriteExecute = false; # 科学包常要 JIT (cython/numba), 放宽写执行
        RestrictNamespaces = false;
        # 只允许写 runtime/workspace, 其余根只读
        ReadWritePaths = [ cfg.runtimeDir cfg.workspaceDir ];
        # 自动重启, 限制重启频率防抖
        Restart = "on-failure";
        RestartSec = "5s";
        # before 健康检查: 快速失败
      };

      # 只开健康/诊断可无鉴权, 其余全走 API key (配合 server.py 默认中间件)
    };
  };
}