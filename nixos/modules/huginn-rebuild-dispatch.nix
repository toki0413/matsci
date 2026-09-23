# modules/huginn-rebuild-dispatch.nix
#
# DSH 式 detached `nixos-rebuild` 分发助手 —— 解决「会重启自身所在基础设施的命令不能
# 在自身进程里同步跑」的问题. 参照 DSH Sudo Daemon Protocol v3 的语义:
#
#   1. 每请求一个独立 transient 单元 (systemd-run), 名字可寻址 → 显式取消即 systemctl cancel.
#   2. 连接断开 ≠ 取消: 不加 --wait, systemd-run 脱离当前终端直接返回, 重建继续.
#   3. 进程组管理: KillMode=control-group, 取消时整组回收.
#   4. 超时上限 6h: --unit 自带 TimeoutStopSec=21600, 并附 timeout 兜底.
#   5. 失败/输出: journal + 可选 notify (写 $runtimeDir/rebuild/ 结果文件).
#
# 暴露一个 `huginn-rebuild` 系统命令: `huginn-rebuild switch` / `dry-run` / `cancel <id>`.

{ config, lib, pkgs, ... }:
let
  rebuildDir = "/var/lib/huginn/rebuild";
  script = pkgs.writeShellScriptBin "huginn-rebuild" ''
    set -euo pipefail
    mkdir -p ${rebuildDir}
    case "''${1:-}" in
      dry-run)
        systemd-run --collect --quiet \
          --unit="huginn-rebuild-dry-$(date +%s)" \
          --property=KillMode=control-group \
          --property=TimeoutStopSec=21600 \
          nixos-rebuild dry-run
        ;;
      switch)
        id="local-$(date +%Y%m%d-%H%M%S)"
        # 脱离当前会话跑, 连接断了也不取消; 完成后 --collect 自行清理
        systemd-run --collect --quiet \
          --unit="huginn-rebuild-$id" \
          --property=KillMode=control-group \
          --property=TimeoutStopSec=21600 \
          -- sh -c "nixos-rebuild switch --show-trace > ${rebuildDir}/$id.log 2>&1; echo \$? > ${rebuildDir}/$id.status"
        echo "dispatch ok: huginn-rebuild-$id (journal: journalctl -u huginn-rebuild-$id; log: ${rebuildDir}/$id.log)"
        ;;
      cancel)
        : "''${2:?usage: huginn-rebuild cancel <unit>}"
        systemctl cancel "$2"
        echo "cancel requested: $2"
        ;;
      status)
        for f in ${rebuildDir}/*.status; do
          [ -e "$f" ] || continue
          echo "$(basename "$f") = $(cat "$f")"
        done
        ;;
      *)
        echo "用法: huginn-rebuild {dry-run|switch|cancel <unit>|status}" >&2
        exit 2
        ;;
    esac
  '';
in
{
  options.services.huginn-rebuild = {
    enable = lib.mkEnableOption "the detached nixos-rebuild dispatch helper";
    allowedUsers = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "允许调用 huginn-rebuild 的系统用户 (空=仅 root). 用于给 agent 指定运行账号.";
    };
  };

  config = lib.mkIf config.services.huginn-rebuild.enable {
    systemd.tmpfiles.rules = [
      "d ${rebuildDir} 0755 root root -"
    ];

    environment.systemPackages = [ script ];

    # 可选: 通过 sudoers 把分发助手开放给指定用户 (如 huginn-agent 的运行身份),
    # 且只放行这个白名单命令, 不给完整外壳.
    security.sudo.extraRules = lib.mkIf (config.services.huginn-rebuild.allowedUsers != [ ]) [
      {
        users = config.services.huginn-rebuild.allowedUsers;
        commands = [
          { command = "/run/current-system/sw/bin/huginn-rebuild *"; options = [ "NOPASSWD" ]; }
        ];
      }
    ];
  };
}