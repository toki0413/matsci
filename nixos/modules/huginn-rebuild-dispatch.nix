# modules/huginn-rebuild-dispatch.nix
#
# DSH 式 detached `nixos-rebuild` 分发助手。
#
# 语义对齐参考实现 `dsh-nixos-shell` (DeepSeek Harness NixOS 插件) 的
# `wrapDetachedSystemCommand` + sudo daemon Protocol v3 ("Dsh Sudo Daemon Protocol"):
#
#   为什么 rebuild 必须 DETACHED:
#     nixos-rebuild switch 的激活阶段 (switch-to-configuration) 会 *重启 dsh.service*
#     并 *stop/start nixkits-sudo.socket*。若 rebuild 作为 socket 常驻子进程同步跑,
#     socket 一 stop 就会连同 switch 进程一起整树杀掉 → 激活中途死掉, 系统留在
#     部分激活状态, socket 也起不来。瞬态 systemd 单元有独立 cgroup: socket stop
#     与 dsh 重启都够不到它, 激活能完整跑完, socket 随后自己恢复。
#
#   因此:
#     - 内部命令用转义后的 bash -c, 显式 export NixOS profile PATH —— 瞬态单元
#       继承的是 systemd manager-default 环境 (只有 coreutils/findutils/... store
#       路径), 不 export 的话 nixos-rebuild 根本解析不到。
#     - 所有可执行器 (systemd-run / bash / nixos-rebuild) 用绝对 store 路径。
#     - 连接断开 ≠ 取消: systemd-run 不加 --wait, 调用方断连后瞬态单元继续跑。
#     - 取消是显式带内操作: `huginn-rebuild cancel <unit>` → systemctl cancel,
#       KillMode=control-group 整组回收 (shell + 孙进程)。
#     - 不给 switch 硬超时/run 上限: 守护侧 6h MAX_TIMEOUT_MS 只约束普通命令,
#       激活中途被 SIGTERM 留下部分激活状态比超时更糟 (对齐参考未对 detached
#       rebuild 单元设 RuntimeMaxSec / TimeoutStopSec 大值的取舍)。
#
# 命令: huginn-rebuild {dry-run|switch|cancel <unit>|status}
{
  config, lib, pkgs, ... }:
let
  cfg = config.services.huginn-rebuild;

  dispatchDir = cfg.dispatchDir;

  # 绝对 store 可执行器
  systemdRunBin = "/run/current-system/sw/bin/systemd-run";
  systemctlBin = "/run/current-system/sw/bin/systemctl";
  bashBin = "/run/current-system/sw/bin/bash";
  nixosRebuildBin = cfg.rebuildBin;

  # 显式 NixOS profile PATH (对齐守护进程的 PATH 注入: 必须整体覆盖, 而非只追加,
  # 因为 manager-default 环境会覆盖掉 profile 工具解析.)
  dshPathEnv = "/run/current-system/sw/bin:/run/wrappers/bin:/etc/profiles/per-user/root/bin:/nix/var/nix/profiles/default/bin:/usr/local/bin:/usr/bin:/bin";

  script = pkgs.writeShellScriptBin "huginn-rebuild" ''
    set -euo pipefail
    mkdir -p ${dispatchDir}
    case "''${1:-}" in
      dry-run)
        ${systemdRunBin} --collect --quiet \
          --unit="huginn-rebuild-dry-$(date +%s)" \
          --property=KillMode=control-group -- \
          ${bashBin} -c "export PATH=${dshPathEnv}; cd / && ${nixosRebuildBin} dry-run"
        ;;

      switch)
        unit="huginn-rebuild-$(date +%s)"
        # 脱离当前会话: 不加 --wait, 断连不取消; --collect 完成后自行清理.
        # 输出落到 dispatchDir 供 status 摘要, 实时进度看 journalctl -u $unit.
        ${systemdRunBin} --collect --quiet --unit="$unit" \
          --property=KillMode=control-group -- \
          ${bashBin} -c "export PATH=${dshPathEnv}; cd / && ${nixosRebuildBin} switch --show-trace > ${dispatchDir}/$unit.log 2>&1; echo \$? > ${dispatchDir}/$unit.status"
        echo "dispatch ok: $unit — 进展: journalctl -u $unit ; 摘要: ${dispatchDir}/$unit.log"
        ;;

      cancel)
        : "''${2:?usage: huginn-rebuild cancel <unit>}"
        ${systemctlBin} cancel "$2"
        echo "cancel requested: $2"
        ;;

      status)
        out=0
        for f in ${dispatchDir}/*.status; do
          [ -e "$f" ] || continue
          u="$(basename "$f" .status)"
          echo "$u = $(cat "$f")  log: ${dispatchDir}/$u.log"
          out=1
        done
        [ "$out" = 0 ] && echo "(no rebuild runs yet)"
        ;;

      *)
        echo "用法: huginn-rebuild {dry-run|switch|cancel <unit>|status} (DSH v3: 断连≠取消, 取消走 cancel <unit>)" >&2
        exit 2
        ;;
    esac
  '';
in
{
  options.services.huginn-rebuild = {
    enable = lib.mkEnableOption "the detached nixos-rebuild dispatch helper (DSH v3 semantics)";

    dispatchDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/huginn/rebuild";
      description = "重建进度摘要落盘目录 (log + .status 码).";
    };

    rebuildBin = lib.mkOption {
      type = lib.types.path;
      default = "/run/current-system/sw/bin/nixos-rebuild";
      description = "nixos-rebuild 绝对路径 (当前系统 store 路径).";
    };

    allowedUsers = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "允许调用 huginn-rebuild 的系统用户 (空=仅 root). 用于给 agent 指定运行账号.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.tmpfiles.rules = [
      "d ${dispatchDir} 0755 root root -"
    ];

    environment.systemPackages = [ script ];

    # 可选: 通过 sudoers 把分发助手开放给指定用户 (如 huginn-agent 的运行身份),
    # 且只放行这个白名单命令, 不给完整外壳.
    security.sudo.extraRules = lib.mkIf (cfg.allowedUsers != [ ]) [
      {
        users = cfg.allowedUsers;
        commands = [
          { command = "/run/current-system/sw/bin/huginn-rebuild *"; options = [ "NOPASSWD" ]; }
        ];
      }
    ];
  };
}