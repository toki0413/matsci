# modules/huginn-nixos-cli.nix
#
# 给 agent 后端提供 NixOS **只读** 诊断命令 + job 路由策略。
#
# 对齐参考 `dsh-nixos-shell` 的 `nixos_cli` (读 lib/nixos-cli handler, nixos-gate):
#   只读诊断走本命令 (无 sudo), 变更性操作 (nixos-rebuild / nixos apply /
#   systemctl restart dsh) 一律走 `huginn-rebuild`(改造分发助手, detached + root)。
#
#   capabilities        traditional→modern 命令对照 + 重建指引
#   system-status       系统运行态 + 失败单元     (systemctl is-system-running / --failed)
#   generations         系统代际 (只读列 /nix/var/nix/profiles 符号链接, 默认 20, 上限 200)
#   journal <unit> [n]  按单元的日志尾 (默认 50, 上限 500; 支持 journalctl glob, 尾 @=&
#                       =模板所有实例)
#   audit-store-paths   扫描用户配置里的硬编码 /nix/store/ 路径 (gc 后失效的坑)
#
# 命令行: huginn-nixos-cli <op> [args...]
{
  config, lib, pkgs, ... }:
let
  cfg = config.services.huginn-nixos-cli;

  script = pkgs.writeShellScriptBin "huginn-nixos-cli" ''
    set -euo pipefail
    op="''${1:-}"
    case "$op" in
      capabilities)
        echo "traditional -> modern (NixOS):"
        echo "  nix-env -i -> nix profile install"
        echo "  nix-env -e -> nix profile remove"
        echo "  nixos-rebuild switch -> nixos apply /etc/nixos (或 nixos-rebuild switch --flake /etc/nixos)"
        echo "  nix-collect-garbage -> nix store gc / nix store optimise"
        echo "  nix-channel -> flakes / nix registry"
        echo "重建指引: 变更 /etc/nixos 后必须 rebuild; 失败停留上一代际, 修复后再试."
        ;;

      system-status)
        echo "is-system-running: $(systemctl is-system-running 2>&1)"
        echo "--- failed units ---"
        systemctl --failed --no-legend --no-pager || true
        ;;

      generations)
        dir="/nix/var/nix/profiles"
        limit="''${2:-20}"
        case "$limit" in (*[!0-9]*|'') limit=20;; esac
        [ "$limit" -gt 200 ] && limit=200
        if [ -L "$dir/system" ]; then
          echo "current: $(readlink "$dir/system")"
        else
          echo "current: (none)"
        fi
        echo "--- generations (newest last, up to $limit) ---"
        ls -1d "$dir"/system-*-link 2>/dev/null | sort -V | tail -n "$limit" || true
        ;;

      journal)
        unit="''${2:?usage: huginn-nixos-cli journal <unit> [lines]}"
        lines="''${3:-50}"
        case "$lines" in (*[!0-9]*|'') lines=50;; esac
        [ "$lines" -gt 500 ] && lines=500
        # 对齐参考: 尾 @ 表示模板所有实例 → 加 *; 只白名单合法字符
        case "$unit" in (*@) unit="$unit*";; esac
        case "$unit" in
          *[!A-Za-z0-9@._:*%-]*)
            echo "invalid unit name: allowed A-Za-z0-9@._:*%-" >&2; exit 3 ;;
        esac
        journalctl -u "$unit" -n "$lines" --no-pager || true
        ;;

      audit-store-paths)
        # 扫描用户配置里硬编码 /nix/store/ 绝对路径 —— gc 后立即失效的坑
        for f in "$HOME"/.gitconfig "$HOME"/.bashrc "$HOME"/.zshrc "$HOME"/.profile; do
          [ -f "$f" ] || continue
          grep -n '/nix/store/' "$f" 2>/dev/null | sed "s#^#$f:#" || true
        done
        echo "--- git credential helper ---"
        git config --get credential.https://github.com.helper || echo "(unset)"
        echo "规则: 配置文件里优先裸命令名($PATH) 或 /run/current-system/sw/bin 稳定符号链接,"
        echo "      不要写 /nix/store/ 绝对路径 (nix store gc 后失效)."
        ;;

      *)
        echo "用法: huginn-nixos-cli {capabilities|system-status|generations [n]|journal <unit> [n]|audit-store-paths}" >&2
        exit 2
        ;;
    esac
  '';
in
{
  options.services.huginn-nixos-cli = {
    enable = lib.mkEnableOption "the read-only NixOS diagnostics CLI for the agent backend";
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ script ];
  };
}