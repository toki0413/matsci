# flake.nix —— Huginn agent 的 NixOS 集成入口
#
# 提供三类资产 (对齐 NixKits 的主线「万物皆插件」):
#   1. nixosModules.huginn-agent        — 把 Huginn FastAPI 后端声明式跑成 hardened systemd 服务
#   2. nixosModules.huginn-rebuild      — DSH 式 detached `nixos-rebuild` 分发助手 (systemd-run --collect)
#   3. devShells.default                — 本地 Nix 开发环境
#
# 设计约束:
#   - agent 代码体量大、科学依赖众 (pymatgen/scipy/z3/langchain…), 不强行 nix 翻译每个依赖,
#     而是复用仓库自带 deploy 脚本的 venv 约定 (见 deploy_and_run_long_research.sh):
#     模块按需自举 `.venv`, systemd 只负责把服务声明式、可复现、带 hardening 地跑起来.
#   - 所有密钥只走 EnvironmentFile 注入的环境, 不落盘到代码/日志.
#   - 不对无 NixOS 的机器做任何假设: 模块默认关断言, 纯声明即可 import.
{
  description = "Huginn scientific autonomous agent — NixOS integration";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      # legacyPackages.${system} 本身就是 pkgs (含 mkShell / python311 / ruff),
      # 直接整体传下去, 不依赖 .pkgs 别名.
      perSystem = fn:
        nixpkgs.lib.genAttrs systems (system: fn nixpkgs.legacyPackages.${system});
    in
    {
      # ── 可 import 的 NixOS 模块 (用法见仓库 HUGINN_GUIDE 或 README) ──
      nixosModules = {
        huginn-agent = import ./modules/huginn-agent.nix;
        huginn-rebuild = import ./modules/huginn-rebuild-dispatch.nix;
        # 一次全载: agent 服务 + 重建分发助手
        default = {
          imports = [
            self.nixosModules.huginn-agent
            self.nixosModules.huginn-rebuild
          ];
        };
      };

      # ── 本地开发环境 ──
      devShells = perSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.python311
            (pkgs.python311.withPackages (ps: with ps; [
              fastapi uvicorn httpx pydantic
            ]))
            pkgs.ruff
          ];
          shellHook = ''
            echo "Huginn Nix dev shell. agent 代码在 ./agent, 其余科学依赖用 venv 自装."
          '';
        };
      });
    };
}