#!/usr/bin/env node
// 通用启动器: 选 server -> 找命令(PATH) -> spawn 并透传 stdio (MCP stdio 必需).
'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const SERVERS = {
  'mat-db-mcp': { cmd: 'mat-db-mcp', args: [], need: 'matsci-mat-db-mcp' },
  'math-anything-mcp': { cmd: 'math-anything-mcp', args: [], need: 'matsci-math-anything-mcp' },
  'vision-pixel-mcp': { cmd: 'vision-pixel-mcp', args: [], need: 'matsci-vision-pixel-mcp' },
  // Huginn CLI console script 是 huginn-agent (旧版曾用 huginn), 两者都兜底.
  'capabilities-mcp': { cmds: ['huginn-agent', 'huginn'], args: ['capabilities-mcp'], need: 'huginn-agent' },
  'workflows-mcp': { cmds: ['huginn-agent', 'huginn'], args: ['workflows-mcp'], need: 'huginn-agent' },
};

const EXTS = process.platform === 'win32' ? ['', '.exe', '.cmd', '.bat'] : [''];

function which(cmd) {
  const dirs = (process.env.PATH || '').split(path.delimiter);
  for (const d of dirs) {
    if (!d) continue;
    for (const ext of EXTS) {
      const p = path.join(d, cmd + ext);
      if (fs.existsSync(p)) return p;
    }
  }
  return null;
}

function start(serverName) {
  const spec = SERVERS[serverName];
  if (!spec) {
    console.error(`[huginn-mcp] 未知 server: ${serverName}`);
    process.exit(1);
  }
  const cmds = spec.cmds || [spec.cmd];
  const bin = cmds.map(which).find(Boolean);
  if (!bin) {
    console.error(
      `[${serverName}] 找不到 ${cmds.map((c) => `'${c}'`).join(' / ')}。请先安装依赖：pip install ${spec.need}`
    );
    process.exit(1);
  }
  const child = spawn(bin, spec.args, { stdio: 'inherit' });
  child.on('error', (err) => {
    console.error(`[${serverName}] 启动失败: ${err.message}`);
    process.exit(1);
  });
  child.on('exit', (code) => {
    process.exit(code ?? 1);
  });
}

module.exports = start;