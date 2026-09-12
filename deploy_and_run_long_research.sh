#!/usr/bin/env bash
#
# Huginn 长期自主科研闭环 —— 强机(8vCPU/32GB)一键部署 + 启动
# 目标: 方法级机理闭环 (维恩位移 / 斯涅耳折射 / 托里拆利出口速度)
# 推理: DeepSeek API
#
# 用法(在目标机器 Ubuntu 22.04+ / python3.11 上, root 或 sudo):
#   export DEEPSEEK_API_KEY=你的deepseek_key
#   bash deploy_and_run_long_research.sh
#
# 注意:
#   - 会 git clone 仓库到 /opt/matsci
#   - 默认把 workspace 放 /data/huginn_runtime (持久, 不被 /tmp 清空)
#   - 所有密钥只读进环境, 不写进任何文件/日志
set -euo pipefail

# ── 0. 必填环境变量(启动前设置) ─────────────────────────────
: "${DEEPSEEK_API_KEY:?需先 export DEEPSEEK_API_KEY=你的deepseek_key}"
HUGINN_PROVIDER="${HUGINN_PROVIDER:-deepseek}"
HUGINN_MODEL="${HUGINN_MODEL:-deepseek-chat}"

# ── 1. 路径 ────────────────────────────────────────────────
APP_DIR="${APP_DIR:-/opt/matsci}"
RUNTIME_DIR="${RUNTIME_DIR:-/data/huginn_runtime}"
MEMORY_DIR="${MEMORY_DIR:-$RUNTIME_DIR/memory}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$RUNTIME_DIR/workspace}"
GIT_REPO="${GIT_REPO:-https://github.com/toki0413/matsci.git}"

# 闭环目标(方法级机理: 维恩位移 + 斯涅耳折射 + 托里拆利)
DEFAULT_OBJECTIVE="${DEFAULT_OBJECTIVE:-Run a closed-form physics mechanism loop: (1) Wien displacement peak wavelength for T=3000 K; (2) Snell refraction angle n1=1->n2=1.5 at 60 deg incidence; (3) Torricelli exit velocity for h=2 m. For each, REAL-EXECUTE the numeric computation, give the numeric value, and a PASS/FAIL verdict against the known reference.}"
DEFAULT_ITER="${DEFAULT_ITER:-12}"

echo "==> [0/5] 校验环境"
command -v python3.11 >/dev/null 2>&1 || { echo "需要 python3.11"; exit 1; }

echo "==> [1/5] 克隆/更新仓库 → $APP_DIR"
if [ -d "$APP_DIR/agent/huginn" ]; then
  echo "  已存在, git pull 更新"
  git -C "$APP_DIR" pull --ff-only || echo "  (pull 失败, 继续用现有代码)"
else
  git clone "$GIT_REPO" "$APP_DIR"
fi
cd "$APP_DIR/agent"

echo "==> [2/5] 建立 python venv + 依赖"
if [ ! -x .venv/bin/python ]; then python3.11 -m venv .venv; fi
source .venv/bin/activate
pip install -q -e ".[dev]" 2>&1 | tail -3 || true
# 兜底补齐已知必需的核心运行时依赖(弱机实测缺过)
pip install -q "cryptography" "pydantic" "langchain" "langchain-core" \
  "langchain-openai" "langgraph" "scipy" "sympy" "z3-solver" "aiohttp" \
  "fastapi" "uvicorn" "httpx" "tenacity" "requests" 2>&1 | tail -2 || true

echo "==> [3/5] 持久运行目录(不被 /tmp 清空)"
mkdir -p "$RUNTIME_DIR" "$MEMORY_DIR" "$WORKSPACE_DIR"
export HUGINN_CACHE_DIR="$RUNTIME_DIR"
export HUGINN_MEMORY_DIR="$MEMORY_DIR"
export WORKSPACE_DIR

echo "==> [4/5] 冒烟: 确认引擎可实例化"
python - <<PY
import os
from pathlib import Path
from huginn.memory.manager import MemoryManager
from huginn.autoloop.engine import AutoloopEngine
ws = os.environ.get("WORKSPACE_DIR", "/data/huginn_runtime/workspace")
eng = AutoloopEngine(workspace=ws, memory_manager=MemoryManager())
print("SMOKE_OK", type(eng).__name__)
PY

echo "==> [5/5] 跑长期闭环 (max_iter=$DEFAULT_ITER)"
echo "  目标: ${DEFAULT_OBJECTIVE}"
export HUGINN_PROVIDER HUGINN_MODEL DEEPSEEK_API_KEY
PYTHONPATH="$APP_DIR/agent" python -m huginn.run_cognitive_driver "$DEFAULT_OBJECTIVE" "$DEFAULT_ITER"

echo "==> 完成。结果在: $WORKSPACE_DIR/run_result.json"