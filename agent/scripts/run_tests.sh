#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HUGINN_PROMPT_CACHE_CONTROL="0"

echo "Running lint..."
ruff check huginn tests
black --check huginn tests

echo "Running tests with coverage..."
python -m pytest tests -q "$@"
