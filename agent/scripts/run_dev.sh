#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HUGINN_PROMPT_CACHE_CONTROL="${HUGINN_PROMPT_CACHE_CONTROL:-0}"
export HUGINN_WORKSPACE="${HUGINN_WORKSPACE:-./workspace}"
mkdir -p "$HUGINN_WORKSPACE"

echo "Starting Huginn API server..."
python -m huginn.server
