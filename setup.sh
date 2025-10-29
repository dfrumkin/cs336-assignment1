#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not on PATH." >&2
  echo "Install from https://github.com/astral-sh/uv and re-run this script." >&2
  exit 1
fi

echo "==> Syncing environment with uv..."
uv sync

echo "==> Activating the environment..."
source .venv/bin/activate

echo "==> Enabling nbdime Git integration..."
nbdime config-git --enable

echo "==> Installing pre-commit hooks..."
pre-commit install
