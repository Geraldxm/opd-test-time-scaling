#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$ROOT/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.11}"

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
  echo "requirements-gpu.lock supports Linux x86_64 only" >&2
  exit 1
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ -n "${DOWNLOAD_PROXY:-}" ]]; then
  export HTTP_PROXY="$DOWNLOAD_PROXY" HTTPS_PROXY="$DOWNLOAD_PROXY"
  export http_proxy="$DOWNLOAD_PROXY" https_proxy="$DOWNLOAD_PROXY"
fi

"$UV_BIN" python install "$PYTHON_VERSION"
"$UV_BIN" venv "$ENV_DIR" --python "$PYTHON_VERSION" --seed
"$UV_BIN" pip sync "$ROOT/requirements-gpu.lock" \
  --python "$ENV_DIR/bin/python" --torch-backend=auto
"$UV_BIN" pip check --python "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" -c 'import torch, vllm; print("python env ok", torch.__version__, vllm.__version__, torch.cuda.is_available())'
