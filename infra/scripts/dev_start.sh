#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
PYTHON="${PYTHON:-python3}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

check_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command '$cmd' not found in PATH." >&2
    exit 1
  fi
}

setup_backend() {
  cd "$BACKEND_DIR"
  if [[ "$SKIP_INSTALL" != "1" ]]; then
    check_command "$PYTHON"
    if [[ ! -d .venv ]]; then
      echo "[backend] Creating virtual environment (.venv)..."
      "$PYTHON" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "[backend] Installing Python dependencies..."
    pip install -r requirements.txt
  elif [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
}

start_backend() {
  cd "$BACKEND_DIR"
  echo "[backend] Starting API on port $BACKEND_PORT (reload enabled)..."
  exec uvicorn src.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
}

setup_frontend() {
  cd "$FRONTEND_DIR"
  check_command npm
  if [[ "$SKIP_INSTALL" != "1" && ! -d node_modules ]]; then
    echo "[frontend] Installing npm dependencies..."
    npm install
  fi
}

start_frontend() {
  cd "$FRONTEND_DIR"
  echo "[frontend] Starting dev server on port $FRONTEND_PORT..."
  exec npm run dev -- --hostname 0.0.0.0 --port "$FRONTEND_PORT"
}

main() {
  echo "============================================"
  echo "   SecOps AI — Development Environment"
  echo "============================================"
  echo "Root directory:      $ROOT_DIR"
  echo "Backend directory:   $BACKEND_DIR"
  echo "Frontend directory:  $FRONTEND_DIR"
  echo "Backend port:        $BACKEND_PORT"
  echo "Frontend port:       $FRONTEND_PORT"
  echo "Skip installs:       $SKIP_INSTALL"
  echo ""

  setup_backend
  check_command uvicorn
  setup_frontend

  trap 'echo "\nStopping dev processes..."; jobs -p | xargs -r kill' EXIT

  (
    start_backend
  ) &

  (
    start_frontend
  ) &

  wait
}

main "$@"
