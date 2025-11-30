#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)/.."

# Start backend (uvicorn) and frontend (next dev) in parallel
(
  cd "$ROOT_DIR/backend"
  echo "[backend] Installing deps and starting API..."
  pip install -r requirements.txt
  uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
) &

(
  cd "$ROOT_DIR/frontend"
  echo "[frontend] Installing deps and starting dev server..."
  npm install
  npm run dev
) &

wait
