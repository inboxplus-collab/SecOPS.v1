#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../backend"

if [ ! -f "alembic.ini" ]; then
  echo "alembic.ini not found in backend/. Are you in the right place?"
  exit 1
fi

echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete."
