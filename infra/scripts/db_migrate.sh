#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run Alembic migrations for the backend service.

Usage:
  ./infra/scripts/db_migrate.sh [upgrade [revision]]
  ./infra/scripts/db_migrate.sh downgrade [revision]
  ./infra/scripts/db_migrate.sh history|current|heads
  ./infra/scripts/db_migrate.sh revision "message for new migration"
  ./infra/scripts/db_migrate.sh stamp [revision]

Examples:
  ./infra/scripts/db_migrate.sh               # upgrade to head
  ./infra/scripts/db_migrate.sh downgrade -1  # revert last migration
  ./infra/scripts/db_migrate.sh revision "Add users table"

Environment:
  DATABASE_URL    Database connection string (loaded from backend/.env if present)
  PYTHON          Python executable (default: python3)
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
ALEMBIC_INI="$BACKEND_DIR/alembic.ini"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$ALEMBIC_INI" ]]; then
  echo "alembic.ini not found at $ALEMBIC_INI"
  exit 1
fi

if ! command -v alembic >/dev/null 2>&1; then
  echo "alembic command not found. Install backend dependencies first (pip install -r requirements.txt)."
  exit 1
fi

ACTION="${1:-upgrade}"
shift || true

case "$ACTION" in
  upgrade|downgrade)
    TARGET="${1:-head}"
    ALEMBIC_ARGS=("$ACTION" "$TARGET")
    ;;
  history|current|heads)
    ALEMBIC_ARGS=("$ACTION")
    ;;
  stamp)
    TARGET="${1:-head}"
    ALEMBIC_ARGS=("stamp" "$TARGET")
    ;;
  revision)
    if [[ $# -eq 0 ]]; then
      echo "Error: provide a migration message for revision creation."
      usage
      exit 1
    fi
    ALEMBIC_ARGS=("revision" "-m" "$*")
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown action: $ACTION"
    usage
    exit 1
    ;;
esac

pushd "$BACKEND_DIR" >/dev/null

if [[ -f .env ]]; then
  echo "Loading environment variables from backend/.env"
  set -a
  source .env
  set +a
fi

echo "Running Alembic ${ALEMBIC_ARGS[*]} in $BACKEND_DIR"
"$PYTHON" -m alembic "${ALEMBIC_ARGS[@]}"

echo "Done."

popd >/dev/null
