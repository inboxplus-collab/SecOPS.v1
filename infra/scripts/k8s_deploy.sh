#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Apply Kubernetes manifests for the SecOps AI stack.

Usage:
  ./infra/scripts/k8s_deploy.sh [--namespace NAME] [--context CONTEXT] [--dry-run]

Options:
  --namespace, -n   Kubernetes namespace to use (default: secops)
  --context, -c     kubectl context to target (optional)
  --dry-run         Use 'kubectl diff' instead of 'apply' to preview changes
  --help, -h        Show this help message

Environment:
  KUBECTL           Override kubectl executable (default: kubectl)
USAGE
}

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/k8s"
KUBECTL_BIN="${KUBECTL:-kubectl}"
NAMESPACE="secops"
CONTEXT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace|-n)
      NAMESPACE="$2"
      shift 2
      ;;
    --context|-c)
      CONTEXT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$KUBECTL_BIN" >/dev/null 2>&1; then
  echo "kubectl not found (checked: $KUBECTL_BIN)." >&2
  exit 1
fi

if [[ ! -d "$K8S_DIR" ]]; then
  echo "Kubernetes directory missing: $K8S_DIR" >&2
  exit 1
fi

CMD_ARGS=("$KUBECTL_BIN")
if [[ -n "$CONTEXT" ]]; then
  CMD_ARGS+=("--context" "$CONTEXT")
fi

ACTION="apply"
if [[ $DRY_RUN -eq 1 ]]; then
  ACTION="diff"
fi

FILES=(
  backend-deployment.yaml
  backend-service.yaml
  frontend-deployment.yaml
  frontend-service.yaml
  ingress.yaml
)

set -x
for file in "${FILES[@]}"; do
  MANIFEST="$K8S_DIR/$file"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
  fi
  "${CMD_ARGS[@]}" "$ACTION" -n "$NAMESPACE" -f "$MANIFEST"
done
set +x

echo "Kubernetes resources ${ACTION}d in namespace '$NAMESPACE'."
