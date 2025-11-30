#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../k8s"

echo "Applying backend deployment & service..."
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml

echo "Applying frontend deployment & service..."
kubectl apply -f frontend-deployment.yaml
kubectl apply -f frontend-service.yaml

echo "Applying ingress..."
kubectl apply -f ingress.yaml

echo "Kubernetes resources applied."
