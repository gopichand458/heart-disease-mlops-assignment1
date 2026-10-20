#!/usr/bin/env bash
# Deploy the API + Prometheus + Grafana to a local minikube cluster.
# Prereqs: docker, minikube, models/best_model.joblib (run `python -m src.train` first).
set -euo pipefail
cd "$(dirname "$0")/.."
KUBECTL="minikube kubectl --"

minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=4096

# Build the image straight into minikube's Docker daemon (no registry needed)
eval "$(minikube docker-env)"
SK=$(python -c "import sklearn; print(sklearn.__version__)")
docker build --build-arg SKLEARN_VERSION="$SK" -t heart-disease-api:1.0 .

$KUBECTL apply -f k8s/deployment.yaml -f k8s/service.yaml
$KUBECTL create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ --dry-run=client -o yaml | $KUBECTL apply -f -
$KUBECTL apply -f k8s/monitoring/prometheus.yaml -f k8s/monitoring/grafana.yaml

$KUBECTL rollout status deployment/heart-api --timeout=180s
$KUBECTL rollout status deployment/prometheus --timeout=180s
$KUBECTL rollout status deployment/grafana --timeout=180s
$KUBECTL get pods,svc -o wide

cat <<MSG

API:        run 'minikube tunnel' in a separate terminal (needs sudo) then
            $KUBECTL get svc heart-api   -> EXTERNAL-IP, or use:
            minikube service heart-api --url
Prometheus: minikube service prometheus --url
Grafana:    minikube service grafana --url      (admin / admin)
MSG
