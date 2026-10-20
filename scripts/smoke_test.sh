#!/usr/bin/env bash
# Send a sample request to a running API. Usage: scripts/smoke_test.sh [base_url]
BASE="${1:-http://localhost:8000}"
echo "--- /health";  curl -sf "$BASE/health"; echo
echo "--- /predict (high-risk profile)"
curl -sf -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{
  "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
  "thalach": 110, "exang": 1, "oldpeak": 2.8, "slope": 2, "ca": 2, "thal": 7}'; echo
echo "--- /predict (low-risk profile)"
curl -sf -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{
  "age": 41, "sex": 0, "cp": 2, "trestbps": 120, "chol": 190, "fbs": 0, "restecg": 0,
  "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}'; echo
echo "--- invalid input (expect 422)"
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{"age": 500}'
