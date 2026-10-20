#!/usr/bin/env bash
# Admin API tour: register, hot-swap and delete models at runtime.
# The server must be running with the same token, e.g.:
#   ADMIN_TOKEN=demo-secret ./server --addr :8080
set -euo pipefail
BASE="${BASE:-http://localhost:8080}"
TOKEN="${ADMIN_TOKEN:-demo-secret}"
AUTH=(-H "Authorization: Bearer $TOKEN")

echo "== admin without token -> 401 =="
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$BASE/v1/admin/models" \
  -H 'Content-Type: application/json' -d '{"id":"x","type":"echo"}'

echo "== register an mlp model =="
curl -s -X POST "$BASE/v1/admin/models" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"id":"custom-mlp","type":"mlp","version":"v2","input_dim":3,"output_dim":2,"hidden":8,"seed":123}'; echo

echo "== predict with the new model =="
curl -s -X POST "$BASE/v1/models/custom-mlp/predict" \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[0.5,-0.2,0.9],[1.0,0.0,-1.0]]}'; echo

echo "== hot-swap to a bigger net (same id, new version) =="
curl -s -X PUT "$BASE/v1/admin/models/custom-mlp" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"type":"mlp","version":"v3","input_dim":3,"output_dim":2,"hidden":64,"seed":7}'; echo

echo "== bad spec -> 400 =="
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$BASE/v1/admin/models" "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d '{"id":"bad","type":"nope"}'

echo "== duplicate id -> 409 =="
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$BASE/v1/admin/models" "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d '{"id":"custom-mlp","type":"echo"}'

echo "== delete the model =="
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE "$BASE/v1/admin/models/custom-mlp" "${AUTH[@]}"

echo "== gone from the registry =="
curl -s "$BASE/v1/models" | head -c 300; echo
