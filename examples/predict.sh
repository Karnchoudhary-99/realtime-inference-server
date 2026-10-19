#!/usr/bin/env bash
# Example requests against a locally running realtime-inference-server.
set -euo pipefail
BASE="${BASE:-http://localhost:8080}"

echo "== health =="
curl -s "$BASE/healthz"; echo

echo "== list models =="
curl -s "$BASE/v1/models" | head -c 400; echo; echo

echo "== predict (iris-mlp) =="
curl -s -X POST "$BASE/v1/models/iris-mlp/predict" \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[5.1,3.5,1.4,0.2],[6.0,2.2,4.0,1.0]]}'; echo

echo "== stream (storygen, first frames) =="
curl -s -N "$BASE/v1/models/storygen/stream?prompt=hello" | head -8

echo "== metrics (sample) =="
curl -s "$BASE/metrics" | grep -E '^(inference_requests_total|inference_latency_seconds_count)' | head -6
