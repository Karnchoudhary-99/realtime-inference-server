# Realtime Inference Server

A production-style ML inference server in Go: serve models over HTTP with
**dynamic request batching**, **SSE token streaming**, a **hot-swappable model
registry**, and **Prometheus metrics** — built from scratch on the standard
library, no serving framework.

## Quick start

```bash
go build ./...
./server --addr :8080          # or: go run ./cmd/server
```

Try it:

```bash
# list models
curl localhost:8080/v1/models

# batched inference (iris-mlp: 4 features -> 3 class probabilities)
curl -X POST localhost:8080/v1/models/iris-mlp/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[5.1,3.5,1.4,0.2],[6.0,2.2,4.0,1.0]]}'

# token streaming over SSE
curl -N 'localhost:8080/v1/models/storygen/stream?prompt=hello'

# prometheus metrics
curl localhost:8080/metrics
```

`examples/predict.sh` runs the whole tour. `cmd/loadtest` is the benchmark
harness used for the numbers below.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | liveness probe |
| `GET` | `/readyz` | readiness probe (503 until a model is registered) |
| `GET` | `/v1/models` | list registered models (id, version, dims, streaming) |
| `POST` | `/v1/models/{id}/predict` | `{"inputs": [[...], ...]}` → `{"model","version","outputs","latency_ms","request_id"}` |
| `GET` | `/v1/models/{id}/stream?prompt=...` | SSE: one `data:` frame per token, `[DONE]` terminator |
| `GET` | `/metrics` | Prometheus text exposition |

Errors are JSON: `{"error": {"code": 404, "message": "unknown model x"}}`.
Status codes: `400` bad JSON / shape / non-streaming model, `404` unknown
model, `429` saturated, `500` inference failure, `503` not ready, `504`
request timeout.

## Architecture

```
client --HTTP--> middleware (request id, timeout, concurrency limit)
                    |
                    v
              per-model Batcher --collect up to MaxBatchSize rows
                    |             or MaxBatchLatency--> single Predict call
                    v
              Model.Predict(ctx, [][]float64) -> [][]float64
```

- **`internal/models`** — `Model` interface (batched `Predict`) plus
  `StreamingModel` (token `Stream`); built-ins: `iris-mlp` (MLP classifier),
  `house-price` (linear regression), `echo` (identity, for benchmarking),
  `storygen` (deterministic token generator for the SSE demo). Weights use a
  deterministic PRNG so demos are reproducible.
- **`internal/registry`** — thread-safe register/get/list/unregister plus
  **atomic hot-swap**: replace a model's implementation without dropping
  in-flight requests.
- **`internal/batcher`** — dynamic batching: merges concurrent submits into
  one `Predict` call within a latency budget; drops cancelled waiters before
  dispatch; propagates batch errors to all waiters.
- **`internal/metrics`** — hand-rolled Prometheus counters, gauges and
  histograms (stdlib only).
- **`internal/server`** — net/http API, lazy per-model batchers, graceful
  shutdown.

## Benchmarks

`go run ./cmd/loadtest -addr http://localhost:8080 -model iris-mlp -dim 4 -rows 4 -n 2000 -c 32`
on this dev VM (flags: `--max-batch-size 32 --max-batch-latency 25ms`):

| model | req/s | rows/s | p50 | p95 | p99 |
|---|---|---|---|---|---|
| iris-mlp (MLP 4→16→3) | 6,430 | 25,720 | 3.5 ms | 16.0 ms | 24.9 ms |
| echo (identity) | 5,424 | 43,390 | 4.5 ms | 17.5 ms | 27.3 ms |

Under load the 32-row batch fills instantly, so p50 stays far below the
25 ms batching window; p99 converges on the window as expected.

## Configuration

```
--addr :8080                 listen address
--max-batch-size 32          max rows per Predict call
--max-batch-latency 25ms     how long to wait to fill a batch
--max-per-request 256        max rows in one predict request
--max-inflight 200           concurrent requests before 429
--request-timeout 30s        per-request deadline
```

## Project layout

- `cmd/server` — entrypoint: flags, built-in models, graceful shutdown
- `cmd/loadtest` — throughput/latency benchmark harness
- `internal/server` — HTTP API, middleware, SSE streaming
- `internal/registry` — model registry with hot-swap
- `internal/batcher` — dynamic request batching
- `internal/models` — model interface + built-in models
- `internal/metrics` — Prometheus-style metrics
- `examples/` — sample requests

## Roadmap

- [x] Project scaffold (go.mod, layout, README, CI)
- [x] Model registry: register/version/list models, hot-swap
- [x] HTTP API: `POST /v1/models/{id}/predict`, health + readiness probes
- [x] Dynamic request batching (max batch size / latency budget)
- [x] Streaming responses (SSE) for token-style outputs
- [x] Concurrency limits, graceful shutdown, request timeouts
- [x] Metrics: latency histograms, throughput, queue depth (`/metrics`)
- [x] Load-test harness + benchmark numbers in README
- [ ] Admin API for registering custom models at runtime
- [ ] Persistent model artifacts (ONNX weights via onnxruntime-go)
- [ ] Distributed tracing hooks

## License

MIT — see [LICENSE](LICENSE).
