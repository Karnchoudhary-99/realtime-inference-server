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
| `POST` | `/v1/admin/models` | register a model from a `ModelSpec` → `201` (conflict `409`, bad spec `400`) |
| `PUT` | `/v1/admin/models/{id}` | hot-swap the implementation behind `id` (in-flight requests keep going) |
| `DELETE` | `/v1/admin/models/{id}` | unregister `id` → `204` (unknown `404`) |

Errors are JSON: `{"error": {"code": 404, "message": "unknown model x"}}`.
Status codes: `400` bad JSON / shape / non-streaming model, `404` unknown
model, `409` duplicate model id, `429` saturated, `500` inference failure,
`503` not ready, `504` request timeout.

## Admin API (runtime model registration)

The admin API lets operators add, hot-swap and remove models **without
redeploying**. Only the built-in zoo types can be instantiated (deliberate
security boundary — arbitrary code can't be loaded at runtime):

```bash
# register a custom MLP (try examples/admin.sh for the full tour)
curl -X POST localhost:8080/v1/admin/models \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"id":"custom-mlp","type":"mlp","version":"v2","input_dim":3,"output_dim":2,"hidden":8,"seed":123}'

# hot-swap it to a bigger net — same id, new version, zero downtime
curl -X PUT localhost:8080/v1/admin/models/custom-mlp \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"type":"mlp","version":"v3","input_dim":3,"output_dim":2,"hidden":64,"seed":7}'

# remove it
curl -X DELETE localhost:8080/v1/admin/models/custom-mlp \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

`ModelSpec` fields: `type` (`mlp` | `linear` | `echo` | `storygen`),
`version` (default `v1`), `input_dim` / `output_dim` (mlp, linear),
`hidden` (mlp, default 16), `seed` (deterministic weights),
`num_tokens` / `token_delay_ms` (storygen). Set `ADMIN_TOKEN` (or
`--admin-token`) to protect the admin routes with a bearer token; leave it
empty only for local development — the server warns on startup.

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
--admin-token <token>        bearer token for /v1/admin/* (or ADMIN_TOKEN env)
```

Or run it in Docker:

```bash
docker build -t inference-server .
docker run -p 8080:8080 -e ADMIN_TOKEN=secret inference-server

## Project layout

- `cmd/server` — entrypoint: flags, built-in models, graceful shutdown
- `cmd/loadtest` — throughput/latency benchmark harness
- `internal/server` — HTTP API, middleware, SSE streaming
- `internal/registry` — model registry with hot-swap
- `internal/batcher` — dynamic request batching
- `internal/models` — model interface + built-in models
- `internal/metrics` — Prometheus-style metrics
- `examples/` — sample requests (`predict.sh`, `admin.sh`)
- `Dockerfile` — multi-stage build for a minimal container image

## Roadmap

- [x] Project scaffold (go.mod, layout, README, CI)
- [x] Model registry: register/version/list models, hot-swap
- [x] HTTP API: `POST /v1/models/{id}/predict`, health + readiness probes
- [x] Dynamic request batching (max batch size / latency budget)
- [x] Streaming responses (SSE) for token-style outputs
- [x] Concurrency limits, graceful shutdown, request timeouts
- [x] Metrics: latency histograms, throughput, queue depth (`/metrics`)
- [x] Load-test harness + benchmark numbers in README
- [x] Admin API for registering custom models at runtime (register/hot-swap/delete, bearer token)
- [ ] Persistent model artifacts (ONNX weights via onnxruntime-go)
- [ ] Distributed tracing hooks

## License

MIT — see [LICENSE](LICENSE).
