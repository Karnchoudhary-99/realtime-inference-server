# Realtime Inference Server

A production-style ML inference server in Go: serve models over HTTP with
request batching, streaming responses, a model registry, and Prometheus-style
metrics — built from scratch, no heavyweight serving framework.

## Roadmap

- [ ] Project scaffold (go.mod, layout, README, CI)
- [ ] Model registry: register/version/list models, hot-swap
- [ ] HTTP API: `POST /v1/models/{id}/predict`, health + readiness probes
- [ ] Dynamic request batching (max batch size / latency budget)
- [ ] Streaming responses (SSE) for token-style outputs
- [ ] Concurrency limits, graceful shutdown, request timeouts
- [ ] Metrics: latency histograms, throughput, queue depth (`/metrics`)
- [ ] Load-test harness + benchmark numbers in README
- [ ] Tests + docs + MIT license

## Project layout

- `cmd/server` — server entrypoint
- `internal/server` — HTTP server, routing, middleware
- `internal/registry` — model registry (register/version/hot-swap)
- `internal/batcher` — dynamic request batching
- `internal/models` — model interface + built-in stub models
- `internal/metrics` — latency/throughput/queue-depth metrics
- `examples/` — sample requests and client snippets

## Quick start

```bash
go build ./...
go run ./cmd/server
```

## License

MIT — see [LICENSE](LICENSE).
