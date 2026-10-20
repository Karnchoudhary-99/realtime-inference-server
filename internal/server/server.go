// Package server wires the registry, batchers and metrics into an HTTP API:
//
//	GET  /healthz                  liveness probe
//	GET  /readyz                   readiness probe (true once a model is registered)
//	GET  /v1/models                list registered models
//	POST /v1/models/{id}/predict   batched inference (JSON)
//	GET  /v1/models/{id}/stream     token streaming over SSE (see stream.go)
//	POST /v1/admin/models           register a model at runtime (see admin.go)
//	PUT  /v1/admin/models/{id}      hot-swap the implementation behind id
//	DELETE /v1/admin/models/{id}    unregister id
//	GET  /metrics                  Prometheus exposition
package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/batcher"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/metrics"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/registry"
	"strconv"
)

// Config tunes the server's runtime behavior.
type Config struct {
	// MaxBatchSize caps rows per Predict call.
	MaxBatchSize int
	// MaxBatchLatency is how long a batcher waits to fill a batch.
	MaxBatchLatency time.Duration
	// MaxPerRequest caps rows in a single predict request.
	MaxPerRequest int
	// MaxInflight caps concurrent predict/stream requests; excess gets 429.
	MaxInflight int
	// RequestTimeout bounds each predict/stream request.
	RequestTimeout time.Duration
	// AdminToken, when non-empty, requires `Authorization: Bearer <token>`
	// on /v1/admin/* routes. Leave empty only for local development.
	AdminToken string
}

// DefaultConfig returns sane defaults for local development.
func DefaultConfig() Config {
	return Config{
		MaxBatchSize:    32,
		MaxBatchLatency: 25 * time.Millisecond,
		MaxPerRequest:   256,
		MaxInflight:     200,
		RequestTimeout:  30 * time.Second,
	}
}

// Server is the inference HTTP server.
type Server struct {
	reg     *registry.Registry
	metrics *metrics.Registry
	cfg     Config

	inflightSem chan struct{}

	mu       sync.Mutex
	batchers map[string]*batcher.Batcher

	reqID atomic.Uint64

	// metric handles
	httpRequests  *metrics.Counter
	inferRequests *metrics.Counter
	inferErrors   *metrics.Counter
	latency       *metrics.Histogram
	batchSize     *metrics.Histogram
	queueDepth    *metrics.Gauge
	inflight      *metrics.Gauge
	adminOps      *metrics.Counter
}

// New builds a Server. Call Shutdown when done to stop batcher goroutines.
func New(reg *registry.Registry, m *metrics.Registry, cfg Config) *Server {
	s := &Server{
		reg:         reg,
		metrics:     m,
		cfg:         cfg,
		inflightSem: make(chan struct{}, cfg.MaxInflight),
		batchers:    make(map[string]*batcher.Batcher),
	}
	s.httpRequests = m.NewCounter("http_requests_total", "HTTP requests by method, route and status")
	s.inferRequests = m.NewCounter("inference_requests_total", "Inference requests by model")
	s.inferErrors = m.NewCounter("inference_errors_total", "Failed inference requests by model and reason")
	s.latency = m.NewHistogram("inference_latency_seconds", "End-to-end inference latency by model", metrics.DefaultLatencyBuckets())
	s.batchSize = m.NewHistogram("inference_batch_size_rows", "Rows per Predict call by model", []float64{1, 2, 4, 8, 16, 32, 64, 128, 256})
	s.queueDepth = m.NewGauge("inference_queue_depth", "Pending requests per model batcher")
	s.inflight = m.NewGauge("inference_inflight", "Requests currently executing")
	s.adminOps = m.NewCounter("admin_operations_total", "Admin API operations by op and result")
	return s
}

// Shutdown stops all batcher goroutines.
func (s *Server) Shutdown() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, b := range s.batchers {
		b.Stop()
	}
}

// batcherFor returns the batcher for a model, creating it lazily.
func (s *Server) batcherFor(id string, mdl models.Model) *batcher.Batcher {
	s.mu.Lock()
	defer s.mu.Unlock()
	if b, ok := s.batchers[id]; ok {
		return b
	}
	labels := metrics.Labels{"model": id}
	b := batcher.New(mdl, s.cfg.MaxBatchSize, s.cfg.MaxBatchLatency, s.cfg.MaxPerRequest,
		func(n int) { s.queueDepth.Set(labels, float64(n)) })
	s.batchers[id] = b
	return b
}

// Handler returns the full route tree with middleware applied.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /v1/models", s.handleListModels)
	mux.HandleFunc("POST /v1/models/{id}/predict", s.handlePredict)
	mux.HandleFunc("GET /v1/models/{id}/stream", s.handleStream)
	mux.HandleFunc("POST /v1/admin/models", s.adminGuard(s.handleAdminRegister))
	mux.HandleFunc("PUT /v1/admin/models/{id}", s.adminGuard(s.handleAdminHotswap))
	mux.HandleFunc("DELETE /v1/admin/models/{id}", s.adminGuard(s.handleAdminDelete))
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	return s.withMiddleware(mux)
}

// --- middleware ---

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := s.reqID.Add(1)
		w.Header().Set("X-Request-ID", itoa(requestID))
		rw := &statusWriter{ResponseWriter: w, status: http.StatusOK}

		// Concurrency limit: fail fast with 429 instead of queueing forever.
		select {
		case s.inflightSem <- struct{}{}:
			defer func() { <-s.inflightSem }()
		default:
			s.inferErrors.Inc(metrics.Labels{"model": r.PathValue("id"), "reason": "overloaded"}, 1)
			writeError(rw, http.StatusTooManyRequests, "server overloaded, retry later")
			return
		}
		s.inflight.Add(metrics.Labels{}, 1)
		defer s.inflight.Add(metrics.Labels{}, -1)

		// Per-request timeout.
		ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
		defer cancel()

		next.ServeHTTP(rw, r.WithContext(ctx))
		s.httpRequests.Inc(metrics.Labels{
			"method": r.Method,
			"route":  routeOf(r),
			"status": http.StatusText(rw.status),
		}, 1)
	})
}

// routeOf collapses concrete model ids so route cardinality stays low.
func routeOf(r *http.Request) string {
	p := r.URL.Path
	switch {
	case len(p) >= len("/v1/models/") && p[:len("/v1/models/")] == "/v1/models/":
		rest := p[len("/v1/models/"):]
		for i := 0; i < len(rest); i++ {
			if rest[i] == '/' {
				return "/v1/models/{id}" + rest[i:]
			}
		}
		return "/v1/models/{id}"
	default:
		return p
	}
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

// Flush forwards http.Flusher so handlers (e.g. SSE) keep working behind
// this middleware's wrapper.
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// --- handlers ---

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if s.reg.Len() == 0 {
		writeError(w, http.StatusServiceUnavailable, "no models registered")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready", "models": s.reg.Len()})
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"models": s.reg.List()})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.metrics.WritePrometheus(w)
}

// predictRequest is the JSON body of POST /v1/models/{id}/predict.
type predictRequest struct {
	Inputs [][]float64    `json:"inputs"`
	Params map[string]any `json:"params,omitempty"`
}

// predictResponse is the JSON body returned for a predict call.
type predictResponse struct {
	Model     string      `json:"model"`
	Version   string      `json:"version"`
	Outputs   [][]float64 `json:"outputs"`
	LatencyMs float64     `json:"latency_ms"`
	RequestID uint64      `json:"request_id"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	mdl, ok := s.reg.Get(id)
	if !ok {
		s.inferErrors.Inc(metrics.Labels{"model": id, "reason": "unknown_model"}, 1)
		writeError(w, http.StatusNotFound, "unknown model "+id)
		return
	}
	var req predictRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20)).Decode(&req); err != nil {
		s.inferErrors.Inc(metrics.Labels{"model": id, "reason": "bad_request"}, 1)
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if len(req.Inputs) == 0 {
		writeError(w, http.StatusBadRequest, "inputs must be a non-empty array")
		return
	}
	if len(req.Inputs) > s.cfg.MaxPerRequest {
		writeError(w, http.StatusBadRequest,
			"inputs exceeds per-request limit")
		return
	}
	// Fail fast on shape errors as 400s; anything the model raises later is
	// a genuine inference failure (500).
	if err := models.CheckInputs(mdl.Name(), mdl.InputDim(), req.Inputs); err != nil {
		s.inferErrors.Inc(metrics.Labels{"model": id, "reason": "bad_shape"}, 1)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	start := time.Now()
	requestID := s.reqID.Add(1)
	s.inferRequests.Inc(metrics.Labels{"model": id}, 1)

	b := s.batcherFor(id, mdl)
	outputs, err := b.Submit(r.Context(), req.Inputs)
	elapsed := time.Since(start)
	s.latency.Observe(metrics.Labels{"model": id}, elapsed.Seconds())

	if err != nil {
		reason := "inference_error"
		status := http.StatusInternalServerError
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			reason = "timeout"
			status = http.StatusGatewayTimeout
		}
		s.inferErrors.Inc(metrics.Labels{"model": id, "reason": reason}, 1)
		writeError(w, status, err.Error())
		return
	}

	// Record the effective batch size from the batcher's stats delta is
	// racy; instead observe per-request rows as a throughput signal.
	s.batchSize.Observe(metrics.Labels{"model": id}, float64(len(req.Inputs)))

	writeJSON(w, http.StatusOK, predictResponse{
		Model:     id,
		Version:   mdl.Version(),
		Outputs:   outputs,
		LatencyMs: float64(elapsed.Microseconds()) / 1000,
		RequestID: requestID,
	})
}

// --- helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"code": status, "message": msg}})
}

func itoa(n uint64) string { return strconv.FormatUint(n, 10) }
