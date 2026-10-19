// Command server is the realtime-inference-server entrypoint. It registers
// the built-in demo models, starts the HTTP API, and shuts down gracefully
// on SIGINT/SIGTERM.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/metrics"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/registry"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/server"
)

func main() {
	var (
		addr            = flag.String("addr", ":8080", "listen address")
		maxBatchSize    = flag.Int("max-batch-size", 32, "max rows per Predict call")
		maxBatchLatency = flag.Duration("max-batch-latency", 25*time.Millisecond, "how long to wait to fill a batch")
		maxPerRequest   = flag.Int("max-per-request", 256, "max rows in one predict request")
		maxInflight     = flag.Int("max-inflight", 200, "max concurrent predict/stream requests")
		requestTimeout  = flag.Duration("request-timeout", 30*time.Second, "per-request timeout")
	)
	flag.Parse()

	reg := registry.New()
	mustRegister(reg, "iris-mlp", models.NewMLPClassifier("iris-mlp", "v1", 4, 16, 3, 42))
	mustRegister(reg, "house-price", models.NewLinearModel("house-price", "v1", 8, 1, 7))
	mustRegister(reg, "echo", models.NewEchoModel("echo", "v1"))
	mustRegister(reg, "storygen", models.NewStoryGenerator("storygen", "v1", 24, 40*time.Millisecond))

	m := metrics.New()
	srv := server.New(reg, m, server.Config{
		MaxBatchSize:    *maxBatchSize,
		MaxBatchLatency: *maxBatchLatency,
		MaxPerRequest:   *maxPerRequest,
		MaxInflight:     *maxInflight,
		RequestTimeout:  *requestTimeout,
	})

	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Graceful shutdown: stop accepting, drain in-flight, then stop batchers.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		log.Println("shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			log.Printf("http shutdown: %v", err)
		}
		srv.Shutdown()
	}()

	fmt.Printf("realtime-inference-server listening on %s (%d models)\n", *addr, reg.Len())
	for _, e := range reg.List() {
		fmt.Printf("  - %-12s %-4s in=%d out=%d streaming=%v\n",
			e.ID, e.Version, e.InputDim, e.OutputDim, e.Streaming)
	}
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("listen: %v", err)
	}
	log.Println("stopped")
}

func mustRegister(reg *registry.Registry, id string, m models.Model) {
	if err := reg.Register(id, m); err != nil {
		log.Fatalf("register %s: %v", id, err)
	}
}
