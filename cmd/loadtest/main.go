// Command loadtest hammers POST /v1/models/{id}/predict with concurrent
// workers and reports throughput plus latency percentiles. It is the tool
// behind the benchmark numbers in the README.
//
// Usage:
//
//	go run ./cmd/loadtest -addr http://localhost:8080 -model iris-mlp -dim 4 -n 2000 -c 32
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"sort"
	"sync"
	"time"
)

func main() {
	var (
		addr    = flag.String("addr", "http://localhost:8080", "server base URL")
		model   = flag.String("model", "iris-mlp", "model id to hit")
		dim     = flag.Int("dim", 4, "input feature dimension")
		rows    = flag.Int("rows", 4, "rows per request")
		n       = flag.Int("n", 2000, "total requests")
		workers = flag.Int("c", 32, "concurrent workers")
	)
	flag.Parse()

	payload := func() []byte {
		inputs := make([][]float64, *rows)
		for i := range inputs {
			row := make([]float64, *dim)
			for j := range row {
				row[j] = rand.NormFloat64()
			}
			inputs[i] = row
		}
		body, _ := json.Marshal(map[string]any{"inputs": inputs})
		return body
	}()

	url := *addr + "/v1/models/" + *model + "/predict"
	client := &http.Client{Timeout: 30 * time.Second}

	latencies := make([]time.Duration, 0, *n)
	var mu sync.Mutex
	var failed int64
	jobs := make(chan struct{}, *n)
	for i := 0; i < *n; i++ {
		jobs <- struct{}{}
	}
	close(jobs)

	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range jobs {
				t0 := time.Now()
				resp, err := client.Post(url, "application/json", bytes.NewReader(payload))
				if err != nil {
					mu.Lock()
					failed++
					mu.Unlock()
					continue
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode != http.StatusOK {
					mu.Lock()
					failed++
					mu.Unlock()
					continue
				}
				mu.Lock()
				latencies = append(latencies, time.Since(t0))
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	pct := func(p float64) time.Duration {
		if len(latencies) == 0 {
			return 0
		}
		return latencies[int(p*float64(len(latencies)-1))]
	}
	ok := len(latencies)
	fmt.Printf("model=%s workers=%d requests=%d failed=%d\n", *model, *workers, *n, failed)
	fmt.Printf("throughput: %.0f req/s (%.0f rows/s)\n", float64(ok)/elapsed.Seconds(), float64(ok**rows)/elapsed.Seconds())
	fmt.Printf("latency: p50=%v p95=%v p99=%v max=%v\n", pct(0.50), pct(0.95), pct(0.99), pct(1.0))
	if failed > 0 {
		log.Fatalf("%d requests failed", failed)
	}
}
