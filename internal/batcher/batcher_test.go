package batcher

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
)

// countingModel wraps EchoModel and counts Predict invocations.
type countingModel struct {
	models.Model
	calls atomic.Int64
	rows  atomic.Int64
}

func (m *countingModel) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	m.calls.Add(1)
	m.rows.Add(int64(len(inputs)))
	return m.Model.Predict(ctx, inputs)
}

func TestMergesConcurrentSubmits(t *testing.T) {
	cm := &countingModel{Model: models.NewEchoModel("echo", "v1")}
	b := New(cm, 64, 50*time.Millisecond, 256, nil)
	defer b.Stop()

	const n = 20
	var wg sync.WaitGroup
	errs := make([]error, n)
	outs := make([][][]float64, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			out, err := b.Submit(context.Background(), [][]float64{{float64(i)}})
			outs[i], errs[i] = out, err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		if len(outs[i]) != 1 || outs[i][0][0] != float64(i) {
			t.Fatalf("submit %d routed wrong output: %v", i, outs[i])
		}
	}
	if got := cm.calls.Load(); got > 3 {
		t.Fatalf("20 concurrent submits ran %d Predict calls, want <= 3", got)
	}
	if got := cm.rows.Load(); got != n {
		t.Fatalf("served %d rows, want %d", got, n)
	}
	batches, reqs, avg := b.Stats()
	if batches == 0 || reqs != n {
		t.Fatalf("stats = (%d batches, %d reqs), want >0 batches and %d reqs", batches, reqs, n)
	}
	if avg <= 1 {
		t.Fatalf("avg batch size = %v, want > 1 (batching did nothing)", avg)
	}
}

func TestMiniBatchFanOut(t *testing.T) {
	b := New(models.NewEchoModel("echo", "v1"), 64, 20*time.Millisecond, 256, nil)
	defer b.Stop()
	in := [][]float64{{1}, {2}, {3}}
	out, err := b.Submit(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 || out[0][0] != 1 || out[2][0] != 3 {
		t.Fatalf("fan-out mismatch: %v", out)
	}
}

func TestPerRequestLimit(t *testing.T) {
	b := New(models.NewEchoModel("echo", "v1"), 64, 20*time.Millisecond, 2, nil)
	defer b.Stop()
	if _, err := b.Submit(context.Background(), [][]float64{{1}, {2}, {3}}); err == nil {
		t.Fatal("expected per-request limit error, got nil")
	}
}

func TestCancelledSubmit(t *testing.T) {
	b := New(models.NewEchoModel("echo", "v1"), 64, 20*time.Millisecond, 256, nil)
	defer b.Stop()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := b.Submit(ctx, [][]float64{{1}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

type failingModel struct{ models.Model }

func (failingModel) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	return nil, errors.New("boom")
}

func TestErrorPropagatesToAllWaiters(t *testing.T) {
	b := New(failingModel{models.NewEchoModel("echo", "v1")}, 64, 30*time.Millisecond, 256, nil)
	defer b.Stop()
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := b.Submit(context.Background(), [][]float64{{1}}); err == nil {
				t.Error("expected error, got nil")
			}
		}()
	}
	wg.Wait()
}

func TestQueueDepthCallback(t *testing.T) {
	var mu sync.Mutex
	var depths []int
	b := New(models.NewEchoModel("echo", "v1"), 64, 20*time.Millisecond, 256,
		func(n int) { mu.Lock(); depths = append(depths, n); mu.Unlock() })
	defer b.Stop()
	if _, err := b.Submit(context.Background(), [][]float64{{1}}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond) // let dispatch run
	mu.Lock()
	defer mu.Unlock()
	sawNonZero, sawZero := false, false
	for _, d := range depths {
		if d > 0 {
			sawNonZero = true
		}
		if d == 0 {
			sawZero = true
		}
	}
	if !sawNonZero || !sawZero {
		t.Fatalf("depth callback never saw both states: %v", depths)
	}
}
