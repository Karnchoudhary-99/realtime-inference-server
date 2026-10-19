// Package batcher implements dynamic request batching: individual inference
// requests are collected for up to MaxLatency (or until MaxBatchSize is
// reached) and executed as a single batched Model.Predict call, trading a
// small, bounded latency for much higher throughput — the standard trick of
// production inference servers.
package batcher

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
)

// request is one client's inference call waiting to be batched. One request
// may itself carry a mini-batch of rows; dispatch flattens them and maps
// results back by offset.
type request struct {
	ctx    context.Context
	inputs [][]float64
	result chan result
	done   atomic.Bool // set when the waiter gave up (ctx cancelled)
}

type result struct {
	outputs [][]float64
	err     error
}

// Batcher merges Submit calls into batched Predict invocations.
type Batcher struct {
	model models.Model

	maxBatchSize  int
	maxLatency    time.Duration
	maxPerRequest int

	incoming chan *request
	stopCh   chan struct{}
	wg       sync.WaitGroup

	// stats (atomic)
	batchesRun     atomic.Int64
	requestsServed atomic.Int64
	batchSizeSum   atomic.Int64

	// queueDepth is invoked (if non-nil) whenever the pending queue size changes.
	onQueueDepth func(n int)

	mu      sync.Mutex // guards pending during drain
	pending []*request
}

// New starts the batcher's background loop.
func New(model models.Model, maxBatchSize int, maxLatency time.Duration, maxPerRequest int, onQueueDepth func(n int)) *Batcher {
	b := &Batcher{
		model:         model,
		maxBatchSize:  maxBatchSize,
		maxLatency:    maxLatency,
		maxPerRequest: maxPerRequest,
		incoming:      make(chan *request, maxBatchSize*4),
		stopCh:        make(chan struct{}),
		onQueueDepth:  onQueueDepth,
	}
	b.wg.Add(1)
	go b.loop()
	return b
}

// Submit enqueues one inference request (a mini-batch of rows) and blocks
// until its results are ready or ctx is cancelled. Input slices are not
// retained after Submit returns.
func (b *Batcher) Submit(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	if err := models.CheckBatchSize(b.model.Name(), len(inputs), b.maxPerRequest); err != nil {
		return nil, err
	}
	req := &request{ctx: ctx, inputs: inputs, result: make(chan result, 1)}
	select {
	case <-b.stopCh:
		return nil, errors.New("batcher: shut down")
	case <-ctx.Done():
		return nil, ctx.Err()
	case b.incoming <- req:
	}
	select {
	case <-b.stopCh:
		return nil, errors.New("batcher: shut down")
	case <-ctx.Done():
		req.done.Store(true)
		return nil, ctx.Err()
	case res := <-req.result:
		return res.outputs, res.err
	}
}

// Stats returns lifetime counters.
func (b *Batcher) Stats() (batches int64, requests int64, avgBatch float64) {
	batches = b.batchesRun.Load()
	requests = b.requestsServed.Load()
	if batches > 0 {
		avgBatch = float64(b.batchSizeSum.Load()) / float64(batches)
	}
	return batches, requests, avgBatch
}

// Stop drains no new work and waits for the loop to exit. In-flight batches
// finish; queued requests receive a shutdown error.
func (b *Batcher) Stop() {
	close(b.stopCh)
	b.wg.Wait()
}

// loop is the batching state machine.
func (b *Batcher) loop() {
	defer b.wg.Done()
	for {
		// Idle: wait for the first request or shutdown.
		select {
		case <-b.stopCh:
			b.failAll(errors.New("batcher: shut down"))
			return
		case req := <-b.incoming:
			b.enqueue(req)
		}

		// Collecting: gather until the latency budget expires or the
		// batch is full.
		timer := time.NewTimer(b.maxLatency)
	collect:
		for len(b.pending) < b.maxBatchSize {
			select {
			case <-b.stopCh:
				if !timer.Stop() {
					<-timer.C
				}
				b.failAll(errors.New("batcher: shut down"))
				return
			case req := <-b.incoming:
				b.enqueue(req)
			case <-timer.C:
				break collect
			}
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		b.dispatch()
	}
}

// enqueue adds a request that already passed the cancelled check.
func (b *Batcher) enqueue(req *request) {
	select {
	case <-req.ctx.Done():
		req.done.Store(true)
		return
	default:
	}
	b.mu.Lock()
	b.pending = append(b.pending, req)
	n := len(b.pending)
	b.mu.Unlock()
	if b.onQueueDepth != nil {
		b.onQueueDepth(n)
	}
}

// dispatch runs one batched Predict and fans results back to waiters.
func (b *Batcher) dispatch() {
	b.mu.Lock()
	pending := b.pending
	b.pending = nil
	b.mu.Unlock()
	if b.onQueueDepth != nil {
		b.onQueueDepth(0)
	}
	if len(pending) == 0 {
		return
	}

	// Drop waiters that already gave up; keep ordering for the rest.
	live := pending[:0]
	for _, req := range pending {
		if req.done.Load() {
			continue
		}
		select {
		case <-req.ctx.Done():
			req.done.Store(true)
		default:
			live = append(live, req)
		}
	}
	if len(live) == 0 {
		return
	}

	b.batchesRun.Add(1)
	b.requestsServed.Add(int64(len(live)))
	b.batchSizeSum.Add(int64(len(live)))

	// Flatten the mini-batches into one Predict call, remembering each
	// request's row range so results fan back out correctly.
	type span struct {
		req        *request
		start, end int
	}
	var flat [][]float64
	spans := make([]span, 0, len(live))
	for _, req := range live {
		start := len(flat)
		flat = append(flat, req.inputs...)
		spans = append(spans, span{req: req, start: start, end: len(flat)})
	}
	// The batch runs with a background context: per-request deadlines were
	// already enforced by the waiters, and we must not let one cancelled
	// waiter abort the whole batch.
	outputs, err := b.model.Predict(context.Background(), flat)
	for _, sp := range spans {
		if sp.req.done.Load() {
			continue
		}
		if err != nil {
			sp.req.result <- result{err: err}
			continue
		}
		if sp.end > len(outputs) {
			sp.req.result <- result{err: errors.New("batcher: model returned fewer outputs than inputs")}
			continue
		}
		sp.req.result <- result{outputs: outputs[sp.start:sp.end]}
	}
}

// failAll rejects every queued request, used on shutdown.
func (b *Batcher) failAll(err error) {
	b.mu.Lock()
	pending := b.pending
	b.pending = nil
	b.mu.Unlock()
	for _, req := range pending {
		if !req.done.Load() {
			req.result <- result{err: err}
		}
	}
}
