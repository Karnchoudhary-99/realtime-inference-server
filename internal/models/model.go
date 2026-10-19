// Package models defines the Model interface served by the inference server
// and the built-in models that ship with it. Models are stateless and safe
// for concurrent use; the server batches individual requests before calling
// Predict so that each invocation is one batched forward pass.
package models

import (
	"context"
	"fmt"
)

// Model is a numeric inference model. Inputs and outputs are batches of
// feature vectors; the server merges many client requests into a single
// Predict call for throughput.
type Model interface {
	// Name is the model identifier used in URLs, e.g. "iris-mlp".
	Name() string
	// Version is an opaque version label, e.g. "v1".
	Version() string
	// InputDim is the expected feature dimension; -1 accepts any width.
	InputDim() int
	// OutputDim is the width of each output vector.
	OutputDim() int
	// Predict runs one batched forward pass. len(inputs) > 0 always;
	// the returned slice must have the same length as inputs.
	Predict(ctx context.Context, inputs [][]float64) ([][]float64, error)
}

// StreamingModel is a Model that can emit incremental output tokens,
// used by the SSE streaming endpoint for token-style responses.
type StreamingModel interface {
	Model
	// Stream calls emit once per generated token and blocks until the full
	// sequence is produced or ctx is cancelled. emit must not be retained.
	Stream(ctx context.Context, input string, emit func(token string) error) error
}

// CheckInputs validates a batch against the model's expected input width.
func CheckInputs(name string, wantDim int, inputs [][]float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("model %q: empty input batch", name)
	}
	if wantDim < 0 {
		return nil
	}
	for i, row := range inputs {
		if len(row) != wantDim {
			return fmt.Errorf("model %q: input %d has dim %d, want %d", name, i, len(row), wantDim)
		}
	}
	return nil
}

// CheckBatchSize rejects requests that would blow up a single forward pass.
func CheckBatchSize(name string, n, max int) error {
	if n > max {
		return fmt.Errorf("model %q: batch of %d exceeds per-request limit %d", name, n, max)
	}
	return nil
}
