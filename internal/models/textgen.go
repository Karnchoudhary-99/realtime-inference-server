package models

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// storyWords is the fixed vocabulary the demo generator draws from.
var storyWords = []string{
	"the", "neural", "server", "batches", "requests", "while", "tokens",
	"stream", "through", "open", "connections", "and", "metrics", "tell",
	"the", "story", "of", "every", "inference", "tonight",
}

// StoryGenerator is a deterministic demo StreamingModel: it hashes the
// prompt, then emits a fixed-length token sequence with a small artificial
// per-token delay so the SSE endpoint has something real to stream.
type StoryGenerator struct {
	name       string
	version    string
	numTokens  int
	tokenDelay time.Duration
}

func NewStoryGenerator(name, version string, numTokens int, tokenDelay time.Duration) *StoryGenerator {
	return &StoryGenerator{name: name, version: version, numTokens: numTokens, tokenDelay: tokenDelay}
}

func (m *StoryGenerator) Name() string    { return m.name }
func (m *StoryGenerator) Version() string { return m.version }
func (m *StoryGenerator) InputDim() int   { return -1 }
func (m *StoryGenerator) OutputDim() int  { return -1 }

// Predict treats the first input row as UTF-8 bytes of the prompt and
// returns the generated tokens' vocabulary indices as one output row.
func (m *StoryGenerator) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("model %q: empty input batch", m.name)
	}
	out := make([][]float64, len(inputs))
	for i, row := range inputs {
		prompt := floatsToString(row)
		toks := m.generate(prompt)
		vals := make([]float64, len(toks))
		for j, t := range toks {
			vals[j] = float64(t)
		}
		out[i] = vals
	}
	return out, nil
}

// Stream emits the generated tokens one at a time.
func (m *StoryGenerator) Stream(ctx context.Context, input string, emit func(token string) error) error {
	for _, idx := range m.generate(input) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if m.tokenDelay > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(m.tokenDelay):
			}
		}
		if err := emit(storyWords[idx]); err != nil {
			return err
		}
	}
	return nil
}

// generate deterministically picks numTokens vocabulary indices from the prompt.
func (m *StoryGenerator) generate(prompt string) []int {
	var h uint64 = 14695981039346656037 // FNV-1a offset
	for i := 0; i < len(prompt); i++ {
		h ^= uint64(prompt[i])
		h *= 1099511628211
	}
	rng := newXorshift64(h)
	toks := make([]int, m.numTokens)
	for i := range toks {
		toks[i] = int(rng.next() % uint64(len(storyWords)))
	}
	return toks
}

func floatsToString(row []float64) string {
	var sb strings.Builder
	for _, f := range row {
		if f < 0 || f > 255 {
			continue
		}
		sb.WriteByte(byte(f))
	}
	return sb.String()
}
