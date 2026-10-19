package models

import (
	"context"
	"fmt"
)

// LinearModel is a single affine layer y = W·x + b, useful as a
// regression-style demo model with predictable math.
type LinearModel struct {
	name    string
	version string
	in, out int
	w       []float64
	b       []float64
}

// NewLinearModel builds a deterministic linear model from a seed.
func NewLinearModel(name, version string, in, out int, seed uint64) *LinearModel {
	rng := newXorshift64(seed)
	w := make([]float64, out*in)
	xavierInit(rng, w, out, in)
	b := make([]float64, out)
	for i := range b {
		b[i] = rng.uniform(-0.1, 0.1)
	}
	return &LinearModel{name: name, version: version, in: in, out: out, w: w, b: b}
}

func (m *LinearModel) Name() string    { return m.name }
func (m *LinearModel) Version() string { return m.version }
func (m *LinearModel) InputDim() int   { return m.in }
func (m *LinearModel) OutputDim() int  { return m.out }

func (m *LinearModel) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	if err := CheckInputs(m.name, m.in, inputs); err != nil {
		return nil, err
	}
	out := make([][]float64, len(inputs))
	for i, row := range inputs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		out[i] = matVec(m.w, m.b, m.out, m.in, row)
	}
	return out, nil
}

// MLPClassifier is a two-layer feed-forward network with a softmax head,
// e.g. a 4-feature classifier demo. Weights are deterministic per seed.
type MLPClassifier struct {
	name    string
	version string
	in      int
	hidden  int
	classes int
	w1      []float64
	b1      []float64
	w2      []float64
	b2      []float64
}

// NewMLPClassifier builds a deterministic in→hidden→classes network.
func NewMLPClassifier(name, version string, in, hidden, classes int, seed uint64) *MLPClassifier {
	rng := newXorshift64(seed)
	w1 := make([]float64, hidden*in)
	xavierInit(rng, w1, hidden, in)
	b1 := make([]float64, hidden)
	w2 := make([]float64, classes*hidden)
	xavierInit(rng, w2, classes, hidden)
	b2 := make([]float64, classes)
	return &MLPClassifier{
		name: name, version: version, in: in, hidden: hidden, classes: classes,
		w1: w1, b1: b1, w2: w2, b2: b2,
	}
}

func (m *MLPClassifier) Name() string    { return m.name }
func (m *MLPClassifier) Version() string { return m.version }
func (m *MLPClassifier) InputDim() int   { return m.in }
func (m *MLPClassifier) OutputDim() int  { return m.classes }

func (m *MLPClassifier) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	if err := CheckInputs(m.name, m.in, inputs); err != nil {
		return nil, err
	}
	out := make([][]float64, len(inputs))
	for i, row := range inputs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		h := matVec(m.w1, m.b1, m.hidden, m.in, row)
		relu(h)
		logits := matVec(m.w2, m.b2, m.classes, m.hidden, h)
		softmax(logits)
		out[i] = logits
	}
	return out, nil
}

// EchoModel returns its inputs unchanged. It exists to make batching,
// latency and throughput observable without any model math in the way.
type EchoModel struct {
	name    string
	version string
}

func NewEchoModel(name, version string) *EchoModel { return &EchoModel{name: name, version: version} }

func (m *EchoModel) Name() string    { return m.name }
func (m *EchoModel) Version() string { return m.version }
func (m *EchoModel) InputDim() int   { return -1 }
func (m *EchoModel) OutputDim() int  { return -1 }

func (m *EchoModel) Predict(ctx context.Context, inputs [][]float64) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("model %q: empty input batch", m.name)
	}
	out := make([][]float64, len(inputs))
	for i, row := range inputs {
		cp := make([]float64, len(row))
		copy(cp, row)
		out[i] = cp
	}
	return out, nil
}
