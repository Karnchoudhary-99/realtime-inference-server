package models

import (
	"context"
	"math"
	"testing"
)

func TestSoftmaxSumsToOne(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	softmax(x)
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("softmax sum = %v, want 1", sum)
	}
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Fatalf("softmax not order-preserving: %v", x)
	}
}

func TestMatVec(t *testing.T) {
	// y = [[1,2],[3,4]]·[5,6] + [7,8] = [24, 47]
	w := []float64{1, 2, 3, 4}
	b := []float64{7, 8}
	y := matVec(w, b, 2, 2, []float64{5, 6})
	if y[0] != 24 || y[1] != 47 {
		t.Fatalf("matVec = %v, want [24 47]", y)
	}
}

func TestMLPClassifierOutput(t *testing.T) {
	m := NewMLPClassifier("test", "v1", 4, 16, 3, 42)
	outs, err := m.Predict(context.Background(), [][]float64{{5.1, 3.5, 1.4, 0.2}, {0, 0, 0, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 2 || len(outs[0]) != 3 {
		t.Fatalf("bad output shape: %v", outs)
	}
	for i, row := range outs {
		sum := 0.0
		for _, v := range row {
			if v < 0 || v > 1 {
				t.Fatalf("row %d not a distribution: %v", i, row)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Fatalf("row %d sums to %v, want 1", i, sum)
		}
	}
}

func TestMLPDeterministic(t *testing.T) {
	a := NewMLPClassifier("x", "v1", 4, 8, 2, 99)
	b := NewMLPClassifier("x", "v1", 4, 8, 2, 99)
	in := [][]float64{{1, 2, 3, 4}}
	oa, _ := a.Predict(context.Background(), in)
	ob, _ := b.Predict(context.Background(), in)
	for i := range oa[0] {
		if oa[0][i] != ob[0][i] {
			t.Fatalf("same seed diverged: %v vs %v", oa, ob)
		}
	}
}

func TestCheckInputsRejectsBadDim(t *testing.T) {
	m := NewLinearModel("lin", "v1", 3, 1, 1)
	if _, err := m.Predict(context.Background(), [][]float64{{1, 2}}); err == nil {
		t.Fatal("expected dim error, got nil")
	}
	if _, err := m.Predict(context.Background(), nil); err == nil {
		t.Fatal("expected empty-batch error, got nil")
	}
}

func TestEchoRoundTrip(t *testing.T) {
	m := NewEchoModel("echo", "v1")
	in := [][]float64{{1, 2, 3}, {4, 5}}
	out, err := m.Predict(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	for i := range in {
		for j := range in[i] {
			if out[i][j] != in[i][j] {
				t.Fatalf("echo mismatch at [%d][%d]", i, j)
			}
		}
	}
}

func TestStoryGeneratorDeterministic(t *testing.T) {
	a := NewStoryGenerator("s", "v1", 10, 0)
	b := NewStoryGenerator("s", "v1", 10, 0)
	var ta, tb []string
	_ = a.Stream(context.Background(), "hello", func(tok string) error { ta = append(ta, tok); return nil })
	_ = b.Stream(context.Background(), "hello", func(tok string) error { tb = append(tb, tok); return nil })
	if len(ta) != 10 || len(tb) != 10 {
		t.Fatalf("want 10 tokens, got %d/%d", len(ta), len(tb))
	}
	for i := range ta {
		if ta[i] != tb[i] {
			t.Fatalf("nondeterministic tokens at %d: %q vs %q", i, ta[i], tb[i])
		}
	}
	var tc []string
	_ = a.Stream(context.Background(), "different", func(tok string) error { tc = append(tc, tok); return nil })
	same := true
	for i := range ta {
		if ta[i] != tc[i] {
			same = false
		}
	}
	if same {
		t.Fatal("different prompts produced identical sequences")
	}
}
