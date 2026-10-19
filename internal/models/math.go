package models

import "math"

// matVec computes y = W·x + b where W is out×in row-major and b has length out.
func matVec(w []float64, b []float64, out, in int, x []float64) []float64 {
	y := make([]float64, out)
	for o := 0; o < out; o++ {
		s := b[o]
		row := w[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			s += row[i] * x[i]
		}
		y[o] = s
	}
	return y
}

// relu applies max(0, x) elementwise.
func relu(x []float64) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// softmax normalizes x in place into a probability distribution.
func softmax(x []float64) {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// xorshift64 is a tiny deterministic PRNG so built-in model weights are
// stable across runs and testable without a seed file.
type xorshift64 struct{ s uint64 }

func newXorshift64(seed uint64) *xorshift64 {
	if seed == 0 {
		seed = 0x9E3779B97F4A7C15
	}
	return &xorshift64{s: seed}
}

func (r *xorshift64) next() uint64 {
	x := r.s
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	r.s = x
	return x
}

// uniform returns a value in [lo, hi).
func (r *xorshift64) uniform(lo, hi float64) float64 {
	return lo + (hi-lo)*(float64(r.next()>>11)/float64(1<<53))
}

// xavierInit fills w (out×in row-major) with Xavier-uniform values.
func xavierInit(rng *xorshift64, w []float64, out, in int) {
	limit := math.Sqrt(6.0 / float64(in+out))
	for i := range w {
		w[i] = rng.uniform(-limit, limit)
	}
}
