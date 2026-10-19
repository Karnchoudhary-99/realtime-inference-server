package metrics

import (
	"strings"
	"testing"
)

func TestCounterAndGauge(t *testing.T) {
	r := New()
	c := r.NewCounter("req_total", "requests")
	c.Inc(Labels{"route": "/predict"}, 2)
	c.Inc(Labels{"route": "/predict"}, 3)
	g := r.NewGauge("depth", "queue depth")
	g.Set(Labels{"model": "echo"}, 7)

	var sb strings.Builder
	r.WritePrometheus(&sb)
	out := sb.String()
	if !strings.Contains(out, `req_total{route="/predict"} 5`) {
		t.Fatalf("counter missing or wrong:\n%s", out)
	}
	if !strings.Contains(out, `depth{model="echo"} 7`) {
		t.Fatalf("gauge missing or wrong:\n%s", out)
	}
	if !strings.Contains(out, "# TYPE req_total counter") {
		t.Fatalf("missing TYPE header:\n%s", out)
	}
}

func TestHistogramBucketsCumulative(t *testing.T) {
	r := New()
	h := r.NewHistogram("lat", "latency", []float64{0.1, 1.0})
	h.Observe(Labels{}, 0.05)
	h.Observe(Labels{}, 0.5)
	h.Observe(Labels{}, 5.0)

	var sb strings.Builder
	r.WritePrometheus(&sb)
	out := sb.String()
	for _, want := range []string{
		`lat_bucket{le="0.1"} 1`,
		`lat_bucket{le="1"} 2`,
		`lat_bucket{le="+Inf"} 3`,
		`lat_count 3`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}
