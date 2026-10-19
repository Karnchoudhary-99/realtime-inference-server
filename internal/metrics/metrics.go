// Package metrics is a tiny Prometheus-compatible metrics registry:
// counters, gauges and histograms with text exposition, stdlib only.
package metrics

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
)

// Labels is a set of key=value label pairs.
type Labels map[string]string

func (l Labels) key() string {
	if len(l) == 0 {
		return ""
	}
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&sb, "%s=%q,", k, l[k])
	}
	return sb.String()
}

// Counter is a monotonically increasing counter, partitioned by labels.
type Counter struct {
	mu     sync.Mutex
	name   string
	help   string
	values map[string]float64
	labels map[string]Labels
}

// Gauge is a settable value, partitioned by labels.
type Gauge struct {
	mu     sync.Mutex
	name   string
	help   string
	values map[string]float64
	labels map[string]Labels
}

// Histogram counts observations into cumulative buckets, partitioned by labels.
type Histogram struct {
	mu      sync.Mutex
	name    string
	help    string
	buckets []float64
	counts  map[string][]uint64 // label key -> cumulative bucket counts
	sums    map[string]float64
	totals  map[string]uint64
	labels  map[string]Labels
}

// Registry owns the metric families.
type Registry struct {
	mu         sync.Mutex
	counters   []*Counter
	gauges     []*Gauge
	histograms []*Histogram
}

// New returns an empty Registry.
func New() *Registry { return &Registry{} }

// NewCounter registers a counter family.
func (r *Registry) NewCounter(name, help string) *Counter {
	c := &Counter{name: name, help: help, values: map[string]float64{}, labels: map[string]Labels{}}
	r.mu.Lock()
	r.counters = append(r.counters, c)
	r.mu.Unlock()
	return c
}

// NewGauge registers a gauge family.
func (r *Registry) NewGauge(name, help string) *Gauge {
	g := &Gauge{name: name, help: help, values: map[string]float64{}, labels: map[string]Labels{}}
	r.mu.Lock()
	r.gauges = append(r.gauges, g)
	r.mu.Unlock()
	return g
}

// NewHistogram registers a histogram family with the given bucket bounds.
func (r *Registry) NewHistogram(name, help string, buckets []float64) *Histogram {
	h := &Histogram{
		name: name, help: help, buckets: buckets,
		counts: map[string][]uint64{}, sums: map[string]float64{},
		totals: map[string]uint64{}, labels: map[string]Labels{},
	}
	r.mu.Lock()
	r.histograms = append(r.histograms, h)
	r.mu.Unlock()
	return h
}

// Inc adds delta to the counter series for labels.
func (c *Counter) Inc(labels Labels, delta float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	k := labels.key()
	c.values[k] += delta
	c.labels[k] = labels
}

// Set assigns the gauge series for labels.
func (g *Gauge) Set(labels Labels, v float64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	k := labels.key()
	g.values[k] = v
	g.labels[k] = labels
}

// Add adds delta to the gauge series for labels.
func (g *Gauge) Add(labels Labels, delta float64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	k := labels.key()
	g.values[k] += delta
	g.labels[k] = labels
}

// Observe records one sample in the histogram series for labels.
func (h *Histogram) Observe(labels Labels, v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	k := labels.key()
	counts, ok := h.counts[k]
	if !ok {
		counts = make([]uint64, len(h.buckets))
		h.counts[k] = counts
		h.labels[k] = labels
	}
	for i, b := range h.buckets {
		if v <= b {
			counts[i]++
		}
	}
	h.sums[k] += v
	h.totals[k]++
}

// DefaultLatencyBuckets are second-granularity buckets for inference latency.
func DefaultLatencyBuckets() []float64 {
	return []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}
}

func formatLabels(l Labels) string {
	if len(l) == 0 {
		return ""
	}
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%q", k, l[k]))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

// WritePrometheus renders the registry in Prometheus text exposition format.
func (r *Registry) WritePrometheus(w io.Writer) {
	r.mu.Lock()
	counters := append([]*Counter(nil), r.counters...)
	gauges := append([]*Gauge(nil), r.gauges...)
	histograms := append([]*Histogram(nil), r.histograms...)
	r.mu.Unlock()

	for _, c := range counters {
		c.mu.Lock()
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n", c.name, c.help, c.name)
		keys := sortedKeys(c.values)
		for _, k := range keys {
			fmt.Fprintf(w, "%s%s %v\n", c.name, formatLabels(c.labels[k]), c.values[k])
		}
		c.mu.Unlock()
	}
	for _, g := range gauges {
		g.mu.Lock()
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n", g.name, g.help, g.name)
		keys := sortedKeys(g.values)
		for _, k := range keys {
			fmt.Fprintf(w, "%s%s %v\n", g.name, formatLabels(g.labels[k]), g.values[k])
		}
		g.mu.Unlock()
	}
	for _, h := range histograms {
		h.mu.Lock()
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s histogram\n", h.name, h.help, h.name)
		keys := sortedKeysU64(h.totals)
		for _, k := range keys {
			counts := h.counts[k]
			for i, b := range h.buckets {
				bl := appendLabels(h.labels[k], "le", fmt.Sprintf("%v", b))
				fmt.Fprintf(w, "%s_bucket%s %d\n", h.name, formatLabels(bl), counts[i])
			}
			bl := appendLabels(h.labels[k], "le", "+Inf")
			fmt.Fprintf(w, "%s_bucket%s %d\n", h.name, formatLabels(bl), h.totals[k])
			fmt.Fprintf(w, "%s_sum%s %v\n", h.name, formatLabels(h.labels[k]), h.sums[k])
			fmt.Fprintf(w, "%s_count%s %d\n", h.name, formatLabels(h.labels[k]), h.totals[k])
		}
		h.mu.Unlock()
	}
}

func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func sortedKeysU64(m map[string]uint64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func appendLabels(l Labels, k, v string) Labels {
	out := make(Labels, len(l)+1)
	for key, val := range l {
		out[key] = val
	}
	out[k] = v
	return out
}
