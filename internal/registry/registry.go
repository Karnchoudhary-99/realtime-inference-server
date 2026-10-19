// Package registry is a concurrency-safe store of named models. It supports
// registration, lookup, listing, removal, and atomic hot-swap of a model's
// implementation without dropping in-flight requests.
package registry

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
)

// Entry describes one registered model.
type Entry struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Version      string    `json:"version"`
	InputDim     int       `json:"input_dim"`
	OutputDim    int       `json:"output_dim"`
	Streaming    bool      `json:"streaming"`
	RegisteredAt time.Time `json:"registered_at"`
}

// Registry maps model IDs to implementations.
type Registry struct {
	mu     sync.RWMutex
	models map[string]models.Model
	times  map[string]time.Time
}

// New returns an empty Registry.
func New() *Registry {
	return &Registry{models: make(map[string]models.Model), times: make(map[string]time.Time)}
}

// Register adds a model under id. It fails if id is taken; use HotSwap to
// replace an existing implementation.
func (r *Registry) Register(id string, m models.Model) error {
	if id == "" {
		return fmt.Errorf("registry: empty model id")
	}
	if m == nil {
		return fmt.Errorf("registry: nil model for id %q", id)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[id]; ok {
		return fmt.Errorf("registry: model %q already registered (use HotSwap)", id)
	}
	r.models[id] = m
	r.times[id] = time.Now().UTC()
	return nil
}

// HotSwap atomically replaces the implementation behind id. Readers holding
// the old model keep using it; new lookups see the replacement.
func (r *Registry) HotSwap(id string, m models.Model) error {
	if m == nil {
		return fmt.Errorf("registry: nil model for id %q", id)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[id]; !ok {
		return fmt.Errorf("registry: model %q not found", id)
	}
	r.models[id] = m
	r.times[id] = time.Now().UTC()
	return nil
}

// Unregister removes id. It fails if id is unknown.
func (r *Registry) Unregister(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.models[id]; !ok {
		return fmt.Errorf("registry: model %q not found", id)
	}
	delete(r.models, id)
	delete(r.times, id)
	return nil
}

// Get returns the model registered under id.
func (r *Registry) Get(id string) (models.Model, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[id]
	return m, ok
}

// List returns all registered models sorted by id.
func (r *Registry) List() []Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entries := make([]Entry, 0, len(r.models))
	for id, m := range r.models {
		_, streaming := m.(models.StreamingModel)
		entries = append(entries, Entry{
			ID:           id,
			Name:         m.Name(),
			Version:      m.Version(),
			InputDim:     m.InputDim(),
			OutputDim:    m.OutputDim(),
			Streaming:    streaming,
			RegisteredAt: r.times[id],
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].ID < entries[j].ID })
	return entries
}

// Len returns the number of registered models.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}
