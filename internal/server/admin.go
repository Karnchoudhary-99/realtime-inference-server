// Admin API: register, hot-swap and delete models at runtime.
//
//	POST   /v1/admin/models       register a new model from a ModelSpec
//	PUT    /v1/admin/models/{id}  hot-swap the implementation behind id
//	DELETE /v1/admin/models/{id}  unregister id
//
// Admin routes are protected by a shared bearer token when the server is
// started with --admin-token (or ADMIN_TOKEN); without a token they are open,
// which is only appropriate for local development.
package server

import (
	"encoding/json"
	"net/http"
	"regexp"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/metrics"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/registry"
)

// modelIDPattern keeps model ids URL-safe.
var modelIDPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`)

// adminRegisterRequest is the body of POST /v1/admin/models: an id plus the
// spec of the model to instantiate. adminHotswapRequest (PUT) carries the
// same spec fields with the id taken from the URL.
type adminRegisterRequest struct {
	ID string `json:"id"`
	models.ModelSpec
}

type adminHotswapRequest struct {
	models.ModelSpec
}

// adminGuard wraps an admin handler with bearer-token auth when a token is
// configured. It reads the token per request so a rotated Config takes
// effect without a restart.
func (s *Server) adminGuard(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if token := s.cfg.AdminToken; token != "" {
			if r.Header.Get("Authorization") != "Bearer "+token {
				s.adminOps.Inc(metrics.Labels{"op": "auth", "result": "unauthorized"}, 1)
				writeError(w, http.StatusUnauthorized, "admin API requires a bearer token")
				return
			}
		}
		next(w, r)
	}
}

func (s *Server) handleAdminRegister(w http.ResponseWriter, r *http.Request) {
	var req adminRegisterRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "register", "result": "bad_request"}, 1)
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if !modelIDPattern.MatchString(req.ID) {
		s.adminOps.Inc(metrics.Labels{"op": "register", "result": "bad_id"}, 1)
		writeError(w, http.StatusBadRequest, "id must match [a-zA-Z0-9_-], max 64 chars")
		return
	}
	mdl, err := models.BuildModel(req.ID, req.ModelSpec)
	if err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "register", "result": "bad_spec"}, 1)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := s.reg.Register(req.ID, mdl); err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "register", "result": "conflict"}, 1)
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	s.adminOps.Inc(metrics.Labels{"op": "register", "result": "ok"}, 1)
	writeJSON(w, http.StatusCreated, map[string]any{"model": findEntry(s.reg, req.ID)})
}

// handleAdminHotswap replaces the implementation behind id and points the
// live batcher at the new model, without dropping in-flight requests.
func (s *Server) handleAdminHotswap(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req adminHotswapRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "hotswap", "result": "bad_request"}, 1)
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	mdl, err := models.BuildModel(id, req.ModelSpec)
	if err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "hotswap", "result": "bad_spec"}, 1)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := s.hotswapModel(id, mdl); err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "hotswap", "result": "not_found"}, 1)
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	s.adminOps.Inc(metrics.Labels{"op": "hotswap", "result": "ok"}, 1)
	writeJSON(w, http.StatusOK, map[string]any{"model": findEntry(s.reg, id)})
}

// handleAdminDelete unregisters id and stops its batcher so no goroutine
// or queue lingers behind.
func (s *Server) handleAdminDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := s.reg.Unregister(id); err != nil {
		s.adminOps.Inc(metrics.Labels{"op": "delete", "result": "not_found"}, 1)
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	s.removeBatcher(id)
	s.adminOps.Inc(metrics.Labels{"op": "delete", "result": "ok"}, 1)
	w.WriteHeader(http.StatusNoContent)
}

// hotswapModel replaces the registry entry and, when a batcher already
// exists for id, points it at the new implementation.
func (s *Server) hotswapModel(id string, mdl models.Model) error {
	if err := s.reg.HotSwap(id, mdl); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if b, ok := s.batchers[id]; ok {
		b.SwapModel(mdl)
	}
	return nil
}

// removeBatcher stops and drops the batcher for id, if any.
func (s *Server) removeBatcher(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if b, ok := s.batchers[id]; ok {
		b.Stop()
		delete(s.batchers, id)
	}
}

// findEntry returns the registry entry for id; the id is known to exist.
func findEntry(reg *registry.Registry, id string) registry.Entry {
	for _, e := range reg.List() {
		if e.ID == id {
			return e
		}
	}
	return registry.Entry{ID: id}
}
