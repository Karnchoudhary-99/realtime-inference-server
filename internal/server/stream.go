package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/metrics"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
)

// streamEvent is one SSE data frame.
type streamEvent struct {
	Token string `json:"token"`
	Index int    `json:"index"`
}

// GET /v1/models/{id}/stream?prompt=... — server-sent events, one frame per
// token, terminated by a literal "[DONE]" frame. Only models implementing
// models.StreamingModel can be streamed.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	mdl, ok := s.reg.Get(id)
	if !ok {
		writeError(w, http.StatusNotFound, "unknown model "+id)
		return
	}
	sm, ok := mdl.(models.StreamingModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model "+id+" does not support streaming")
		return
	}
	prompt := r.URL.Query().Get("prompt")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	s.inferRequests.Inc(metrics.Labels{"model": id}, 1)
	requestID := s.reqID.Add(1)
	fmt.Fprintf(w, ": connected request_id=%d\n\n", requestID)
	flusher.Flush()

	idx := 0
	err := sm.Stream(r.Context(), prompt, func(token string) error {
		payload, _ := json.Marshal(streamEvent{Token: token, Index: idx})
		if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
			return err
		}
		flusher.Flush()
		idx++
		return nil
	})
	if err != nil {
		// Client disconnects surface as context errors; anything else is
		// reported as a terminal error frame.
		if r.Context().Err() == nil {
			s.inferErrors.Inc(metrics.Labels{"model": id, "reason": "stream_error"}, 1)
			payload, _ := json.Marshal(map[string]string{"error": err.Error()})
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
			flusher.Flush()
		}
		return
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
