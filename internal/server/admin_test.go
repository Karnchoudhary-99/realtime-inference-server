package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/metrics"
	"github.com/Karnchoudhary-99/realtime-inference-server/internal/registry"
)

// adminTestServer builds a server with no admin token (open admin API).
func adminTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := New(registry.New(), metrics.New(), DefaultConfig())
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Shutdown() })
	return ts
}

func adminDo(t *testing.T, ts *httptest.Server, method, path, body string) *http.Response {
	t.Helper()
	var req *http.Request
	var err error
	if body == "" {
		req, err = http.NewRequest(method, ts.URL+path, nil)
	} else {
		req, err = http.NewRequest(method, ts.URL+path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
	}
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestAdminRegisterPredictDelete(t *testing.T) {
	ts := adminTestServer(t)

	// Register a new MLP.
	resp := adminDo(t, ts, "POST", "/v1/admin/models",
		`{"id":"iris-v2","type":"mlp","version":"v2","input_dim":4,"output_dim":3,"seed":7}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("register: status %d, want 201", resp.StatusCode)
	}
	var regBody struct {
		Model registry.Entry `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&regBody); err != nil {
		t.Fatal(err)
	}
	if regBody.Model.Version != "v2" || regBody.Model.InputDim != 4 {
		t.Fatalf("registered entry = %+v", regBody.Model)
	}

	// The new model serves predictions.
	predict := adminDo(t, ts, "POST", "/v1/models/iris-v2/predict",
		`{"inputs": [[5.1,3.5,1.4,0.2]]}`)
	if predict.StatusCode != http.StatusOK {
		t.Fatalf("predict on new model: status %d, want 200", predict.StatusCode)
	}
	predict.Body.Close()

	// It shows up in the model list.
	listResp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var list struct {
		Models []registry.Entry `json:"models"`
	}
	if err := json.NewDecoder(listResp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	listResp.Body.Close()
	if len(list.Models) != 1 || list.Models[0].ID != "iris-v2" {
		t.Fatalf("list = %+v", list.Models)
	}

	// Delete it.
	del := adminDo(t, ts, "DELETE", "/v1/admin/models/iris-v2", "")
	if del.StatusCode != http.StatusNoContent {
		t.Fatalf("delete: status %d, want 204", del.StatusCode)
	}
	del.Body.Close()

	// Predicting now 404s.
	gone := adminDo(t, ts, "POST", "/v1/models/iris-v2/predict", `{"inputs": [[1,2,3,4]]}`)
	if gone.StatusCode != http.StatusNotFound {
		t.Fatalf("predict after delete: status %d, want 404", gone.StatusCode)
	}
	gone.Body.Close()

	// Deleting again 404s.
	del2 := adminDo(t, ts, "DELETE", "/v1/admin/models/iris-v2", "")
	if del2.StatusCode != http.StatusNotFound {
		t.Fatalf("second delete: status %d, want 404", del2.StatusCode)
	}
	del2.Body.Close()
}

func TestAdminRegisterConflictsAndValidation(t *testing.T) {
	ts := adminTestServer(t)

	if resp := adminDo(t, ts, "POST", "/v1/admin/models", `{"id":"m1","type":"echo"}`); resp.StatusCode != http.StatusCreated {
		t.Fatalf("first register: status %d, want 201", resp.StatusCode)
	} else {
		resp.Body.Close()
	}
	if resp := adminDo(t, ts, "POST", "/v1/admin/models", `{"id":"m1","type":"echo"}`); resp.StatusCode != http.StatusConflict {
		t.Fatalf("duplicate register: status %d, want 409", resp.StatusCode)
	} else {
		resp.Body.Close()
	}

	for name, body := range map[string]string{
		"bad json":       `{oops`,
		"bad id":         `{"id":"not a model!","type":"echo"}`,
		"empty id":       `{"id":"","type":"echo"}`,
		"unknown type":   `{"id":"m2","type":"transformer"}`,
		"bad spec":       `{"id":"m2","type":"mlp","input_dim":0,"output_dim":3}`,
		"storygen spec":  `{"id":"m2","type":"storygen","num_tokens":-2}`,
		"streaming spec": `{"id":"story","type":"storygen","num_tokens":5,"token_delay_ms":0}`,
	} {
		want := http.StatusBadRequest
		if name == "streaming spec" {
			want = http.StatusCreated
		}
		resp := adminDo(t, ts, "POST", "/v1/admin/models", body)
		if resp.StatusCode != want {
			t.Errorf("%s: status %d, want %d", name, resp.StatusCode, want)
		}
		resp.Body.Close()
	}

	// The streaming model registered above supports the SSE endpoint.
	streamResp, err := http.Get(ts.URL + "/v1/models/story/stream?prompt=hi")
	if err != nil {
		t.Fatal(err)
	}
	streamResp.Body.Close()
	if streamResp.StatusCode != http.StatusOK {
		t.Fatalf("stream on admin-registered storygen: status %d", streamResp.StatusCode)
	}
}

func TestAdminHotswap(t *testing.T) {
	ts := adminTestServer(t)

	reg := adminDo(t, ts, "POST", "/v1/admin/models",
		`{"id":"clf","type":"mlp","version":"v1","input_dim":4,"output_dim":3,"seed":1}`)
	reg.Body.Close()
	if reg.StatusCode != http.StatusCreated {
		t.Fatalf("register: status %d", reg.StatusCode)
	}

	// Predict once (this also spins up the per-model batcher).
	before := adminDo(t, ts, "POST", "/v1/models/clf/predict", `{"inputs": [[1,2,3,4]]}`)
	var beforeBody struct {
		Outputs [][]float64 `json:"outputs"`
	}
	if err := json.NewDecoder(before.Body).Decode(&beforeBody); err != nil {
		t.Fatal(err)
	}
	before.Body.Close()

	// Hot-swap to a different seed.
	swap := adminDo(t, ts, "PUT", "/v1/admin/models/clf",
		`{"type":"mlp","version":"v2","input_dim":4,"output_dim":3,"seed":99}`)
	defer swap.Body.Close()
	if swap.StatusCode != http.StatusOK {
		t.Fatalf("hotswap: status %d, want 200", swap.StatusCode)
	}
	var swapBody struct {
		Model registry.Entry `json:"model"`
	}
	if err := json.NewDecoder(swap.Body).Decode(&swapBody); err != nil {
		t.Fatal(err)
	}
	if swapBody.Model.Version != "v2" {
		t.Fatalf("after hotswap version = %q, want v2", swapBody.Model.Version)
	}

	// Predictions still flow through the SAME batcher and reflect the new weights.
	after := adminDo(t, ts, "POST", "/v1/models/clf/predict", `{"inputs": [[1,2,3,4]]}`)
	var afterBody struct {
		Outputs [][]float64 `json:"outputs"`
	}
	if err := json.NewDecoder(after.Body).Decode(&afterBody); err != nil {
		t.Fatal(err)
	}
	after.Body.Close()
	if len(afterBody.Outputs) != 1 || len(afterBody.Outputs[0]) != 3 {
		t.Fatalf("after hotswap outputs = %v", afterBody.Outputs)
	}
	same := true
	for i := range beforeBody.Outputs[0] {
		if beforeBody.Outputs[0][i] != afterBody.Outputs[0][i] {
			same = false
		}
	}
	if same {
		t.Fatal("hotswap: outputs identical after seed change, batcher still on old model?")
	}

	// Hot-swapping an unknown id 404s.
	missing := adminDo(t, ts, "PUT", "/v1/admin/models/nope", `{"type":"echo"}`)
	if missing.StatusCode != http.StatusNotFound {
		t.Fatalf("hotswap unknown: status %d, want 404", missing.StatusCode)
	}
	missing.Body.Close()
}

func TestAdminAuth(t *testing.T) {
	cfg := DefaultConfig()
	cfg.AdminToken = "s3cret"
	s := New(registry.New(), metrics.New(), cfg)
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Shutdown() })

	req, _ := http.NewRequest("POST", ts.URL+"/v1/admin/models", bytes.NewReader([]byte(`{"id":"x","type":"echo"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token: status %d, want 401", resp.StatusCode)
	}

	req, _ = http.NewRequest("POST", ts.URL+"/v1/admin/models", bytes.NewReader([]byte(`{"id":"x","type":"echo"}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer wrong")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong token: status %d, want 401", resp.StatusCode)
	}

	req, _ = http.NewRequest("POST", ts.URL+"/v1/admin/models", bytes.NewReader([]byte(`{"id":"x","type":"echo"}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer s3cret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("right token: status %d, want 201", resp.StatusCode)
	}
}

func TestAdminMetricsRecorded(t *testing.T) {
	m := metrics.New()
	s := New(registry.New(), m, DefaultConfig())
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Shutdown() })

	adminDo(t, ts, "POST", "/v1/admin/models", `{"id":"m1","type":"echo"}`).Body.Close()
	adminDo(t, ts, "DELETE", "/v1/admin/models/m1", "").Body.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	out := buf.String()
	if !strings.Contains(out, `admin_operations_total{op="register",result="ok"}`) {
		t.Error("expected admin_operations_total for successful register in /metrics")
	}
}
