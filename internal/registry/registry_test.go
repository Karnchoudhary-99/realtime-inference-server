package registry

import (
	"sync"
	"testing"

	"github.com/Karnchoudhary-99/realtime-inference-server/internal/models"
)

func TestRegisterGetList(t *testing.T) {
	r := New()
	m := models.NewEchoModel("echo", "v1")
	if err := r.Register("echo", m); err != nil {
		t.Fatal(err)
	}
	got, ok := r.Get("echo")
	if !ok || got != m {
		t.Fatal("Get did not return the registered model")
	}
	if _, ok := r.Get("nope"); ok {
		t.Fatal("Get returned a model for an unknown id")
	}
	entries := r.List()
	if len(entries) != 1 || entries[0].ID != "echo" || entries[0].Version != "v1" {
		t.Fatalf("bad list: %+v", entries)
	}
}

func TestRegisterDuplicateFails(t *testing.T) {
	r := New()
	m := models.NewEchoModel("echo", "v1")
	if err := r.Register("echo", m); err != nil {
		t.Fatal(err)
	}
	if err := r.Register("echo", m); err == nil {
		t.Fatal("expected duplicate error, got nil")
	}
}

func TestHotSwap(t *testing.T) {
	r := New()
	v1 := models.NewEchoModel("echo", "v1")
	v2 := models.NewEchoModel("echo", "v2")
	if err := r.Register("echo", v1); err != nil {
		t.Fatal(err)
	}
	if err := r.HotSwap("missing", v2); err == nil {
		t.Fatal("expected not-found error for hot-swap of unknown id")
	}
	if err := r.HotSwap("echo", v2); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("echo")
	if got.Version() != "v2" {
		t.Fatalf("hot-swap did not take effect: %s", got.Version())
	}
}

func TestUnregister(t *testing.T) {
	r := New()
	if err := r.Unregister("ghost"); err == nil {
		t.Fatal("expected not-found error")
	}
	if err := r.Register("echo", models.NewEchoModel("echo", "v1")); err != nil {
		t.Fatal(err)
	}
	if err := r.Unregister("echo"); err != nil {
		t.Fatal(err)
	}
	if r.Len() != 0 {
		t.Fatal("registry not empty after unregister")
	}
}

func TestConcurrentAccess(t *testing.T) {
	r := New()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				m := models.NewEchoModel("e", "v1")
				_ = r.Register("e", m)
				_, _ = r.Get("e")
				_ = r.List()
				_ = r.HotSwap("e", m)
			}
		}()
	}
	wg.Wait()
}
