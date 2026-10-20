package models

import "testing"

func TestBuildModelMLP(t *testing.T) {
	m, err := BuildModel("test-mlp", ModelSpec{Type: "mlp", InputDim: 4, OutputDim: 3, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if m.InputDim() != 4 || m.OutputDim() != 3 {
		t.Fatalf("dims = %d/%d, want 4/3", m.InputDim(), m.OutputDim())
	}
	if m.Version() != "v1" {
		t.Fatalf("default version = %q, want v1", m.Version())
	}
}

func TestBuildModelDefaults(t *testing.T) {
	m, err := BuildModel("gen", ModelSpec{Type: "storygen", Version: "v3"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.(StreamingModel); !ok {
		t.Fatal("storygen spec should produce a StreamingModel")
	}
	if m.Version() != "v3" {
		t.Fatalf("version = %q, want v3", m.Version())
	}
}

func TestBuildModelCaseInsensitive(t *testing.T) {
	if _, err := BuildModel("e", ModelSpec{Type: "ECHO"}); err != nil {
		t.Fatal(err)
	}
}

func TestBuildModelErrors(t *testing.T) {
	cases := []struct {
		name string
		spec ModelSpec
	}{
		{"unknown type", ModelSpec{Type: "transformer"}},
		{"mlp missing dims", ModelSpec{Type: "mlp", OutputDim: 3}},
		{"linear missing dims", ModelSpec{Type: "linear", InputDim: 4}},
		{"storygen negative tokens", ModelSpec{Type: "storygen", NumTokens: -1}},
		{"storygen negative delay", ModelSpec{Type: "storygen", TokenDelayMs: -1}},
	}
	for _, c := range cases {
		if _, err := BuildModel("x", c.spec); err == nil {
			t.Errorf("%s: expected error, got nil", c.name)
		}
	}
	if _, err := BuildModel("", ModelSpec{Type: "echo"}); err == nil {
		t.Error("empty name: expected error, got nil")
	}
}
