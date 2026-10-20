package models

import (
	"fmt"
	"strings"
	"time"
)

// ModelSpec is a JSON-serializable description of one built-in model
// instance. The admin API turns specs into real Model implementations at
// runtime, so operators can add or replace models without redeploying.
//
// Arbitary model code cannot be loaded at runtime — that is a deliberate
// security boundary — so the zoo only instantiates the built-ins with
// operator-chosen dimensions, seeds and version labels.
type ModelSpec struct {
	// Type selects the built-in: "mlp", "linear", "echo" or "storygen".
	Type string `json:"type"`
	// Version is an opaque label, e.g. "v1". Empty defaults to "v1".
	Version string `json:"version,omitempty"`
	// InputDim / OutputDim are the feature dimensions for mlp and linear.
	InputDim  int `json:"input_dim,omitempty"`
	OutputDim int `json:"output_dim,omitempty"`
	// Hidden is the MLP hidden width (mlp only). Empty defaults to 16.
	Hidden int `json:"hidden,omitempty"`
	// Seed makes the model's weights deterministic.
	Seed uint64 `json:"seed,omitempty"`
	// NumTokens caps generated tokens (storygen only). Empty defaults to 24.
	NumTokens int `json:"num_tokens,omitempty"`
	// TokenDelayMs is the artificial per-token delay (storygen only).
	TokenDelayMs int `json:"token_delay_ms,omitempty"`
}

// BuildModel instantiates a Model from name and spec, validating the spec
// for the requested type.
func BuildModel(name string, spec ModelSpec) (Model, error) {
	if name == "" {
		return nil, fmt.Errorf("zoo: empty model name")
	}
	version := spec.Version
	if version == "" {
		version = "v1"
	}
	switch strings.ToLower(spec.Type) {
	case "mlp":
		if spec.InputDim <= 0 {
			return nil, fmt.Errorf("zoo: mlp %q needs input_dim > 0", name)
		}
		if spec.OutputDim <= 0 {
			return nil, fmt.Errorf("zoo: mlp %q needs output_dim > 0", name)
		}
		hidden := spec.Hidden
		if hidden == 0 {
			hidden = 16
		}
		if hidden < 0 {
			return nil, fmt.Errorf("zoo: mlp %q needs hidden >= 0", name)
		}
		return NewMLPClassifier(name, version, spec.InputDim, hidden, spec.OutputDim, spec.Seed), nil
	case "linear":
		if spec.InputDim <= 0 {
			return nil, fmt.Errorf("zoo: linear %q needs input_dim > 0", name)
		}
		if spec.OutputDim <= 0 {
			return nil, fmt.Errorf("zoo: linear %q needs output_dim > 0", name)
		}
		return NewLinearModel(name, version, spec.InputDim, spec.OutputDim, spec.Seed), nil
	case "echo":
		return NewEchoModel(name, version), nil
	case "storygen":
		numTokens := spec.NumTokens
		if numTokens == 0 {
			numTokens = 24
		}
		if numTokens < 0 {
			return nil, fmt.Errorf("zoo: storygen %q needs num_tokens >= 0", name)
		}
		if spec.TokenDelayMs < 0 {
			return nil, fmt.Errorf("zoo: storygen %q needs token_delay_ms >= 0", name)
		}
		return NewStoryGenerator(name, version, numTokens,
			time.Duration(spec.TokenDelayMs)*time.Millisecond), nil
	default:
		return nil, fmt.Errorf("zoo: unknown model type %q (want mlp, linear, echo or storygen)", spec.Type)
	}
}
