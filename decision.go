// Package decision defines one interface for the closed decisions an agent
// makes inside its loop (route this ticket, allow this tool call, does this
// answer pass the rule) and a Gate that escalates low-confidence decisions
// from a cheap backend to a more expensive one, and finally to a human.
//
// Backends live in sub-packages: rules (deterministic), local (pure-Go
// trainable classifier), llm (any OpenAI-compatible chat endpoint with
// structured output) and httpdecision (generic adapter for hosted decision
// APIs, configured by a JSON template).
package decision

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

// Label is one allowed answer to a Question.
type Label struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// Question is a typed, closed question: the answer must be one of Labels.
type Question struct {
	Name        string  `json:"name"`
	Description string  `json:"description,omitempty"`
	Labels      []Label `json:"labels"`
}

// LabelNames returns the label names in declaration order.
func (q Question) LabelNames() []string {
	out := make([]string, len(q.Labels))
	for i, l := range q.Labels {
		out[i] = l.Name
	}
	return out
}

// Has reports whether name is one of the question's labels.
func (q Question) Has(name string) bool {
	for _, l := range q.Labels {
		if l.Name == name {
			return true
		}
	}
	return false
}

// Validate checks that the question has a name and at least two distinct labels.
func (q Question) Validate() error {
	if q.Name == "" {
		return errors.New("question: empty name")
	}
	if len(q.Labels) < 2 {
		return fmt.Errorf("question %q: need at least 2 labels", q.Name)
	}
	seen := map[string]bool{}
	for _, l := range q.Labels {
		if l.Name == "" {
			return fmt.Errorf("question %q: empty label name", q.Name)
		}
		if seen[l.Name] {
			return fmt.Errorf("question %q: duplicate label %q", q.Name, l.Name)
		}
		seen[l.Name] = true
	}
	return nil
}

// LoadQuestion reads a Question from a JSON file.
func LoadQuestion(path string) (Question, error) {
	var q Question
	b, err := os.ReadFile(path)
	if err != nil {
		return q, err
	}
	if err := json.Unmarshal(b, &q); err != nil {
		return q, fmt.Errorf("%s: %w", path, err)
	}
	return q, q.Validate()
}

// Result is the typed answer of one backend.
type Result struct {
	Label      string             `json:"label"`
	Confidence float64            `json:"confidence"`
	Probs      map[string]float64 `json:"probs,omitempty"`
	Backend    string             `json:"backend"`
	Latency    time.Duration      `json:"latency_ns"`
	// CostUSD is the marginal cost of this decision as configured by the
	// caller (zero for rules and local).
	CostUSD float64 `json:"cost_usd,omitempty"`
}

// Decider answers a typed question about a state (free text or JSON).
type Decider interface {
	Name() string
	Decide(ctx context.Context, q Question, state string) (Result, error)
}

// ErrInvalidLabel is returned when a backend answers outside the label enum.
var ErrInvalidLabel = errors.New("label not in question enum")

// FromProbs builds a Result from a probability map: the arg-max label and
// its probability as confidence. Ties break by label name for determinism.
func FromProbs(backend string, probs map[string]float64) Result {
	keys := make([]string, 0, len(probs))
	for k := range probs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best, bestP := "", -1.0
	for _, k := range keys {
		if probs[k] > bestP {
			best, bestP = k, probs[k]
		}
	}
	if bestP < 0 {
		bestP = 0
	}
	return Result{Label: best, Confidence: bestP, Probs: probs, Backend: backend}
}

// Clamp01 bounds x to [0, 1].
func Clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
