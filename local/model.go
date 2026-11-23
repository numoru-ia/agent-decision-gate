// Package local is a small trainable text classifier in pure Go: no CGO, no
// network, no external dependencies. It is the "decision model you can run on
// your own infrastructure" backend: train it from a labelled JSONL file,
// serialise it to a gzipped JSON file, and load it next to your agent.
//
// Two algorithms are provided:
//
//   - "logreg": multinomial logistic regression (softmax) trained by SGD with
//     L2 regularisation over sublinear-tf, L2-normalised n-gram features. Its
//     probabilities are usable for confidence gating. This is the default.
//   - "nb": multinomial naive Bayes with additive smoothing. Fast and a good
//     baseline, but its probabilities are notoriously over-confident, so do
//     not gate on them without calibration.
package local

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
)

// Algorithms.
const (
	LogReg = "logreg"
	NB     = "nb"
)

// Model is a trained linear classifier: score_k = Bias_k + sum_f x_f * W_k[f].
type Model struct {
	Version int         `json:"version"`
	Algo    string      `json:"algo"`
	Labels  []string    `json:"labels"`
	Vocab   []string    `json:"vocab"`
	W       [][]float32 `json:"w"`
	Bias    []float64   `json:"bias"`

	index map[string]int
}

func (m *Model) init() error {
	if len(m.Labels) < 2 {
		return errors.New("local: model needs at least 2 labels")
	}
	if len(m.W) != len(m.Labels) || len(m.Bias) != len(m.Labels) {
		return errors.New("local: weight shape does not match labels")
	}
	m.index = make(map[string]int, len(m.Vocab))
	for i, f := range m.Vocab {
		m.index[f] = i
	}
	for k := range m.W {
		if len(m.W[k]) != len(m.Vocab) {
			return fmt.Errorf("local: weight row %d has %d columns, vocab has %d", k, len(m.W[k]), len(m.Vocab))
		}
	}
	return nil
}

// Probs returns the class probabilities for text.
func (m *Model) Probs(text string) map[string]float64 {
	f := Features(text)
	if m.Algo == LogReg {
		f = sublinearL2(f)
	}
	scores := make([]float64, len(m.Labels))
	for k := range m.Labels {
		s := m.Bias[k]
		for feat, x := range f {
			if j, ok := m.index[feat]; ok {
				s += x * float64(m.W[k][j])
			}
		}
		scores[k] = s
	}
	p := softmax(scores)
	out := make(map[string]float64, len(m.Labels))
	for k, l := range m.Labels {
		out[l] = p[k]
	}
	return out
}

func softmax(s []float64) []float64 {
	maxS := math.Inf(-1)
	for _, v := range s {
		if v > maxS {
			maxS = v
		}
	}
	out := make([]float64, len(s))
	sum := 0.0
	for i, v := range s {
		out[i] = math.Exp(v - maxS)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Save writes the model as gzipped JSON.
func (m *Model) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	zw := gzip.NewWriter(f)
	if err := json.NewEncoder(zw).Encode(m); err != nil {
		f.Close()
		return err
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads a model written by Save.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var m Model
	if err := json.NewDecoder(zr).Decode(&m); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if err := m.init(); err != nil {
		return nil, err
	}
	return &m, nil
}

// Decider wraps a Model as a decision.Decider.
type Decider struct{ M *Model }

// Name implements decision.Decider.
func (d Decider) Name() string { return "local-" + d.M.Algo }

// Decide implements decision.Decider. Probabilities are restricted to the
// question's labels and renormalised, so one model can serve a question
// that uses a subset of its labels.
func (d Decider) Decide(ctx context.Context, q decision.Question, state string) (decision.Result, error) {
	start := time.Now()
	all := d.M.Probs(state)
	probs := map[string]float64{}
	sum := 0.0
	for _, l := range q.LabelNames() {
		if p, ok := all[l]; ok {
			probs[l] = p
			sum += p
		}
	}
	if sum == 0 {
		return decision.Result{}, fmt.Errorf("local: model shares no labels with question %q", q.Name)
	}
	for k := range probs {
		probs[k] /= sum
	}
	r := decision.FromProbs(d.Name(), probs)
	r.Latency = time.Since(start)
	return r, nil
}
