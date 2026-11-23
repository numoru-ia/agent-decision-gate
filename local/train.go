package local

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Example is one labelled training text.
type Example struct {
	Text  string
	Label string
}

// Options control training.
type Options struct {
	Algo     string  // LogReg (default) or NB
	Epochs   int     // logreg: passes over the data (default 40)
	LR       float64 // logreg: initial learning rate (default 0.5)
	L2       float64 // logreg: L2 strength (default 1e-5)
	Alpha    float64 // nb: additive smoothing (default 0.1)
	MinCount int     // drop features seen fewer times in training (default 1)
	Seed     int64   // shuffle seed (default 1)
}

func (o *Options) defaults() {
	if o.Algo == "" {
		o.Algo = LogReg
	}
	if o.Epochs == 0 {
		o.Epochs = 40
	}
	if o.LR == 0 {
		o.LR = 0.5
	}
	if o.L2 == 0 {
		o.L2 = 1e-5
	}
	if o.Alpha == 0 {
		o.Alpha = 0.1
	}
	if o.MinCount == 0 {
		o.MinCount = 1
	}
	if o.Seed == 0 {
		o.Seed = 1
	}
}

// Train fits a model. Training is deterministic for a given seed.
func Train(data []Example, opt Options) (*Model, error) {
	opt.defaults()
	if len(data) == 0 {
		return nil, errors.New("local: empty training set")
	}
	labelSet := map[string]bool{}
	for _, e := range data {
		if e.Label == "" {
			return nil, errors.New("local: example with empty label")
		}
		labelSet[e.Label] = true
	}
	if len(labelSet) < 2 {
		return nil, errors.New("local: need at least 2 distinct labels")
	}
	labels := make([]string, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	lidx := map[string]int{}
	for i, l := range labels {
		lidx[l] = i
	}

	raw := make([]map[string]float64, len(data))
	counts := map[string]int{}
	for i, e := range data {
		raw[i] = Features(e.Text)
		for f, c := range raw[i] {
			counts[f] += int(c)
		}
	}
	vocab := make([]string, 0, len(counts))
	for f, c := range counts {
		if c >= opt.MinCount {
			vocab = append(vocab, f)
		}
	}
	sort.Strings(vocab)
	m := &Model{Version: 1, Algo: opt.Algo, Labels: labels, Vocab: vocab}
	m.index = make(map[string]int, len(vocab))
	for i, f := range vocab {
		m.index[f] = i
	}
	m.W = make([][]float32, len(labels))
	for k := range m.W {
		m.W[k] = make([]float32, len(vocab))
	}
	m.Bias = make([]float64, len(labels))

	switch opt.Algo {
	case NB:
		trainNB(m, data, raw, lidx, opt)
	case LogReg:
		trainLR(m, data, raw, lidx, opt)
	default:
		return nil, fmt.Errorf("local: unknown algo %q (want %q or %q)", opt.Algo, LogReg, NB)
	}
	return m, nil
}

func trainNB(m *Model, data []Example, raw []map[string]float64, lidx map[string]int, opt Options) {
	K, V := len(m.Labels), len(m.Vocab)
	docs := make([]float64, K)
	tok := make([][]float64, K)
	totals := make([]float64, K)
	for k := range tok {
		tok[k] = make([]float64, V)
	}
	for i, e := range data {
		k := lidx[e.Label]
		docs[k]++
		for f, c := range raw[i] {
			if j, ok := m.index[f]; ok {
				tok[k][j] += c
				totals[k] += c
			}
		}
	}
	for k := 0; k < K; k++ {
		m.Bias[k] = math.Log(docs[k] / float64(len(data)))
		den := totals[k] + opt.Alpha*float64(V)
		for j := 0; j < V; j++ {
			m.W[k][j] = float32(math.Log((tok[k][j] + opt.Alpha) / den))
		}
	}
}

type sparse struct {
	idx []int
	val []float64
}

func trainLR(m *Model, data []Example, raw []map[string]float64, lidx map[string]int, opt Options) {
	K := len(m.Labels)
	xs := make([]sparse, len(data))
	ys := make([]int, len(data))
	for i, e := range data {
		f := sublinearL2(raw[i])
		keys := make([]string, 0, len(f))
		for k := range f {
			keys = append(keys, k)
		}
		sort.Strings(keys) // deterministic accumulation order
		for _, k := range keys {
			if j, ok := m.index[k]; ok {
				xs[i].idx = append(xs[i].idx, j)
				xs[i].val = append(xs[i].val, f[k])
			}
		}
		ys[i] = lidx[e.Label]
	}
	// float64 working weights, copied to float32 at the end.
	w := make([][]float64, K)
	for k := range w {
		w[k] = make([]float64, len(m.Vocab))
	}
	rng := rand.New(rand.NewSource(opt.Seed))
	order := rng.Perm(len(data))
	scores := make([]float64, K)
	step := 0
	for ep := 0; ep < opt.Epochs; ep++ {
		rng.Shuffle(len(order), func(a, b int) { order[a], order[b] = order[b], order[a] })
		for _, i := range order {
			step++
			lr := opt.LR / (1 + 0.01*float64(step)/float64(len(data)))
			x := xs[i]
			for k := 0; k < K; k++ {
				s := m.Bias[k]
				for t, j := range x.idx {
					s += w[k][j] * x.val[t]
				}
				scores[k] = s
			}
			p := softmax(scores)
			for k := 0; k < K; k++ {
				g := p[k]
				if k == ys[i] {
					g -= 1
				}
				m.Bias[k] -= lr * g
				for t, j := range x.idx {
					w[k][j] -= lr * (g*x.val[t] + opt.L2*w[k][j])
				}
			}
		}
	}
	for k := range w {
		for j := range w[k] {
			m.W[k][j] = float32(w[k][j])
		}
	}
}
