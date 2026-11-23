// Package bench evaluates a Decider over a labelled set: accuracy, macro-F1,
// expected calibration error, latency percentiles, cost per decision, and
// what a confidence gate at a given threshold would let through.
package bench

import (
	"context"
	"fmt"
	"io"
	"math"
	"sort"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/dataset"
)

// Report is the result of one backend over one test set.
type Report struct {
	Backend  string        `json:"backend"`
	N        int           `json:"n"`
	Errors   int           `json:"errors"`
	Accuracy float64       `json:"accuracy"`
	MacroF1  float64       `json:"macro_f1"`
	ECE      float64       `json:"ece"`
	P50      time.Duration `json:"p50_ns"`
	P95      time.Duration `json:"p95_ns"`
	// CostPerDecision is the mean CostUSD reported by the backend.
	CostPerDecision float64 `json:"cost_per_decision_usd"`
	// Threshold, Coverage and AccAtThreshold describe a gate: the share of
	// decisions with confidence >= Threshold, and accuracy on that share.
	Threshold      float64            `json:"threshold"`
	Coverage       float64            `json:"coverage"`
	AccAtThreshold float64            `json:"acc_at_threshold"`
	PerLabelF1     map[string]float64 `json:"per_label_f1"`
}

// Run evaluates d on examples. Backend errors count as wrong answers.
func Run(ctx context.Context, d decision.Decider, q decision.Question, ex []dataset.Example, threshold float64) Report {
	rep := Report{Backend: d.Name(), N: len(ex), Threshold: threshold}
	gold := make([]string, len(ex))
	pred := make([]string, len(ex))
	conf := make([]float64, len(ex))
	lat := make([]time.Duration, 0, len(ex))
	cost := 0.0
	for i, e := range ex {
		gold[i] = e.Label
		start := time.Now()
		r, err := d.Decide(ctx, q, e.Text)
		el := time.Since(start)
		if r.Latency == 0 {
			r.Latency = el
		}
		lat = append(lat, r.Latency)
		if err != nil {
			rep.Errors++
			continue
		}
		pred[i] = r.Label
		conf[i] = r.Confidence
		cost += r.CostUSD
	}
	correct, covered, coveredOK := 0, 0, 0
	for i := range ex {
		ok := pred[i] == gold[i]
		if ok {
			correct++
		}
		if pred[i] != "" && conf[i] >= threshold {
			covered++
			if ok {
				coveredOK++
			}
		}
	}
	if len(ex) > 0 {
		rep.Accuracy = float64(correct) / float64(len(ex))
		rep.Coverage = float64(covered) / float64(len(ex))
		rep.CostPerDecision = cost / float64(len(ex))
	}
	if covered > 0 {
		rep.AccAtThreshold = float64(coveredOK) / float64(covered)
	}
	rep.MacroF1, rep.PerLabelF1 = MacroF1(gold, pred)
	rep.ECE = ECE(gold, pred, conf, 10)
	rep.P50 = Percentile(lat, 0.50)
	rep.P95 = Percentile(lat, 0.95)
	return rep
}

// MacroF1 averages per-label F1 over every label present in gold or pred.
func MacroF1(gold, pred []string) (float64, map[string]float64) {
	labels := map[string]bool{}
	for _, g := range gold {
		labels[g] = true
	}
	for _, p := range pred {
		if p != "" {
			labels[p] = true
		}
	}
	per := map[string]float64{}
	if len(labels) == 0 {
		return 0, per
	}
	sum := 0.0
	for l := range labels {
		tp, fp, fn := 0.0, 0.0, 0.0
		for i := range gold {
			switch {
			case pred[i] == l && gold[i] == l:
				tp++
			case pred[i] == l:
				fp++
			case gold[i] == l:
				fn++
			}
		}
		f := 0.0
		if tp > 0 {
			p, r := tp/(tp+fp), tp/(tp+fn)
			f = 2 * p * r / (p + r)
		}
		per[l] = f
		sum += f
	}
	return sum / float64(len(labels)), per
}

// ECE is the expected calibration error with equal-width confidence bins.
// Errors (empty prediction) are skipped.
func ECE(gold, pred []string, conf []float64, bins int) float64 {
	type bin struct{ n, ok, conf float64 }
	bs := make([]bin, bins)
	total := 0.0
	for i := range gold {
		if pred[i] == "" {
			continue
		}
		b := int(conf[i] * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		bs[b].n++
		bs[b].conf += conf[i]
		if pred[i] == gold[i] {
			bs[b].ok++
		}
		total++
	}
	if total == 0 {
		return 0
	}
	e := 0.0
	for _, b := range bs {
		if b.n == 0 {
			continue
		}
		e += b.n / total * math.Abs(b.ok/b.n-b.conf/b.n)
	}
	return e
}

// Percentile uses the nearest-rank method.
func Percentile(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	k := int(math.Ceil(p*float64(len(s)))) - 1
	if k < 0 {
		k = 0
	}
	return s[k]
}

// FormatDuration prints a latency with a unit that fits its size.
func FormatDuration(d time.Duration) string {
	switch {
	case d < time.Millisecond:
		return fmt.Sprintf("%.0f µs", float64(d)/float64(time.Microsecond))
	case d < time.Second:
		return fmt.Sprintf("%.1f ms", float64(d)/float64(time.Millisecond))
	default:
		return fmt.Sprintf("%.2f s", d.Seconds())
	}
}

// WriteMarkdown prints reports as a Markdown table.
func WriteMarkdown(w io.Writer, reps []Report) {
	if len(reps) == 0 {
		return
	}
	th := reps[0].Threshold
	fmt.Fprintf(w, "| backend | n | accuracy | macro-F1 | ECE | p50 | p95 | cost/decision (USD) | coverage@%.2f | acc@%.2f | errors |\n", th, th)
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|---|---|")
	for _, r := range reps {
		fmt.Fprintf(w, "| %s | %d | %.3f | %.3f | %.3f | %s | %s | %.6f | %.2f | %.3f | %d |\n",
			r.Backend, r.N, r.Accuracy, r.MacroF1, r.ECE, FormatDuration(r.P50), FormatDuration(r.P95),
			r.CostPerDecision, r.Coverage, r.AccAtThreshold, r.Errors)
	}
}
