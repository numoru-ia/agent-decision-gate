package bench

import (
	"bytes"
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/dataset"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestMacroF1(t *testing.T) {
	gold := []string{"a", "a", "b", "b"}
	pred := []string{"a", "b", "b", "b"}
	// a: p=1 r=.5 f=2/3 ; b: p=2/3 r=1 f=.8
	m, per := MacroF1(gold, pred)
	if !near(per["a"], 2.0/3) || !near(per["b"], 0.8) || !near(m, (2.0/3+0.8)/2) {
		t.Fatal(m, per)
	}
	m, _ = MacroF1([]string{"a"}, []string{""})
	if m != 0 {
		t.Fatal(m)
	}
	if m, _ := MacroF1(nil, nil); m != 0 {
		t.Fatal(m)
	}
}

func TestECE(t *testing.T) {
	// two at 0.95 both right, two at 0.55 one right
	gold := []string{"a", "a", "a", "a", "a"}
	pred := []string{"a", "a", "a", "b", ""}
	conf := []float64{0.95, 0.95, 0.55, 0.55, 0.9}
	got := ECE(gold, pred, conf, 10)
	want := 0.5*math.Abs(1-0.95) + 0.5*math.Abs(0.5-0.55)
	if !near(got, want) {
		t.Fatal(got, want)
	}
	if ECE(gold, []string{"", "", "", "", ""}, conf, 10) != 0 {
		t.Fatal("all errors")
	}
	if !near(ECE([]string{"a"}, []string{"a"}, []float64{1}, 10), 0) {
		t.Fatal("conf 1 lands in the last bin")
	}
}

func TestPercentileAndFormat(t *testing.T) {
	d := []time.Duration{5, 1, 4, 2, 3}
	if Percentile(d, 0.5) != 3 || Percentile(d, 0.95) != 5 || Percentile(d, 0) != 1 || Percentile(nil, 0.5) != 0 {
		t.Fatal("percentile")
	}
	for in, want := range map[time.Duration]string{
		42 * time.Microsecond: "42 µs", 12500 * time.Microsecond: "12.5 ms", 1500 * time.Millisecond: "1.50 s",
	} {
		if got := FormatDuration(in); got != want {
			t.Errorf("%v: %s", in, got)
		}
	}
}

type script struct{ i int }

func (s *script) Name() string { return "script" }
func (s *script) Decide(ctx context.Context, q decision.Question, st string) (decision.Result, error) {
	s.i++
	switch s.i {
	case 1:
		return decision.Result{Label: "a", Confidence: 0.9, CostUSD: 0.01, Latency: time.Millisecond}, nil
	case 2:
		return decision.Result{Label: "a", Confidence: 0.4, CostUSD: 0.01}, nil
	default:
		return decision.Result{}, errors.New("down")
	}
}

func TestRunAndMarkdown(t *testing.T) {
	q := decision.Question{Name: "q", Labels: []decision.Label{{Name: "a"}, {Name: "b"}}}
	ex := []dataset.Example{{Text: "1", Label: "a"}, {Text: "2", Label: "b"}, {Text: "3", Label: "a"}}
	r := Run(context.Background(), &script{}, q, ex, 0.5)
	if r.N != 3 || r.Errors != 1 || !near(r.Accuracy, 1.0/3) || !near(r.Coverage, 1.0/3) || r.AccAtThreshold != 1 || !near(r.CostPerDecision, 0.02/3) {
		t.Fatalf("%+v", r)
	}
	var b bytes.Buffer
	WriteMarkdown(&b, []Report{r})
	if !strings.Contains(b.String(), "| script | 3 | 0.333") || !strings.Contains(b.String(), "coverage@0.50") {
		t.Fatal(b.String())
	}
	b.Reset()
	WriteMarkdown(&b, nil)
	if b.Len() != 0 {
		t.Fatal("empty")
	}
}
