package decision

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var q = Question{Name: "q", Labels: []Label{{Name: "a"}, {Name: "b"}, {Name: "c"}}}

type fake struct {
	name string
	r    Result
	err  error
	n    int
}

func (f *fake) Name() string { return f.name }
func (f *fake) Decide(ctx context.Context, q Question, s string) (Result, error) {
	f.n++
	return f.r, f.err
}

func TestQuestionValidate(t *testing.T) {
	if err := q.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := []Question{
		{Labels: []Label{{Name: "a"}, {Name: "b"}}},
		{Name: "x", Labels: []Label{{Name: "a"}}},
		{Name: "x", Labels: []Label{{Name: "a"}, {Name: "a"}}},
		{Name: "x", Labels: []Label{{Name: "a"}, {Name: ""}}},
	}
	for i, b := range bad {
		if b.Validate() == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
	if !q.Has("b") || q.Has("z") {
		t.Fatal("Has")
	}
	if got := strings.Join(q.LabelNames(), ","); got != "a,b,c" {
		t.Fatal(got)
	}
}

func TestLoadQuestion(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "q.json")
	os.WriteFile(p, []byte(`{"name":"n","labels":[{"name":"x"},{"name":"y"}]}`), 0o600)
	got, err := LoadQuestion(p)
	if err != nil || got.Name != "n" || len(got.Labels) != 2 {
		t.Fatal(got, err)
	}
	os.WriteFile(p, []byte(`{`), 0o600)
	if _, err := LoadQuestion(p); err == nil {
		t.Fatal("want parse error")
	}
	if _, err := LoadQuestion(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("want missing file error")
	}
}

func TestFromProbsAndClamp(t *testing.T) {
	r := FromProbs("x", map[string]float64{"b": 0.4, "a": 0.4, "c": 0.2})
	if r.Label != "a" || r.Confidence != 0.4 {
		t.Fatalf("tie should break by name: %+v", r)
	}
	if r := FromProbs("x", nil); r.Confidence != 0 {
		t.Fatal(r)
	}
	if Clamp01(-1) != 0 || Clamp01(2) != 1 || Clamp01(0.3) != 0.3 {
		t.Fatal("clamp")
	}
}

func TestGateAcceptsFirstConfident(t *testing.T) {
	a := &fake{name: "cheap", r: Result{Label: "a", Confidence: 0.95, CostUSD: 0}}
	b := &fake{name: "llm", r: Result{Label: "b", Confidence: 0.9, CostUSD: 0.001}}
	var log bytes.Buffer
	g := &Gate{Stages: []Stage{{a, 0.8}, {b, 0.5}}, Log: &log, Now: func() time.Time { return time.Unix(0, 0) }}
	o, err := g.Decide(context.Background(), q, "texto")
	if err != nil {
		t.Fatal(err)
	}
	if o.Label != "a" || o.Stage != 0 || o.NeedsHuman || b.n != 0 {
		t.Fatalf("%+v", o)
	}
	var e LogEntry
	if err := json.Unmarshal(log.Bytes(), &e); err != nil {
		t.Fatal(err)
	}
	if !e.Accepted || e.Backend != "cheap" || e.InputHash != HashInput("texto") || strings.Contains(log.String(), "texto") {
		t.Fatalf("log entry %+v / %s", e, log.String())
	}
}

func TestGateEscalates(t *testing.T) {
	a := &fake{name: "cheap", r: Result{Label: "a", Confidence: 0.4}}
	b := &fake{name: "llm", r: Result{Label: "b", Confidence: 0.9, CostUSD: 0.002}}
	var log bytes.Buffer
	g := &Gate{Stages: []Stage{{a, 0.8}, {b, 0.5}}, Log: &log}
	o, _ := g.Decide(context.Background(), q, "s")
	if o.Label != "b" || o.Stage != 1 || len(o.Trail) != 2 || o.CostUSD != 0.002 {
		t.Fatalf("%+v", o)
	}
	if n := strings.Count(log.String(), "\n"); n != 2 {
		t.Fatalf("want 2 log lines, got %d", n)
	}
}

func TestGateHumanAndErrors(t *testing.T) {
	a := &fake{name: "broken", err: errors.New("boom")}
	b := &fake{name: "liar", r: Result{Label: "zzz", Confidence: 1}}
	c := &fake{name: "unsure", r: Result{Label: "c", Confidence: 0.1, CostUSD: 0.01}}
	var log bytes.Buffer
	g := &Gate{Stages: []Stage{{a, 0.5}, {b, 0.5}, {c, 0.5}}, Log: &log, HumanLabel: "persona"}
	o, err := g.Decide(context.Background(), q, "s")
	if err != nil {
		t.Fatal(err)
	}
	if !o.NeedsHuman || o.Label != "persona" || o.Stage != 3 || o.CostUSD != 0.01 {
		t.Fatalf("%+v", o)
	}
	sc := bufio.NewScanner(&log)
	var entries []LogEntry
	for sc.Scan() {
		var e LogEntry
		json.Unmarshal(sc.Bytes(), &e)
		entries = append(entries, e)
	}
	if len(entries) != 4 || entries[0].Error != "boom" || !strings.Contains(entries[1].Error, "enum") || entries[3].Backend != "human" {
		t.Fatalf("%+v", entries)
	}
	g2 := &Gate{Stages: []Stage{{c, 0.5}}}
	if o, _ := g2.Decide(context.Background(), q, "s"); o.Label != "human_review" {
		t.Fatal("default human label")
	}
}

func TestGateEdgeCases(t *testing.T) {
	if _, err := (&Gate{}).Decide(context.Background(), q, "s"); !errors.Is(err, ErrNoStages) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	g := &Gate{Stages: []Stage{{&fake{name: "a", r: Result{Label: "a", Confidence: 1}}, 0.5}}}
	if _, err := g.Decide(ctx, q, "s"); err == nil {
		t.Fatal("want ctx error")
	}
}

func TestAsDecider(t *testing.T) {
	ok := &Gate{Stages: []Stage{{&fake{name: "a", r: Result{Label: "a", Confidence: 0.9}}, 0.5}}}
	r, err := ok.AsDecider("g").Decide(context.Background(), q, "s")
	if err != nil || r.Confidence != 1 || r.Label != "a" || ok.AsDecider("g").Name() != "g" {
		t.Fatal(r, err)
	}
	no := &Gate{Stages: []Stage{{&fake{name: "a", r: Result{Label: "a", Confidence: 0.1}}, 0.5}}}
	r, _ = no.AsDecider("g").Decide(context.Background(), q, "s")
	if r.Confidence != 0 || r.Label != "human_review" {
		t.Fatal(r)
	}
	if _, err := (&Gate{}).AsDecider("g").Decide(context.Background(), q, "s"); err == nil {
		t.Fatal("want error")
	}
}
