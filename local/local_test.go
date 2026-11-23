package local

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/dataset"
)

var tiny = []Example{
	{"no puedo iniciar sesión, olvidé mi contraseña", "acceso"},
	{"mi cuenta está bloqueada", "acceso"},
	{"no me llega el código de verificación", "acceso"},
	{"quiero cancelar una factura", "cancelacion"},
	{"cancelé la factura y sigue vigente", "cancelacion"},
	{"motivo de cancelación 02", "cancelacion"},
}

var q = decision.Question{Name: "q", Labels: []decision.Label{{Name: "acceso"}, {Name: "cancelacion"}}}

func TestTrainPredictBothAlgos(t *testing.T) {
	for _, algo := range []string{LogReg, NB} {
		m, err := Train(tiny, Options{Algo: algo})
		if err != nil {
			t.Fatal(err)
		}
		d := Decider{M: m}
		r, err := d.Decide(context.Background(), q, "olvide mi contrasena")
		if err != nil || r.Label != "acceso" {
			t.Fatalf("%s: %+v %v", algo, r, err)
		}
		r, _ = d.Decide(context.Background(), q, "cancelar facturra")
		if r.Label != "cancelacion" || r.Confidence <= 0.5 || r.Confidence > 1 {
			t.Fatalf("%s typo: %+v", algo, r)
		}
		if d.Name() != "local-"+algo {
			t.Fatal(d.Name())
		}
	}
}

func TestDeterministicAndRoundTrip(t *testing.T) {
	a, _ := Train(tiny, Options{Seed: 7})
	b, _ := Train(tiny, Options{Seed: 7})
	if !reflect.DeepEqual(a.W, b.W) {
		t.Fatal("training is not deterministic")
	}
	p := filepath.Join(t.TempDir(), "m.gz")
	if err := a.Save(p); err != nil {
		t.Fatal(err)
	}
	c, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	pa, pc := a.Probs("cuenta bloqueada"), c.Probs("cuenta bloqueada")
	for k := range pa {
		if d := pa[k] - pc[k]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("roundtrip drift %v vs %v", pa, pc)
		}
	}
}

func TestSubsetQuestion(t *testing.T) {
	m, _ := Train(append(tiny, Example{"precio del plan", "ventas"}, Example{"cotización", "ventas"}), Options{})
	r, err := Decider{M: m}.Decide(context.Background(), q, "precio del plan")
	if err != nil {
		t.Fatal(err)
	}
	if r.Label == "ventas" || len(r.Probs) != 2 {
		t.Fatalf("labels outside the question must be dropped: %+v", r)
	}
	other := decision.Question{Name: "o", Labels: []decision.Label{{Name: "x"}, {Name: "y"}}}
	if _, err := (Decider{M: m}).Decide(context.Background(), other, "a"); err == nil {
		t.Fatal("want no-shared-labels error")
	}
}

func TestTrainErrors(t *testing.T) {
	if _, err := Train(nil, Options{}); err == nil {
		t.Fatal("empty")
	}
	if _, err := Train([]Example{{"a", "x"}, {"b", "x"}}, Options{}); err == nil {
		t.Fatal("one label")
	}
	if _, err := Train([]Example{{"a", ""}}, Options{}); err == nil {
		t.Fatal("empty label")
	}
	if _, err := Train(tiny, Options{Algo: "svm"}); err == nil {
		t.Fatal("bad algo")
	}
}

func TestLoadErrors(t *testing.T) {
	dir := t.TempDir()
	if _, err := Load(filepath.Join(dir, "nope")); err == nil {
		t.Fatal("missing")
	}
	p := filepath.Join(dir, "plain")
	os.WriteFile(p, []byte("not gzip"), 0o600)
	if _, err := Load(p); err == nil {
		t.Fatal("not gzip")
	}
	m := &Model{Labels: []string{"a"}}
	if m.init() == nil {
		t.Fatal("one label")
	}
	m = &Model{Labels: []string{"a", "b"}, W: [][]float32{{1}}, Bias: []float64{0}}
	if m.init() == nil {
		t.Fatal("shape")
	}
	m = &Model{Labels: []string{"a", "b"}, Vocab: []string{"x"}, W: [][]float32{{1}, {}}, Bias: []float64{0, 0}}
	if m.init() == nil {
		t.Fatal("row width")
	}
	if err := (&Model{}).Save(filepath.Join(dir, "no", "such", "dir")); err == nil {
		t.Fatal("save to bad path")
	}
}

// The synthetic ticket set must stay learnable but not trivial: this guards
// against a generator change that leaks test phrasings into training.
func TestSyntheticTicketsSanity(t *testing.T) {
	tr, err := dataset.Load("../examples/tickets-es/synthetic-train.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	te, err := dataset.Load("../examples/tickets-es/synthetic-test.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	ex := make([]Example, len(tr))
	for i, e := range tr {
		ex[i] = Example{e.Text, e.Label}
	}
	m, err := Train(ex, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ok := 0
	for _, e := range te {
		p := m.Probs(e.Text)
		best, bp := "", -1.0
		for k, v := range p {
			if v > bp {
				best, bp = k, v
			}
		}
		if best == e.Label {
			ok++
		}
	}
	acc := float64(ok) / float64(len(te))
	if acc < 0.6 || acc > 0.97 {
		t.Fatalf("accuracy %.2f outside the expected band", acc)
	}
}
