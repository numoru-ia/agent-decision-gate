package rules

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
)

var q = decision.Question{Name: "q", Labels: []decision.Label{{Name: "allow"}, {Name: "deny"}, {Name: "ask_human"}}}

func mustNew(t *testing.T, c Config) *Decider {
	t.Helper()
	d, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestKeywordsAndScores(t *testing.T) {
	d := mustNew(t, Config{Default: "ask_human", Rules: []Rule{
		{Label: "allow", Any: []string{"Búsqueda"}},
		{Label: "deny", Any: []string{"borrar"}, Weight: 3},
	}})
	ctx := context.Background()
	r, _ := d.Decide(ctx, q, "busqueda en la base")
	if r.Label != "allow" || r.Confidence != 1 {
		t.Fatalf("%+v", r)
	}
	r, _ = d.Decide(ctx, q, "busqueda y BORRAR todo")
	if r.Label != "deny" || r.Confidence != 0.75 {
		t.Fatalf("%+v", r)
	}
	r, _ = d.Decide(ctx, q, "nada")
	if r.Label != "ask_human" || r.Confidence != 0 {
		t.Fatalf("%+v", r)
	}
	if d.Name() != "rules" {
		t.Fatal(d.Name())
	}
}

func TestDefaultFallsBackToFirstLabel(t *testing.T) {
	d := mustNew(t, Config{Default: "not-a-label", Rules: []Rule{{Label: "deny", Any: []string{"x"}}}})
	r, _ := d.Decide(context.Background(), q, "zzz")
	if r.Label != "allow" {
		t.Fatal(r)
	}
}

func TestFinalRegexAndFields(t *testing.T) {
	d := mustNew(t, Config{Rules: []Rule{
		{Label: "allow", Fields: []Cond{{Field: "tool", Op: "==", Value: "db_query"}}},
		{Label: "deny", Regex: []string{`\bdrop\b`}, Fields: []Cond{{Field: "tool", Op: "==", Value: "db_query"}}, Final: true},
	}})
	r, _ := d.Decide(context.Background(), q, `{"tool":"db_query","args":{"sql":"DROP TABLE x"}}`)
	if r.Label != "deny" || r.Confidence != 1 {
		t.Fatalf("%+v", r)
	}
	r, _ = d.Decide(context.Background(), q, `{"tool":"db_query","args":{"sql":"select 1"}}`)
	if r.Label != "allow" {
		t.Fatalf("%+v", r)
	}
	// fields never match non-JSON state
	r, _ = d.Decide(context.Background(), q, `tool db_query drop`)
	if r.Confidence != 0 {
		t.Fatalf("%+v", r)
	}
}

func TestOps(t *testing.T) {
	now := time.Date(2026, 11, 22, 0, 0, 0, 0, time.UTC)
	doc := map[string]any{
		"amount": 15000.0, "name": "Comercial Ñandú", "tool": "shell", "flag": true,
		"due": "2026-10-01", "ts": "2026-11-20T10:00:00Z", "items": []any{map[string]any{"sku": "A1"}},
		"strnum": "42",
	}
	d := &Decider{Now: func() time.Time { return now }}
	cases := []struct {
		c    Cond
		want bool
	}{
		{Cond{"amount", ">", 10000.0}, true},
		{Cond{"amount", ">=", 15000.0}, true},
		{Cond{"amount", "<", 10000.0}, false},
		{Cond{"amount", "<=", 15000.0}, true},
		{Cond{"amount", ">", "abc"}, false},
		{Cond{"strnum", "==", 42.0}, true},
		{Cond{"name", "==", "comercial nandu"}, true},
		{Cond{"name", "!=", "otra"}, true},
		{Cond{"flag", "==", true}, true},
		{Cond{"flag", "==", "true"}, false},
		{Cond{"tool", "in", []any{"shell", "x"}}, true},
		{Cond{"tool", "not_in", []any{"shell"}}, false},
		{Cond{"name", "contains", "ÑANDÚ"}, true},
		{Cond{"items.0.sku", "==", "A1"}, true},
		{Cond{"items.5.sku", "exists", nil}, false},
		{Cond{"items.x", "exists", nil}, false},
		{Cond{"amount.x", "exists", nil}, false},
		{Cond{"missing", "==", 1.0}, false},
		{Cond{"due", "older_than_days", 30.0}, true},
		{Cond{"due", "newer_than_days", 30.0}, false},
		{Cond{"ts", "newer_than_days", 7.0}, true},
		{Cond{"name", "older_than_days", 1.0}, false},
		{Cond{"amount", "older_than_days", 1.0}, false},
		{Cond{"amount", "bogus", 1.0}, false},
	}
	for _, c := range cases {
		if got := d.eval(c.c, doc); got != c.want {
			t.Errorf("%+v: got %v want %v", c.c, got, c.want)
		}
	}
}

func TestNewErrors(t *testing.T) {
	bad := []Config{
		{Rules: []Rule{{Any: []string{"x"}}}},
		{Rules: []Rule{{Label: "a"}}},
		{Rules: []Rule{{Label: "a", Regex: []string{"("}}}},
	}
	for i, c := range bad {
		if _, err := New(c); err == nil {
			t.Errorf("case %d: want error", i)
		}
	}
}

func TestLoadExamples(t *testing.T) {
	for _, p := range []string{"../examples/tickets-es/rules.json", "../examples/tool-gating/rules.json"} {
		if _, err := Load(p); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
	}
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	os.WriteFile(bad, []byte("{"), 0o600)
	if _, err := Load(bad); err == nil {
		t.Fatal("want parse error")
	}
	if _, err := Load(filepath.Join(dir, "nope")); err == nil {
		t.Fatal("want missing error")
	}
}

func TestToolGatingRules(t *testing.T) {
	d, err := Load("../examples/tool-gating/rules.json")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]string{
		`{"tool":"shell","args":{"cmd":"ls"}}`:               "deny",
		`{"tool":"issue_refund","args":{"amount":15000}}`:    "ask_human",
		`{"tool":"issue_refund","args":{"amount":10}}`:       "ask_human",
		`{"tool":"search_kb","args":{"query":"x"}}`:          "allow",
		`{"tool":"db_query","args":{"sql":"DELETE FROM t"}}`: "deny",
		`{"tool":"db_query","args":{"sql":"SELECT 1"}}`:      "allow",
	}
	for in, want := range cases {
		r, _ := d.Decide(context.Background(), q, in)
		if r.Label != want {
			t.Errorf("%s: got %s want %s", in, r.Label, want)
		}
	}
}
