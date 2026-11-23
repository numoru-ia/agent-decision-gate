package httpdecision

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	decision "github.com/numoru-ia/agent-decision-gate"
)

var q = decision.Question{Name: "cola", Description: "¿Qué cola?", Labels: []decision.Label{
	{Name: "cancelacion", Description: "cancelar \"CFDI\""}, {Name: "acceso_cuenta", Description: "login"},
}}

func loadExample(t *testing.T, path, url string) *Decider {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.URL = url
	d, err := New(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// The Jev example config renders the request shape shown in the public
// quickstart and reads answers.<id>.choice / confidence / probabilities.
func TestJevExampleConfig(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "secret")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("auth %q", r.Header.Get("Authorization"))
		}
		b, _ := io.ReadAll(r.Body)
		var body struct {
			Model     string `json:"model"`
			State     string `json:"state"`
			Questions map[string]struct {
				Type     string            `json:"type"`
				Criteria map[string]string `json:"criteria"`
			} `json:"questions"`
		}
		if err := json.Unmarshal(b, &body); err != nil {
			t.Fatalf("body %s: %v", b, err)
		}
		if body.State != "quiero \"cancelar\"\nya" || body.Questions["q"].Type != "choice" || body.Questions["q"].Criteria["cancelacion"] != "cancelar \"CFDI\"" {
			t.Errorf("rendered body: %s", b)
		}
		w.Write([]byte(`{"model":"jev-1.14.0","answers":{"q":{"type":"choice","choice":"cancelacion","confidence":0.91,"probabilities":{"cancelacion":0.91,"acceso_cuenta":0.09}}},"usage":{"input_tokens":120,"output_tokens":5}}`))
	}))
	defer srv.Close()
	d := loadExample(t, "../examples/httpdecision/jev-choice.json", srv.URL)
	r, err := d.Decide(context.Background(), q, "quiero \"cancelar\"\nya")
	if err != nil {
		t.Fatal(err)
	}
	if r.Label != "cancelacion" || r.Confidence != 0.91 || r.Probs["acceso_cuenta"] != 0.09 || r.Backend != "http:jev" {
		t.Fatalf("%+v", r)
	}
}

func TestGenericExampleAndProbsOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"label":"acceso_cuenta","confidence":"0.7","out":{"p":{"acceso_cuenta":0.6}}}`))
	}))
	defer srv.Close()
	d := loadExample(t, "../examples/httpdecision/generic.json", srv.URL)
	r, err := d.Decide(context.Background(), q, "login")
	if err != nil || r.Confidence != 0.7 {
		t.Fatalf("%+v %v", r, err)
	}
	d2, _ := New(Config{URL: srv.URL, Body: `{}`, LabelPath: "label", ProbsPath: "out.p"}, nil)
	r, _ = d2.Decide(context.Background(), q, "x")
	if r.Confidence != 0.6 {
		t.Fatalf("probs-only confidence: %+v", r)
	}
	d3, _ := New(Config{URL: srv.URL, Body: `{}`, LabelPath: "label"}, nil)
	r, _ = d3.Decide(context.Background(), q, "x")
	if r.Confidence != 1 || d3.Name() != "http:http" {
		t.Fatalf("no confidence path: %+v", r)
	}
}

func TestErrors(t *testing.T) {
	respond := func(status int, body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
			w.Write([]byte(body))
		}))
	}
	cases := []struct {
		status int
		body   string
		cfg    Config
		want   string
	}{
		{500, strings.Repeat("x", 400), Config{Body: `{}`, LabelPath: "label"}, "HTTP 500"},
		{200, `nope`, Config{Body: `{}`, LabelPath: "label"}, "decode"},
		{200, `{"x":1}`, Config{Body: `{}`, LabelPath: "label"}, "no string"},
		{200, `{"label":"otra"}`, Config{Body: `{}`, LabelPath: "label"}, "enum"},
		{200, `{"label":"cancelacion"}`, Config{Body: `{}`, LabelPath: "label", ConfidencePath: "c"}, "no number"},
		{200, `{}`, Config{Body: `{"s": {{.State}}}`, LabelPath: "label"}, "not valid JSON"},
		{200, `{}`, Config{Body: `{{.Missing.Field}}`, LabelPath: "label"}, ""},
	}
	for i, c := range cases {
		srv := respond(c.status, c.body)
		c.cfg.URL = srv.URL
		d, err := New(c.cfg, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = d.Decide(context.Background(), q, "plain text")
		srv.Close()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("case %d: got %v", i, err)
		}
		if c.want == "enum" && !errors.Is(err, decision.ErrInvalidLabel) {
			t.Errorf("case %d: want ErrInvalidLabel", i)
		}
	}
	if _, err := New(Config{URL: "x"}, nil); err == nil {
		t.Fatal("required fields")
	}
	if _, err := New(Config{URL: "x", Body: "{{", LabelPath: "l"}, nil); err == nil {
		t.Fatal("bad template")
	}
	d, _ := New(Config{URL: "http://127.0.0.1:1", Body: "{}", LabelPath: "l", TimeoutSeconds: 1}, nil)
	if _, err := d.Decide(context.Background(), q, "x"); err == nil {
		t.Fatal("connection error")
	}
	d, _ = New(Config{URL: "::bad", Body: "{}", LabelPath: "l"}, nil)
	if _, err := d.Decide(context.Background(), q, "x"); err == nil {
		t.Fatal("bad url")
	}
}

func TestLoad(t *testing.T) {
	if _, err := Load("../examples/httpdecision/jev-choice.json"); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.json")
	os.WriteFile(p, []byte("{"), 0o600)
	if _, err := Load(p); err == nil {
		t.Fatal("parse")
	}
	if _, err := Load(filepath.Join(dir, "none")); err == nil {
		t.Fatal("missing")
	}
}
