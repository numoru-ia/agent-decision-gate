package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const ex = "../../examples/tickets-es/"
const tg = "../../examples/tool-gating/"

func runCLI(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := run(context.Background(), args, strings.NewReader(stdin), &out, &errb)
	return code, out.String(), errb.String()
}

func TestTrainPredictBench(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m.gz")
	code, out, errs := runCLI(t, "", "train", "-data", ex+"synthetic-train.jsonl", "-out", model)
	if code != 0 || !strings.Contains(out, "trained logreg on 300 examples") {
		t.Fatal(code, out, errs)
	}
	code, out, errs = runCLI(t, "", "predict", "-question", ex+"question.json", "-backend", "local="+model, "olvide", "mi", "contrasena")
	if code != 0 || !strings.Contains(out, `"label":"acceso_cuenta"`) {
		t.Fatal(code, out, errs)
	}
	log := filepath.Join(dir, "audit.jsonl")
	code, out, errs = runCLI(t, "quiero cancelar la factura 123\n\nmi certificado vencio\n",
		"predict", "-question", ex+"question.json", "-gate", "rules="+ex+"rules.json@0.99,local="+model+"@0.99", "-log", log)
	if code != 0 || strings.Count(out, "\n") != 2 {
		t.Fatal(code, out, errs)
	}
	b, _ := os.ReadFile(log)
	if !strings.Contains(string(b), "input_sha256") || strings.Contains(string(b), "vencio") {
		t.Fatalf("audit log: %s", b)
	}
	code, out, errs = runCLI(t, "", "bench", "-question", ex+"question.json", "-test", ex+"synthetic-test.jsonl",
		"-backends", "rules="+ex+"rules.json,local="+model, "-gate", "rules="+ex+"rules.json@0.99,local="+model+"@0.6")
	if code != 0 || !strings.Contains(out, "| rules | 100 |") || !strings.Contains(out, "| local-logreg | 100 |") || !strings.Contains(out, "gate(rules@0.99 > local-logreg@0.6 > human)") {
		t.Fatal(code, out, errs)
	}
	code, out, _ = runCLI(t, "", "bench", "-question", tg+"question.json", "-test", tg+"synthetic-test.jsonl", "-backends", "rules="+tg+"rules.json", "-format", "json")
	var reps []map[string]any
	if code != 0 || json.Unmarshal([]byte(out), &reps) != nil || reps[0]["backend"] != "rules" {
		t.Fatal(code, out)
	}
}

func TestLLMBackendViaEnv(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"content":"{\"label\":\"deny\",\"confidence\":0.9}"}}],"usage":{"prompt_tokens":100,"completion_tokens":10}}`))
	}))
	defer srv.Close()
	t.Setenv("LLM_BASE_URL", srv.URL)
	t.Setenv("LLM_MODEL", "mock")
	t.Setenv("LLM_PRICE_IN_PER_M", "1")
	code, out, errs := runCLI(t, "", "bench", "-question", tg+"question.json", "-test", tg+"synthetic-test.jsonl", "-backends", "llm")
	if code != 0 || !strings.Contains(out, "| llm:mock | 12 |") || !strings.Contains(out, "0.000100") {
		t.Fatal(code, out, errs)
	}
}

func TestErrorsAndUsage(t *testing.T) {
	cases := [][]string{
		{},
		{"nope"},
		{"train"},
		{"train", "-data", "missing.jsonl"},
		{"predict"},
		{"predict", "-question", ex + "question.json"},
		{"predict", "-question", ex + "question.json", "-backend", "bogus"},
		{"predict", "-question", ex + "question.json", "-gate", "local=x.gz"},
		{"predict", "-question", ex + "question.json", "-gate", "rules=" + ex + "rules.json@abc"},
		{"predict", "-question", ex + "question.json", "-gate", " , "},
		{"bench"},
		{"bench", "-question", ex + "question.json", "-test", tg + "synthetic-test.jsonl", "-backends", "rules=" + ex + "rules.json"},
		{"bench", "-question", ex + "question.json", "-test", ex + "synthetic-test.jsonl", "-backends", "rules"},
		{"bench", "-question", ex + "question.json", "-test", ex + "synthetic-test.jsonl", "-backends", "local"},
		{"bench", "-question", ex + "question.json", "-test", ex + "synthetic-test.jsonl", "-backends", "http"},
		{"bench", "-question", ex + "question.json", "-test", ex + "synthetic-test.jsonl", "-gate", "llm"},
		{"train", "-bogus"},
	}
	for _, c := range cases {
		if code, _, _ := runCLI(t, "", c...); code == 0 {
			t.Errorf("%v: want non-zero exit", c)
		}
	}
	if code, out, _ := runCLI(t, "", "help"); code != 0 || !strings.Contains(out, "usage") {
		t.Fatal("help")
	}
	if code, _, _ := runCLI(t, "", "train", "-h"); code != 0 {
		t.Fatal("-h exits 0")
	}
}

func TestBuildHTTPBackend(t *testing.T) {
	if _, err := buildBackend("http=../../examples/httpdecision/generic.json"); err != nil {
		t.Fatal(err)
	}
}
