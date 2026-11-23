package dataset

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRead(t *testing.T) {
	in := `// comment
{"id":"1","text":"hola","label":"a"}

{"state":{"tool":"shell"},"label":"deny"}
{"state":"texto","label":"b"}
`
	ex, err := Read(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(ex) != 3 || ex[0].ID != "1" || ex[1].Text != `{"tool":"shell"}` || ex[2].Text != "texto" {
		t.Fatalf("%+v", ex)
	}
	for _, bad := range []string{`{`, `{"text":"x"}`, `{"label":"x"}`} {
		if _, err := Read(strings.NewReader(bad)); err == nil {
			t.Errorf("%s: want error", bad)
		}
	}
}

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	if _, err := Load(filepath.Join(dir, "x")); err == nil {
		t.Fatal("missing")
	}
	p := filepath.Join(dir, "bad.jsonl")
	os.WriteFile(p, []byte("{"), 0o600)
	if _, err := Load(p); err == nil || !strings.Contains(err.Error(), "bad.jsonl") {
		t.Fatal(err)
	}
	ex, err := Load("../examples/tickets-es/synthetic-test.jsonl")
	if err != nil || len(ex) != 100 {
		t.Fatal(len(ex), err)
	}
}
