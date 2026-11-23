package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// The committed JSONL files must be exactly what the generator produces.
func TestCommittedFilesAreReproducible(t *testing.T) {
	dir := t.TempDir()
	for _, sub := range []string{"tickets-es", "tool-gating"} {
		os.MkdirAll(filepath.Join(dir, sub), 0o755)
	}
	generate(dir)
	for _, f := range []string{
		"tickets-es/synthetic-train.jsonl", "tickets-es/synthetic-test.jsonl",
		"tool-gating/synthetic-train.jsonl", "tool-gating/synthetic-test.jsonl",
	} {
		want, err := os.ReadFile(filepath.Join("..", f))
		if err != nil {
			t.Fatal(err)
		}
		got, _ := os.ReadFile(filepath.Join(dir, f))
		if !bytes.Equal(got, want) {
			t.Errorf("%s differs from generator output; run: go run ./examples/gen", f)
		}
	}
}

// Bases are unique per label, so the split by base cannot leak a phrasing.
func TestBasesAreUnique(t *testing.T) {
	for label, bs := range bases {
		if len(bs) != 30 {
			t.Errorf("%s: %d bases, want 30", label, len(bs))
		}
		seen := map[string]bool{}
		for _, b := range bs {
			if seen[b] {
				t.Errorf("%s: duplicate base %q", label, b)
			}
			seen[b] = true
		}
	}
}
