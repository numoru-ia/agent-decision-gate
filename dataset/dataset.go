// Package dataset reads labelled JSONL: one {"text": "...", "label": "..."}
// object per line. "state" is accepted as an alias of "text", and a JSON
// object in "state" is re-encoded as a string, so tool-call examples can be
// written as real JSON.
package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// Example is one labelled decision.
type Example struct {
	ID    string `json:"id,omitempty"`
	Text  string `json:"text"`
	Label string `json:"label"`
}

type line struct {
	ID    string          `json:"id"`
	Text  string          `json:"text"`
	State json.RawMessage `json:"state"`
	Label string          `json:"label"`
}

// Read parses JSONL from r. Blank lines and lines starting with // are skipped.
func Read(r io.Reader) ([]Example, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	var out []Example
	n := 0
	for sc.Scan() {
		n++
		s := strings.TrimSpace(sc.Text())
		if s == "" || strings.HasPrefix(s, "//") {
			continue
		}
		var l line
		if err := json.Unmarshal([]byte(s), &l); err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		text := l.Text
		if text == "" && len(l.State) > 0 {
			var str string
			if json.Unmarshal(l.State, &str) == nil {
				text = str
			} else {
				text = string(l.State)
			}
		}
		if text == "" || l.Label == "" {
			return nil, fmt.Errorf("line %d: text/state and label are required", n)
		}
		out = append(out, Example{ID: l.ID, Text: text, Label: l.Label})
	}
	return out, sc.Err()
}

// Load reads a JSONL file.
func Load(path string) ([]Example, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	ex, err := Read(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return ex, nil
}
