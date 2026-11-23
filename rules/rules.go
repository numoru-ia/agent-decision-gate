// Package rules is the deterministic baseline backend. A rule votes for a
// label when all of its condition groups match:
//
//   - any:    at least one keyword appears in the folded text (accent- and case-insensitive);
//   - regex:  at least one regular expression matches the folded text;
//   - fields: every condition on a JSON field of the state holds (numbers,
//     strings, lists and dates). This is where thresholds and date checks
//     belong: they are code, not a model's opinion.
//
// A rule marked final wins outright with confidence 1 (hard deny lists).
// Otherwise votes are summed per label and normalised into probabilities.
package rules

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/internal/textnorm"
)

// Cond is a condition on a field of a JSON state, addressed by dot path
// ("args.amount", "items.0.sku").
//
// Ops: ">", ">=", "<", "<=" (numbers), "==", "!=" (numbers or strings),
// "in", "not_in" (value is a list), "contains" (substring, folded),
// "exists", "older_than_days", "newer_than_days" (field is a date
// YYYY-MM-DD or RFC 3339; value is a number of days).
type Cond struct {
	Field string `json:"field"`
	Op    string `json:"op"`
	Value any    `json:"value,omitempty"`
}

// Rule votes for Label when its condition groups match.
type Rule struct {
	Label  string   `json:"label"`
	Any    []string `json:"any,omitempty"`
	Regex  []string `json:"regex,omitempty"`
	Fields []Cond   `json:"fields,omitempty"`
	Weight float64  `json:"weight,omitempty"`
	Final  bool     `json:"final,omitempty"`
	Note   string   `json:"note,omitempty"`

	any []string
	res []*regexp.Regexp
}

// Config is the JSON file format.
type Config struct {
	// Default is returned with confidence 0 when no rule matches.
	Default string `json:"default"`
	Rules   []Rule `json:"rules"`
}

// Decider is the rules backend.
type Decider struct {
	cfg Config
	// Now is the clock for date conditions.
	Now func() time.Time
}

// New compiles a rule set.
func New(cfg Config) (*Decider, error) {
	for i := range cfg.Rules {
		r := &cfg.Rules[i]
		if r.Label == "" {
			return nil, fmt.Errorf("rule %d: empty label", i)
		}
		if len(r.Any)+len(r.Regex)+len(r.Fields) == 0 {
			return nil, fmt.Errorf("rule %d (%s): no conditions", i, r.Label)
		}
		if r.Weight == 0 {
			r.Weight = 1
		}
		r.any = make([]string, len(r.Any))
		for j, k := range r.Any {
			r.any[j] = textnorm.Fold(k)
		}
		r.res = make([]*regexp.Regexp, len(r.Regex))
		for j, p := range r.Regex {
			re, err := regexp.Compile(p)
			if err != nil {
				return nil, fmt.Errorf("rule %d (%s): %w", i, r.Label, err)
			}
			r.res[j] = re
		}
	}
	return &Decider{cfg: cfg, Now: time.Now}, nil
}

// Load reads and compiles a rules JSON file.
func Load(path string) (*Decider, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return New(cfg)
}

// Name implements decision.Decider.
func (d *Decider) Name() string { return "rules" }

// Decide implements decision.Decider.
func (d *Decider) Decide(ctx context.Context, q decision.Question, state string) (decision.Result, error) {
	start := time.Now()
	folded := textnorm.Fold(state)
	var doc any
	if s := strings.TrimSpace(state); strings.HasPrefix(s, "{") {
		_ = json.Unmarshal([]byte(s), &doc)
	}
	scores := map[string]float64{}
	total := 0.0
	for _, r := range d.cfg.Rules {
		if !q.Has(r.Label) {
			continue
		}
		if !d.match(&r, folded, doc) {
			continue
		}
		if r.Final {
			return decision.Result{
				Label: r.Label, Confidence: 1, Probs: map[string]float64{r.Label: 1},
				Backend: d.Name(), Latency: time.Since(start),
			}, nil
		}
		scores[r.Label] += r.Weight
		total += r.Weight
	}
	if total == 0 {
		def := d.cfg.Default
		if def == "" || !q.Has(def) {
			def = q.Labels[0].Name
		}
		return decision.Result{Label: def, Confidence: 0, Backend: d.Name(), Latency: time.Since(start)}, nil
	}
	for k := range scores {
		scores[k] /= total
	}
	r := decision.FromProbs(d.Name(), scores)
	r.Latency = time.Since(start)
	return r, nil
}

func (d *Decider) match(r *Rule, folded string, doc any) bool {
	if len(r.any) > 0 {
		ok := false
		for _, k := range r.any {
			if strings.Contains(folded, k) {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	if len(r.res) > 0 {
		ok := false
		for _, re := range r.res {
			if re.MatchString(folded) {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	for _, c := range r.Fields {
		if doc == nil || !d.eval(c, doc) {
			return false
		}
	}
	return true
}

// Lookup resolves a dot path inside a decoded JSON document.
func Lookup(doc any, path string) (any, bool) {
	cur := doc
	for _, part := range strings.Split(path, ".") {
		switch v := cur.(type) {
		case map[string]any:
			n, ok := v[part]
			if !ok {
				return nil, false
			}
			cur = n
		case []any:
			i, err := strconv.Atoi(part)
			if err != nil || i < 0 || i >= len(v) {
				return nil, false
			}
			cur = v[i]
		default:
			return nil, false
		}
	}
	return cur, true
}

func (d *Decider) eval(c Cond, doc any) bool {
	v, ok := Lookup(doc, c.Field)
	if c.Op == "exists" {
		return ok
	}
	if !ok {
		return false
	}
	switch c.Op {
	case ">", ">=", "<", "<=":
		a, ok1 := num(v)
		b, ok2 := num(c.Value)
		if !ok1 || !ok2 {
			return false
		}
		switch c.Op {
		case ">":
			return a > b
		case ">=":
			return a >= b
		case "<":
			return a < b
		default:
			return a <= b
		}
	case "==", "!=":
		eq := equal(v, c.Value)
		if c.Op == "==" {
			return eq
		}
		return !eq
	case "in", "not_in":
		list, _ := c.Value.([]any)
		found := false
		for _, x := range list {
			if equal(v, x) {
				found = true
				break
			}
		}
		if c.Op == "in" {
			return found
		}
		return !found
	case "contains":
		s, ok1 := v.(string)
		sub, ok2 := c.Value.(string)
		return ok1 && ok2 && strings.Contains(textnorm.Fold(s), textnorm.Fold(sub))
	case "older_than_days", "newer_than_days":
		s, ok1 := v.(string)
		days, ok2 := num(c.Value)
		if !ok1 || !ok2 {
			return false
		}
		t, err := parseDate(s)
		if err != nil {
			return false
		}
		now := time.Now
		if d.Now != nil {
			now = d.Now
		}
		age := now().Sub(t).Hours() / 24
		if c.Op == "older_than_days" {
			return age > days
		}
		return age < days
	}
	return false
}

func parseDate(s string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	return time.Parse("2006-01-02", s)
}

func num(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case string:
		f, err := strconv.ParseFloat(x, 64)
		return f, err == nil
	}
	return 0, false
}

func equal(a, b any) bool {
	if x, ok := num(a); ok {
		if y, ok := num(b); ok {
			return x == y
		}
	}
	as, ok1 := a.(string)
	bs, ok2 := b.(string)
	if ok1 && ok2 {
		return textnorm.Fold(as) == textnorm.Fold(bs)
	}
	if ab, ok := a.(bool); ok {
		bb, ok := b.(bool)
		return ok && ab == bb
	}
	return false
}
