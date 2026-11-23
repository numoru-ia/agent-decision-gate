// Package httpdecision is a generic adapter for hosted decision APIs. It does
// not hardcode any vendor's schema: the request body is a Go text/template
// and the answer is read from the response with dot paths. Example configs
// live in examples/httpdecision; check them against the vendor's current
// documentation before relying on them.
//
// Template data: .Question (decision.Question), .Labels ([]string), .State
// (string). The template function "json" marshals any value, so strings are
// escaped correctly: {"state": {{json .State}}}.
//
// Header values may reference environment variables as ${NAME}; they are
// expanded at request time so secrets stay out of config files.
package httpdecision

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"text/template"
	"time"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/rules"
)

// Config describes one hosted decision endpoint.
type Config struct {
	Name    string            `json:"name"`
	URL     string            `json:"url"`
	Method  string            `json:"method,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	// Body is a text/template producing the JSON request body.
	Body string `json:"body"`
	// LabelPath is the dot path of the chosen label in the response.
	LabelPath string `json:"label_path"`
	// ConfidencePath is the dot path of the confidence (optional).
	ConfidencePath string `json:"confidence_path,omitempty"`
	// ProbsPath is the dot path of an object label -> probability (optional).
	// When ConfidencePath is empty, confidence = probs[label].
	ProbsPath string `json:"probs_path,omitempty"`
	// CostPerCallUSD is a flat per-decision cost for reporting.
	CostPerCallUSD float64 `json:"cost_per_call_usd,omitempty"`
	TimeoutSeconds float64 `json:"timeout_seconds,omitempty"`
	// Source documents where the schema came from and when it was checked.
	Source string `json:"source,omitempty"`
}

// Decider is the generic HTTP decision backend.
type Decider struct {
	cfg    Config
	tmpl   *template.Template
	client *http.Client
}

// New compiles the template and validates the config.
func New(cfg Config, client *http.Client) (*Decider, error) {
	if cfg.URL == "" || cfg.Body == "" || cfg.LabelPath == "" {
		return nil, errors.New("httpdecision: url, body and label_path are required")
	}
	if cfg.Method == "" {
		cfg.Method = http.MethodPost
	}
	if cfg.Name == "" {
		cfg.Name = "http"
	}
	t, err := template.New(cfg.Name).Funcs(template.FuncMap{
		"json": func(v any) (string, error) {
			b, err := json.Marshal(v)
			return string(b), err
		},
	}).Parse(cfg.Body)
	if err != nil {
		return nil, fmt.Errorf("httpdecision: body template: %w", err)
	}
	if client == nil {
		to := 30 * time.Second
		if cfg.TimeoutSeconds > 0 {
			to = time.Duration(cfg.TimeoutSeconds * float64(time.Second))
		}
		client = &http.Client{Timeout: to}
	}
	return &Decider{cfg: cfg, tmpl: t, client: client}, nil
}

// Load reads a Config from a JSON file.
func Load(path string) (*Decider, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return New(cfg, nil)
}

// Name implements decision.Decider.
func (d *Decider) Name() string { return "http:" + d.cfg.Name }

// Render executes the body template (exported for tests and dry runs).
func (d *Decider) Render(q decision.Question, state string) ([]byte, error) {
	var buf bytes.Buffer
	data := map[string]any{"Question": q, "Labels": q.LabelNames(), "State": state}
	if err := d.tmpl.Execute(&buf, data); err != nil {
		return nil, err
	}
	if !json.Valid(buf.Bytes()) {
		return nil, fmt.Errorf("httpdecision: rendered body is not valid JSON: %s", buf.String())
	}
	return buf.Bytes(), nil
}

// Decide implements decision.Decider.
func (d *Decider) Decide(ctx context.Context, q decision.Question, state string) (decision.Result, error) {
	start := time.Now()
	body, err := d.Render(q, state)
	if err != nil {
		return decision.Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, d.cfg.Method, d.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return decision.Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range d.cfg.Headers {
		req.Header.Set(k, os.ExpandEnv(v))
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return decision.Result{}, fmt.Errorf("httpdecision: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return decision.Result{}, err
	}
	if resp.StatusCode/100 != 2 {
		s := string(raw)
		if len(s) > 300 {
			s = s[:300]
		}
		return decision.Result{}, fmt.Errorf("httpdecision: HTTP %d: %s", resp.StatusCode, s)
	}
	var doc any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return decision.Result{}, fmt.Errorf("httpdecision: decode: %w", err)
	}
	lv, ok := rules.Lookup(doc, d.cfg.LabelPath)
	label, _ := lv.(string)
	if !ok || label == "" {
		return decision.Result{}, fmt.Errorf("httpdecision: no string at %q", d.cfg.LabelPath)
	}
	if !q.Has(label) {
		return decision.Result{}, fmt.Errorf("httpdecision: %w: %q", decision.ErrInvalidLabel, label)
	}
	res := decision.Result{Label: label, Backend: d.Name(), CostUSD: d.cfg.CostPerCallUSD}
	if d.cfg.ProbsPath != "" {
		if pv, ok := rules.Lookup(doc, d.cfg.ProbsPath); ok {
			if pm, ok := pv.(map[string]any); ok {
				res.Probs = map[string]float64{}
				for k, v := range pm {
					if f, ok := toFloat(v); ok {
						res.Probs[k] = f
					}
				}
			}
		}
	}
	switch {
	case d.cfg.ConfidencePath != "":
		cv, ok := rules.Lookup(doc, d.cfg.ConfidencePath)
		f, ok2 := toFloat(cv)
		if !ok || !ok2 {
			return decision.Result{}, fmt.Errorf("httpdecision: no number at %q", d.cfg.ConfidencePath)
		}
		res.Confidence = decision.Clamp01(f)
	case res.Probs != nil:
		res.Confidence = decision.Clamp01(res.Probs[label])
	default:
		res.Confidence = 1
	}
	res.Latency = time.Since(start)
	return res, nil
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		return f, err == nil
	}
	return 0, false
}
