package decision

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"time"
)

// Stage is one rung of the escalation ladder: a backend and the minimum
// confidence at which its answer is accepted.
type Stage struct {
	Decider       Decider
	MinConfidence float64
}

// Gate runs stages in order and returns the first answer whose confidence
// reaches its stage threshold. If no stage is confident (or all fail), the
// decision goes to a human: the outcome carries HumanLabel and NeedsHuman.
// Every stage attempt is written to Log as one JSON line for auditing.
type Gate struct {
	Stages []Stage
	// HumanLabel is the label returned when every stage abstains.
	// Defaults to "human_review".
	HumanLabel string
	// Log receives one JSON object per line. Nil disables logging.
	Log io.Writer
	// Now is the clock used for log timestamps (tests override it).
	Now func() time.Time

	mu sync.Mutex
}

// Outcome is the gate's final answer plus the trail of stage results.
type Outcome struct {
	Result
	NeedsHuman bool     `json:"needs_human"`
	Stage      int      `json:"stage"`
	Trail      []Result `json:"trail"`
}

// LogEntry is the audit record written for each stage attempt and for the
// final human escalation. The input is stored only as a SHA-256 hash.
type LogEntry struct {
	Time       time.Time `json:"ts"`
	Question   string    `json:"question"`
	InputHash  string    `json:"input_sha256"`
	Stage      int       `json:"stage"`
	Backend    string    `json:"backend"`
	Label      string    `json:"label,omitempty"`
	Confidence float64   `json:"confidence"`
	Threshold  float64   `json:"threshold"`
	LatencyMS  float64   `json:"latency_ms"`
	CostUSD    float64   `json:"cost_usd,omitempty"`
	Accepted   bool      `json:"accepted"`
	Error      string    `json:"error,omitempty"`
}

// HashInput returns the hex SHA-256 of the state, used in audit logs so the
// log itself does not store ticket text or tool arguments.
func HashInput(state string) string {
	h := sha256.Sum256([]byte(state))
	return hex.EncodeToString(h[:])
}

// ErrNoStages is returned by a Gate with nothing to run.
var ErrNoStages = errors.New("gate: no stages configured")

// Decide runs the escalation ladder for one decision.
func (g *Gate) Decide(ctx context.Context, q Question, state string) (Outcome, error) {
	if len(g.Stages) == 0 {
		return Outcome{}, ErrNoStages
	}
	hash := HashInput(state)
	var out Outcome
	for i, st := range g.Stages {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		start := time.Now()
		r, err := st.Decider.Decide(ctx, q, state)
		if r.Latency == 0 {
			r.Latency = time.Since(start)
		}
		if r.Backend == "" {
			r.Backend = st.Decider.Name()
		}
		if err == nil && !q.Has(r.Label) {
			err = ErrInvalidLabel
		}
		accepted := err == nil && r.Confidence >= st.MinConfidence
		e := LogEntry{
			Question: q.Name, InputHash: hash, Stage: i, Backend: r.Backend,
			Label: r.Label, Confidence: r.Confidence, Threshold: st.MinConfidence,
			LatencyMS: float64(r.Latency.Microseconds()) / 1000, CostUSD: r.CostUSD,
			Accepted: accepted,
		}
		if err != nil {
			e.Error = err.Error()
			e.Label = ""
		} else {
			out.Trail = append(out.Trail, r)
		}
		g.write(e)
		out.CostUSD += r.CostUSD
		if accepted {
			cost := out.CostUSD
			out.Result = r
			out.CostUSD = cost
			out.Stage = i
			return out, nil
		}
	}
	human := g.HumanLabel
	if human == "" {
		human = "human_review"
	}
	cost := out.CostUSD
	out.Result = Result{Label: human, Backend: "human"}
	out.CostUSD = cost
	out.NeedsHuman = true
	out.Stage = len(g.Stages)
	g.write(LogEntry{Question: q.Name, InputHash: hash, Stage: out.Stage, Backend: "human", Label: human, Accepted: true})
	return out, nil
}

func (g *Gate) write(e LogEntry) {
	if g.Log == nil {
		return
	}
	now := time.Now
	if g.Now != nil {
		now = g.Now
	}
	e.Time = now().UTC()
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	_, _ = g.Log.Write(append(b, '\n'))
}

// AsDecider exposes a Gate as a Decider for benchmarking. An accepted answer
// has confidence 1; an escalation to a human has confidence 0 and carries
// HumanLabel, which counts as "not automated" in bench reports.
func (g *Gate) AsDecider(name string) Decider { return gateDecider{g: g, name: name} }

type gateDecider struct {
	g    *Gate
	name string
}

func (d gateDecider) Name() string { return d.name }

func (d gateDecider) Decide(ctx context.Context, q Question, state string) (Result, error) {
	start := time.Now()
	o, err := d.g.Decide(ctx, q, state)
	if err != nil {
		return Result{}, err
	}
	r := Result{Label: o.Label, Confidence: 1, Backend: d.name, Latency: time.Since(start), CostUSD: o.CostUSD}
	if o.NeedsHuman {
		r.Confidence = 0
	}
	return r, nil
}
