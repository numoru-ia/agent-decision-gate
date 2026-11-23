// Command decide trains, runs and benchmarks decision backends.
//
//	decide train   -data train.jsonl -out model.gz [-algo logreg|nb]
//	decide predict -question q.json -backend local=model.gz "texto del ticket"
//	decide predict -question q.json -gate "rules=r.json@0.99,local=model.gz@0.7,llm@0.6" -log audit.jsonl < tickets.txt
//	decide bench   -question q.json -test test.jsonl -backends "rules=r.json,local=model.gz,llm"
//
// Backend specs: rules=<rules.json>, local=<model.gz>, llm (configured by
// LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_PRICE_IN_PER_M, LLM_PRICE_OUT_PER_M)
// and http=<config.json>.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	decision "github.com/numoru-ia/agent-decision-gate"
	"github.com/numoru-ia/agent-decision-gate/bench"
	"github.com/numoru-ia/agent-decision-gate/dataset"
	"github.com/numoru-ia/agent-decision-gate/httpdecision"
	"github.com/numoru-ia/agent-decision-gate/llm"
	"github.com/numoru-ia/agent-decision-gate/local"
	"github.com/numoru-ia/agent-decision-gate/rules"
)

func main() {
	os.Exit(run(context.Background(), os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

const usage = `usage: decide <train|predict|bench> [flags]

  train    fit a local model from labelled JSONL
  predict  decide one input (args) or one input per stdin line
  bench    evaluate backends on a labelled JSONL test set

Run "decide <command> -h" for flags.
`

func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	var err error
	switch args[0] {
	case "train":
		err = cmdTrain(args[1:], stdout, stderr)
	case "predict":
		err = cmdPredict(ctx, args[1:], stdin, stdout, stderr)
	case "bench":
		err = cmdBench(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usage)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n%s", args[0], usage)
		return 2
	}
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		fmt.Fprintln(stderr, "decide:", err)
		return 1
	}
	return 0
}

func cmdTrain(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	data := fs.String("data", "", "training JSONL (text,label)")
	out := fs.String("out", "model.gz", "output model file")
	algo := fs.String("algo", local.LogReg, "logreg or nb")
	epochs := fs.Int("epochs", 0, "logreg epochs (default 40)")
	seed := fs.Int64("seed", 1, "shuffle seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *data == "" {
		return errors.New("train: -data is required")
	}
	ex, err := dataset.Load(*data)
	if err != nil {
		return err
	}
	tr := make([]local.Example, len(ex))
	for i, e := range ex {
		tr[i] = local.Example{Text: e.Text, Label: e.Label}
	}
	m, err := local.Train(tr, local.Options{Algo: *algo, Epochs: *epochs, Seed: *seed})
	if err != nil {
		return err
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "trained %s on %d examples, %d labels, %d features -> %s\n", m.Algo, len(ex), len(m.Labels), len(m.Vocab), *out)
	return nil
}

// buildBackend turns "kind=arg" into a Decider.
func buildBackend(spec string) (decision.Decider, error) {
	kind, arg, _ := strings.Cut(strings.TrimSpace(spec), "=")
	switch kind {
	case "rules":
		if arg == "" {
			return nil, errors.New("rules backend needs a file: rules=path.json")
		}
		return rules.Load(arg)
	case "local":
		if arg == "" {
			return nil, errors.New("local backend needs a model: local=model.gz")
		}
		m, err := local.Load(arg)
		if err != nil {
			return nil, err
		}
		return local.Decider{M: m}, nil
	case "llm":
		return llm.New(llm.FromEnv())
	case "http":
		if arg == "" {
			return nil, errors.New("http backend needs a config: http=config.json")
		}
		return httpdecision.Load(arg)
	}
	return nil, fmt.Errorf("unknown backend %q (want rules, local, llm or http)", kind)
}

// parseGate turns "rules=r.json@0.99,local=m.gz@0.7,llm@0.6" into stages.
func parseGate(spec string) ([]decision.Stage, error) {
	var st []decision.Stage
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		be, th, ok := strings.Cut(part, "@")
		if !ok {
			return nil, fmt.Errorf("gate stage %q: missing @threshold", part)
		}
		t, err := strconv.ParseFloat(th, 64)
		if err != nil {
			return nil, fmt.Errorf("gate stage %q: %w", part, err)
		}
		d, err := buildBackend(be)
		if err != nil {
			return nil, err
		}
		st = append(st, decision.Stage{Decider: d, MinConfidence: t})
	}
	if len(st) == 0 {
		return nil, errors.New("empty gate")
	}
	return st, nil
}

func cmdPredict(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	fs.SetOutput(stderr)
	qpath := fs.String("question", "", "question JSON (name, labels)")
	backend := fs.String("backend", "", "single backend spec, e.g. local=model.gz")
	gateSpec := fs.String("gate", "", `escalation ladder, e.g. "local=m.gz@0.7,llm@0.6"`)
	logPath := fs.String("log", "", "append audit JSON lines here (gate mode)")
	human := fs.String("human-label", "human_review", "label when every stage abstains")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *qpath == "" {
		return errors.New("predict: -question is required")
	}
	q, err := decision.LoadQuestion(*qpath)
	if err != nil {
		return err
	}
	if (*backend == "") == (*gateSpec == "") {
		return errors.New("predict: set exactly one of -backend or -gate")
	}
	var decide func(string) (any, error)
	if *backend != "" {
		d, err := buildBackend(*backend)
		if err != nil {
			return err
		}
		decide = func(s string) (any, error) { return d.Decide(ctx, q, s) }
	} else {
		stages, err := parseGate(*gateSpec)
		if err != nil {
			return err
		}
		g := &decision.Gate{Stages: stages, HumanLabel: *human}
		if *logPath != "" {
			f, err := os.OpenFile(*logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
			if err != nil {
				return err
			}
			defer f.Close()
			g.Log = f
		}
		decide = func(s string) (any, error) { return g.Decide(ctx, q, s) }
	}
	enc := json.NewEncoder(stdout)
	one := func(s string) error {
		r, err := decide(s)
		if err != nil {
			return err
		}
		return enc.Encode(r)
	}
	if fs.NArg() > 0 {
		return one(strings.Join(fs.Args(), " "))
	}
	sc := bufio.NewScanner(stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" {
			if err := one(s); err != nil {
				return err
			}
		}
	}
	return sc.Err()
}

func cmdBench(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	qpath := fs.String("question", "", "question JSON")
	test := fs.String("test", "", "labelled test JSONL")
	backends := fs.String("backends", "", `comma-separated specs, e.g. "rules=r.json,local=m.gz,llm"`)
	threshold := fs.Float64("threshold", 0.7, "confidence threshold for coverage/acc columns")
	format := fs.String("format", "md", "md or json")
	var gates multiFlag
	fs.Var(&gates, "gate", `also bench an escalation ladder, e.g. "rules=r.json@0.9,local=m.gz@0.8" (repeatable)`)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *qpath == "" || *test == "" || (*backends == "" && len(gates) == 0) {
		return errors.New("bench: -question, -test and -backends (or -gate) are required")
	}
	q, err := decision.LoadQuestion(*qpath)
	if err != nil {
		return err
	}
	ex, err := dataset.Load(*test)
	if err != nil {
		return err
	}
	for _, e := range ex {
		if !q.Has(e.Label) {
			return fmt.Errorf("bench: test label %q is not in question %q", e.Label, q.Name)
		}
	}
	var reps []bench.Report
	for _, spec := range strings.Split(*backends, ",") {
		if strings.TrimSpace(spec) == "" {
			continue
		}
		d, err := buildBackend(spec)
		if err != nil {
			return err
		}
		reps = append(reps, bench.Run(ctx, d, q, ex, *threshold))
	}
	for _, spec := range gates {
		stages, err := parseGate(spec)
		if err != nil {
			return err
		}
		names := make([]string, len(stages))
		for i, st := range stages {
			names[i] = fmt.Sprintf("%s@%g", st.Decider.Name(), st.MinConfidence)
		}
		g := &decision.Gate{Stages: stages}
		reps = append(reps, bench.Run(ctx, g.AsDecider("gate("+strings.Join(names, " > ")+" > human)"), q, ex, *threshold))
	}
	if *format == "json" {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(reps)
	}
	bench.WriteMarkdown(stdout, reps)
	return nil
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ";") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }
