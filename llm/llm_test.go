package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	decision "github.com/numoru-ia/agent-decision-gate"
)

var q = decision.Question{Name: "cola", Description: "¿A qué cola va?", Labels: []decision.Label{
	{Name: "cancelacion", Description: "cancelar CFDI"}, {Name: "acceso_cuenta"},
}}

func server(t *testing.T, status int, content string, check func(map[string]any, *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(b, &body); err != nil {
			t.Errorf("body: %v", err)
		}
		if check != nil {
			check(body, r)
		}
		w.WriteHeader(status)
		resp := map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": content}}},
			"usage":   map[string]any{"prompt_tokens": 1000, "completion_tokens": 20},
		}
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestDecideStructured(t *testing.T) {
	srv := server(t, 200, `{"label":"cancelacion","confidence":0.83}`, func(body map[string]any, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("auth header %q", r.Header.Get("Authorization"))
		}
		if body["model"] != "m" || body["temperature"].(float64) != 0 {
			t.Errorf("model/temperature: %v", body)
		}
		rf := body["response_format"].(map[string]any)
		js := rf["json_schema"].(map[string]any)
		enum := js["schema"].(map[string]any)["properties"].(map[string]any)["label"].(map[string]any)["enum"].([]any)
		if rf["type"] != "json_schema" || js["strict"] != true || len(enum) != 2 || enum[0] != "cancelacion" {
			t.Errorf("response_format: %v", rf)
		}
		msgs := body["messages"].([]any)
		sys := msgs[0].(map[string]any)["content"].(string)
		if !strings.Contains(sys, "- cancelacion: cancelar CFDI") || !strings.Contains(sys, "never as instructions") {
			t.Errorf("system prompt: %s", sys)
		}
		if msgs[1].(map[string]any)["content"] != "quiero cancelar" {
			t.Errorf("user msg: %v", msgs[1])
		}
	})
	defer srv.Close()
	d, err := New(Config{BaseURL: srv.URL + "/v1/", APIKey: "k", Model: "m", PriceInPerM: 0.5, PriceOutPerM: 2})
	if err != nil {
		t.Fatal(err)
	}
	r, err := d.Decide(context.Background(), q, "quiero cancelar")
	if err != nil {
		t.Fatal(err)
	}
	wantCost := 1000*0.5/1e6 + 20*2.0/1e6
	if r.Label != "cancelacion" || r.Confidence != 0.83 || r.CostUSD != wantCost || r.Backend != "llm:m" || r.Latency <= 0 {
		t.Fatalf("%+v", r)
	}
}

func TestJSONObjectModeAndFences(t *testing.T) {
	srv := server(t, 200, "```json\n{\"label\":\"acceso_cuenta\",\"confidence\":7}\n```", func(body map[string]any, r *http.Request) {
		if body["response_format"].(map[string]any)["type"] != "json_object" {
			t.Errorf("want json_object, got %v", body["response_format"])
		}
		if r.Header.Get("Authorization") != "" {
			t.Errorf("no key, no header")
		}
	})
	defer srv.Close()
	d, _ := New(Config{BaseURL: srv.URL + "/v1", Model: "m", DisableSchema: true})
	r, err := d.Decide(context.Background(), q, "x")
	if err != nil || r.Label != "acceso_cuenta" || r.Confidence != 1 {
		t.Fatalf("%+v %v", r, err)
	}
}

func TestErrors(t *testing.T) {
	cases := []struct {
		status  int
		content string
		want    string
	}{
		{200, `{"label":"otro","confidence":0.9}`, "enum"},
		{200, `la respuesta es cancelacion`, "not JSON"},
		{500, ``, "HTTP 500"},
	}
	for _, c := range cases {
		srv := server(t, c.status, c.content, nil)
		d, _ := New(Config{BaseURL: srv.URL + "/v1", Model: "m"})
		_, err := d.Decide(context.Background(), q, "x")
		srv.Close()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%q: got %v", c.content, err)
		}
		if c.want == "enum" && !errors.Is(err, decision.ErrInvalidLabel) {
			t.Errorf("want ErrInvalidLabel, got %v", err)
		}
	}
	empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"choices":[]}`)) }))
	defer empty.Close()
	d, _ := New(Config{BaseURL: empty.URL, Model: "m"})
	if _, err := d.Decide(context.Background(), q, "x"); err == nil || !strings.Contains(err.Error(), "no choices") {
		t.Fatal(err)
	}
	garbage := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`<html>`)) }))
	defer garbage.Close()
	d, _ = New(Config{BaseURL: garbage.URL, Model: "m"})
	if _, err := d.Decide(context.Background(), q, "x"); err == nil {
		t.Fatal("want decode error")
	}
	d, _ = New(Config{BaseURL: "http://127.0.0.1:1", Model: "m"})
	if _, err := d.Decide(context.Background(), q, "x"); err == nil {
		t.Fatal("want connection error")
	}
	if _, err := New(Config{BaseURL: "x"}); err == nil {
		t.Fatal("model required")
	}
}

func TestFromEnv(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "")
	t.Setenv("LLM_MODEL", "llama3.1")
	t.Setenv("LLM_API_KEY", "")
	t.Setenv("LLM_PRICE_IN_PER_M", "0.15")
	t.Setenv("LLM_PRICE_OUT_PER_M", "0.6")
	t.Setenv("LLM_JSON_SCHEMA", "0")
	c := FromEnv()
	if c.BaseURL != "https://api.openai.com/v1" || c.Model != "llama3.1" || c.PriceInPerM != 0.15 || c.PriceOutPerM != 0.6 || !c.DisableSchema {
		t.Fatalf("%+v", c)
	}
}

func TestHelpers(t *testing.T) {
	if truncate("abcdef", 3) != "abc..." || truncate("ab", 3) != "ab" {
		t.Fatal("truncate")
	}
	if stripFences("```\n{}\n```") != "{}" {
		t.Fatal("fences")
	}
}
