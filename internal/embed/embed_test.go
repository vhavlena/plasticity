package embed

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestEncodeDecodeNormalize(t *testing.T) {
	v := Normalize([]float32{3, 4})
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("normalize: %v", v)
	}
	w, err := Decode(Encode(v))
	if err != nil || w[0] != v[0] || w[1] != v[1] {
		t.Fatalf("round trip: %v %v", w, err)
	}
	if _, err := Decode([]byte{1, 2, 3}); err == nil {
		t.Fatal("odd blob accepted")
	}
	if Dot(v, v) < 0.9999 || Dot(v, []float32{1}) != 0 {
		t.Fatal("dot")
	}
	if got := Truncate("a b  c d", 2); got != "a b" {
		t.Fatalf("truncate: %q", got)
	}
}

func TestDefaults(t *testing.T) {
	c := Config{Provider: ProviderOllama}.WithDefaults()
	if c.Model != DefaultOllamaModel || c.Host != DefaultOllamaHost || c.QueryPrefix != "search_query: " {
		t.Fatalf("ollama defaults: %+v", c)
	}
	if h := (Config{Provider: ProviderHugot}).WithDefaults(); h.Model != DefaultHugotModel || h.ID() != "hugot:"+DefaultHugotModel {
		t.Fatalf("hugot defaults: %+v", h)
	}
	if _, err := New(context.Background(), Config{Provider: "nope"}); err == nil {
		t.Fatal("unknown provider accepted")
	}
}

func TestOllama(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": [][]float32{{0, 2, 0}}})
	}))
	defer srv.Close()
	e, err := New(context.Background(), Config{Provider: ProviderOllama, Host: srv.URL + "/"})
	if err != nil {
		t.Fatal(err)
	}
	v, err := e.Embed(context.Background(), "hello", Query)
	if err != nil || v[1] != 1 {
		t.Fatalf("embed: %v %v", v, err)
	}
	if got["model"] != DefaultOllamaModel || got["input"] != "search_query: hello" {
		t.Fatalf("request: %v", got)
	}
	if _, err := e.Embed(context.Background(), "x", Document); err != nil || got["input"] != "search_document: x" {
		t.Fatalf("document prefix: %v %v", got, err)
	}

	srv.Close()
	if _, err := e.Embed(context.Background(), "x", Query); err == nil {
		t.Fatal("unreachable server must fail")
	}
}

// TestHugotModel runs the real model when PLASTICITY_TEST_MODEL points to a
// downloaded all-MiniLM-L6-v2 directory.
func TestHugotModel(t *testing.T) {
	dir := os.Getenv("PLASTICITY_TEST_MODEL")
	if dir == "" {
		t.Skip("set PLASTICITY_TEST_MODEL to a hugot model directory")
	}
	if raceEnabled {
		t.Skip("gomlx's unsafe matmul packing trips checkptr under -race (see race_test.go)")
	}
	ctx := context.Background()
	e, err := New(ctx, Config{Provider: ProviderHugot, Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	emb := func(s string) []float32 {
		v, err := e.Embed(ctx, s, Document)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	q := emb("which database engine supports several processes at the same time")
	sqlite := emb("Use SQLite WAL mode: multiple processes can read concurrently while one writes")
	pasta := emb("Cooking pasta: boil water and add salt")
	if len(q) != 384 || !(Dot(q, sqlite) > Dot(q, pasta)+0.2) {
		t.Fatalf("semantic ordering: sqlite %.3f pasta %.3f", Dot(q, sqlite), Dot(q, pasta))
	}
	for _, s := range []string{"a", "two words", "five words are in here"} { // odd batch-free sizes
		emb(s)
	}
}
