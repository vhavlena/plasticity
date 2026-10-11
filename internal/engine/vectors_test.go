package engine

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/model"
)

// conceptEmbedder is a deterministic fake: each dimension is a concept, and
// a text's vector counts the words belonging to each concept. Texts about
// the same concept are similar even without shared words. name makes
// distinct instances look like distinct models.
type conceptEmbedder struct{ name string }

var concepts = [][]string{
	{"sqlite", "database", "storage", "engine", "badger", "db", "wal", "persistence"},
	{"process", "processes", "concurrent", "parallel", "writer", "writers", "readers", "lock", "simultaneous"},
	{"pasta", "cooking", "boil", "water", "salt", "kitchen", "food", "meal"},
	{"hebbian", "learning", "neurons", "wire", "fire", "plasticity"},
	{"spreading", "activation", "flows", "edges", "propagate"},
}

func (c conceptEmbedder) ID() string {
	if c.name == "" {
		return "fake:concepts"
	}
	return "fake:" + c.name
}
func (conceptEmbedder) Close() error { return nil }
func (conceptEmbedder) Embed(_ context.Context, text string, _ embed.Kind) ([]float32, error) {
	v := make([]float32, len(concepts))
	for _, w := range wordRE.FindAllString(strings.ToLower(text), -1) {
		for d, words := range concepts {
			for _, c := range words {
				if w == c {
					v[d]++
				}
			}
		}
	}
	return embed.Normalize(v), nil
}

// counting wraps an embedder and counts calls (to observe re-indexing).
type counting struct {
	embed.Embedder
	calls *int
}

func (c counting) Embed(ctx context.Context, text string, k embed.Kind) ([]float32, error) {
	*c.calls++
	return c.Embedder.Embed(ctx, text, k)
}

func init() {
	// A custom provider plugged in through the registry, as an experiment
	// would do. Options["name"] selects the model variant.
	embed.Register("concepts", embed.Provider{
		Factory: func(_ context.Context, c embed.Config) (embed.Embedder, error) {
			return conceptEmbedder{name: c.Options["name"]}, nil
		},
	})
}

func TestSemanticSeedWithoutSharedWords(t *testing.T) {
	f := newFixture(t, WithEmbedder(conceptEmbedder{}))
	r := f.recall("kitchen meal", "")
	h, ok := hitIDs(r)[f.ids["pasta"]]
	if !ok || !h.IsSeed || h.Similarity == nil || *h.Similarity < 0.99 {
		t.Fatalf("pasta should be a semantic seed: %+v", r)
	}
	if r.LexicalConfidence != 0 || r.SemanticConfidence != 1 || r.Confidence != 1 {
		t.Fatalf("confidences: lexical %g semantic %g overall %g", r.LexicalConfidence, r.SemanticConfidence, r.Confidence)
	}
	if _, ok := hitIDs(r)[f.ids["sqlite"]]; ok {
		t.Fatal("unrelated node seeded")
	}
}

// The same database compared with and without embeddings (A/B).
func TestWithAndWithoutEmbeddings(t *testing.T) {
	f := newFixture(t, WithEmbedder(conceptEmbedder{}))
	lexical, err := Open(f.path, WithoutEmbeddings())
	if err != nil {
		t.Fatal(err)
	}
	defer lexical.Close()

	a, _ := lexical.Recall(RecallRequest{Query: "hebbian plasticity", NoRecord: true})
	b, _ := f.e.Recall(RecallRequest{Query: "hebbian plasticity", NoRecord: true})
	la, lb := hitIDs(a)[f.ids["hebb"]], hitIDs(b)[f.ids["hebb"]]
	if a.SemanticConfidence != 0 || la.Similarity != nil {
		t.Fatalf("WithoutEmbeddings still used vectors: %+v", a)
	}
	// noisy-OR: 1 − (1 − L)(1 − S) with S = 1
	if !(lb.Activation > la.Activation) || lb.Activation != 1 {
		t.Fatalf("fused seed %g should exceed lexical %g", lb.Activation, la.Activation)
	}
	if r, _ := lexical.Recall(RecallRequest{Query: "kitchen meal", NoRecord: true}); len(r.Hits) != 0 {
		t.Fatalf("lexical-only engine found a paraphrase: %+v", r.Hits)
	}
	if st, _ := lexical.EmbedStatus(); st.Active != "" || st.Models["fake:concepts"] != len(f.ids) {
		t.Fatalf("status without embeddings: %+v", st)
	}
}

func TestUnavailableEmbedderFallsBackToLexical(t *testing.T) {
	down := embed.NewOllama(embed.Config{Provider: embed.ProviderOllama, Model: "m", Host: "http://127.0.0.1:1"})
	f := newFixture(t, WithEmbedder(down))
	if len(f.e.Warnings) != len(f.ids) {
		t.Fatalf("each add should warn about the missing vector: %v", f.e.Warnings)
	}
	f.e.Warnings = nil
	r, err := f.e.Recall(RecallRequest{Query: "sqlite", NoRecord: true})
	if err != nil || len(r.Hits) == 0 || r.Hits[0].Node.ID != f.ids["sqlite"] {
		t.Fatalf("lexical fallback failed: %v %+v", err, r)
	}
	if len(f.e.Warnings) != 1 || !strings.Contains(f.e.Warnings[0], "semantic search disabled") {
		t.Fatalf("expected a warning, got %v", f.e.Warnings)
	}
	if st, _ := f.e.EmbedStatus(); st.Missing != len(f.ids) || st.Indexed != 0 {
		t.Fatalf("nodes should be stored without vectors: %+v", st)
	}
}

func TestUpdateReembedsAndReindexIsIncremental(t *testing.T) {
	calls := 0
	f := newFixture(t, WithEmbedder(counting{conceptEmbedder{}, &calls}))
	if calls != len(f.ids) {
		t.Fatalf("adds should embed each node once: %d", calls)
	}
	if rep, _ := f.e.Reindex(context.Background(), false); rep.Embedded != 0 || rep.UpToDate != len(f.ids) {
		t.Fatalf("reindex should be a no-op: %+v", rep)
	}
	importance := 0.9
	if _, err := f.e.UpdateNode(f.ids["pasta"], NodePatch{Importance: &importance}); err != nil || calls != len(f.ids) {
		t.Fatalf("a non-text update must not re-embed (calls=%d): %v", calls, err)
	}
	content := "parallel writers and simultaneous readers"
	if _, err := f.e.UpdateNode(f.ids["pasta"], NodePatch{Content: &content}); err != nil {
		t.Fatal(err)
	}
	r, _ := f.e.Recall(RecallRequest{Query: "concurrent lock", NoRecord: true})
	if h, ok := hitIDs(r)[f.ids["pasta"]]; !ok || h.Similarity == nil {
		t.Fatalf("updated node not re-embedded: %+v", r.Hits)
	}
	if rep, _ := f.e.Reindex(context.Background(), true); rep.Embedded != len(f.ids) {
		t.Fatalf("full reindex: %+v", rep)
	}
}

// A registered provider is selected by configuration, persists across
// reopening, and several embedders are indexed side by side.
func TestRegisteredProvidersSideBySide(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	use := func(name string) *ReindexReport {
		rep, err := f.e.EnableEmbedding(ctx, embed.Config{Provider: "concepts", Model: name, Options: map[string]string{"name": name}})
		if err != nil {
			t.Fatal(err)
		}
		return rep
	}
	if rep := use("v1"); rep.Embedded != len(f.ids) {
		t.Fatalf("enable v1: %+v", rep)
	}
	if rep := use("v2"); rep.Embedded != len(f.ids) {
		t.Fatalf("enable v2: %+v", rep)
	}
	if rep := use("v1"); rep.Embedded != 0 || rep.UpToDate != len(f.ids) {
		t.Fatalf("switching back to v1 must not re-embed: %+v", rep)
	}
	f.e.Close()

	e, err := Open(f.path)
	if err != nil {
		t.Fatal(err)
	}
	f.e = e
	st, _ := e.EmbedStatus()
	if st.Active != "fake:v1" || st.Config.Options["name"] != "v1" || len(st.Models) != 2 {
		t.Fatalf("status after reopen: %+v", st)
	}
	if r, _ := e.Recall(RecallRequest{Query: "kitchen meal", NoRecord: true}); len(r.Hits) == 0 {
		t.Fatal("configured provider not used after reopen")
	}
	if n, _ := e.PruneVectors(); n != int64(len(f.ids)) {
		t.Fatalf("prune removed %d vectors", n)
	}
	if err := e.DisableEmbedding(); err != nil {
		t.Fatal(err)
	}
	if st, _ := e.EmbedStatus(); st.Active != "" || st.Config != nil || st.Models["fake:v1"] != len(f.ids) {
		t.Fatalf("disable should keep vectors: %+v", st)
	}
	if _, err := e.EnableEmbedding(ctx, embed.Config{Provider: "nope"}); err == nil || !strings.Contains(err.Error(), "concepts") {
		t.Fatalf("unknown provider error should list registered ones: %v", err)
	}
}

// EnableEmbedding with the built-in Ollama provider against a mock server.
func TestEnableOllama(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Input string }
		_ = json.NewDecoder(r.Body).Decode(&req)
		v, _ := conceptEmbedder{}.Embed(r.Context(), req.Input, embed.Document)
		if strings.Contains(req.Input, "probe") {
			v = []float32{1, 0, 0, 0, 0}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": [][]float32{v}})
	}))
	defer srv.Close()

	e, err := Open(filepath.Join(t.TempDir(), "g.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, err := e.AddNode(&model.Node{Type: model.TypeProcedure, Title: "Cooking pasta", Content: "boil water"}); err != nil {
		t.Fatal(err)
	}
	rep, err := e.EnableEmbedding(context.Background(), embed.Config{Provider: embed.ProviderOllama, Model: "m", Host: srv.URL})
	if err != nil || rep.Embedded != 1 {
		t.Fatalf("enable: %v %+v", err, rep)
	}
	if r, err := e.Recall(RecallRequest{Query: "kitchen meal", NoRecord: true}); err != nil || len(r.Hits) != 1 {
		t.Fatalf("semantic recall via ollama: %v %+v", err, r)
	}
}
