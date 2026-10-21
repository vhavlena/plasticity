package eval

import (
	"bytes"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestScore(t *testing.T) {
	m := score([]string{"x", "a", "y", "b"}, []string{"a", "b", "c"}, map[string]bool{"b": true, "c": true})
	if !near(m.Recall, 2.0/3) || !near(m.Precision, 0.5) || !near(m.RR, 0.5) {
		t.Fatalf("recall/precision/rr: %+v", m)
	}
	wantNDCG := (1/math.Log2(3) + 1/math.Log2(5)) / (1 + 1/math.Log2(3) + 1/math.Log2(4))
	if !near(m.NDCG, wantNDCG) {
		t.Fatalf("ndcg %g, want %g", m.NDCG, wantNDCG)
	}
	if m.AssocTotal != 2 || m.AssocFound != 1 {
		t.Fatalf("assoc: %+v", m)
	}
	if e := score(nil, []string{"a"}, nil); !e.Empty || e.Recall != 0 || e.Precision != 0 {
		t.Fatalf("empty: %+v", e)
	}
}

func TestAssociativeAndLatency(t *testing.T) {
	nt := map[string]map[string]bool{
		"db":   terms("SQLite storage engine"),
		"lock": terms("Badger directory locks"),
	}
	got := associative("which storage locking?", []string{"db", "lock"}, nt)
	if got["db"] || got["lock"] { // "storage" matches db; "locking"/"locks" stem to "lock"
		t.Fatalf("lexical overlap misclassified: %v", got)
	}
	if got := associative("kitchen", []string{"db"}, nt); !got["db"] {
		t.Fatal("no overlap must be associative")
	}
	l := latency([]time.Duration{1 * time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond, 100 * time.Millisecond})
	if l.P50 != 2 || l.P95 != 100 || l.Max != 100 || l.Count != 4 {
		t.Fatalf("latency: %+v", l)
	}
}

func TestGenerateIsDeterministicAndWellFormed(t *testing.T) {
	c := DefaultGenConfig()
	c.Nodes, c.Topics, c.Sessions, c.NoiseLinks = 200, 10, 5, 2
	a, err := Generate(c)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Generate(c)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same seed must give the same scenario")
	}
	if len(a.Nodes) != 200 || a.QueryCount() != 5*c.QueriesPerSession {
		t.Fatalf("sizes: %d nodes, %d queries", len(a.Nodes), a.QueryCount())
	}
	nodeTerms := map[string]map[string]bool{}
	for _, n := range a.Nodes {
		nodeTerms[n.Key] = terms(n.Title + " " + n.Content)
	}
	for _, s := range a.Sessions {
		for _, q := range s.Queries {
			if len(q.Relevant) < 2 {
				t.Fatalf("query without associative targets: %+v", q)
			}
			// The query's unique words identify exactly one relevant node.
			u := strings.Fields(q.Q)[:2]
			owners := 0
			for _, n := range a.Nodes {
				if strings.HasPrefix(n.Title, u[0]+" "+u[1]+" ") {
					owners++
				}
			}
			if owners != 1 {
				t.Fatalf("unique words %v owned by %d nodes", u, owners)
			}
		}
	}
	c.Seed = 2
	if d, _ := Generate(c); reflect.DeepEqual(a, d) {
		t.Fatal("different seeds must differ")
	}
}

// On a synthetic scenario with recurring groups, learning must improve
// associative recall over the no-learning baseline, and the explicit upper
// bound must beat everything.
func TestLearningImprovesAssociativeRecall(t *testing.T) {
	c := DefaultGenConfig()
	c.Nodes, c.Topics, c.Sessions, c.QueriesPerSession, c.TypedLinkRate = 120, 6, 8, 10, 0
	s, err := Generate(c)
	if err != nil {
		t.Fatal(err)
	}
	run := func(mode string) *Result {
		r, err := Run(s, Config{Name: mode, Learn: mode, Passes: 2}, t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	none, fb, ex := run(LearnNone), run(LearnFeedback), run(LearnExplicit)
	base := none.Passes[1].AssocRecall
	if base != 0 {
		t.Fatalf("without links or learning, associative targets are unreachable: %g", base)
	}
	if !(fb.Passes[1].AssocRecall > base) || !(ex.Passes[1].AssocRecall > fb.Passes[1].AssocRecall) {
		t.Fatalf("assoc recall: none %g, feedback %g, explicit %g", base, fb.Passes[1].AssocRecall, ex.Passes[1].AssocRecall)
	}
	if !(ex.Passes[1].AssocRecall > ex.Passes[0].AssocRecall) {
		t.Fatalf("explicit learning must improve over passes: %+v", ex.Passes)
	}
	if ex.Graph.Spurious != 0 || ex.Graph.Learned == 0 {
		t.Fatalf("explicit learning links only co-relevant nodes: %+v", ex.Graph)
	}
	if none.Graph.Learned != 0 || none.Perf.Recall.Count != 2*s.QueryCount() {
		t.Fatalf("baseline: %+v %+v", none.Graph, none.Perf.Recall)
	}

	rep := &Report{Scenario: s.Name, Nodes: len(s.Nodes), Sessions: len(s.Sessions), Queries: s.QueryCount(), K: 10,
		Results: []*Result{none, fb, ex}}
	var buf bytes.Buffer
	if err := rep.WriteText(&buf); err != nil || !strings.Contains(buf.String(), "assoc-recall") {
		t.Fatalf("report: %v\n%s", err, buf.String())
	}
}

func TestScenarioValidationAndRoundTrip(t *testing.T) {
	bad := &Scenario{Nodes: []ScenarioNode{{Key: "a", Type: "fact", Title: "A"}},
		Sessions: []Session{{Queries: []Query{{Q: "x", Relevant: []string{"zzz"}}}}}}
	if err := bad.Validate(); err == nil {
		t.Fatal("unknown relevant key accepted")
	}
	c := DefaultGenConfig()
	c.Nodes, c.Topics, c.Sessions = 20, 2, 1
	s, _ := Generate(c)
	path := filepath.Join(t.TempDir(), "s.json")
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil || !reflect.DeepEqual(s, back) {
		t.Fatalf("round trip: %v", err)
	}
}

// Noise metrics, the prompt-hook filter, mid-session adds with context links
// and scripted used reports (the session learning mode).
func TestNoiseStaleAndSessionMode(t *testing.T) {
	s := &Scenario{Name: "noise",
		Nodes: []ScenarioNode{
			{Key: "connect", Type: "procedure", Title: "Connect to the benchmark server", Content: "ssh to the server", Project: "solver"},
			{Key: "progress", Type: "procedure", Title: "Check benchmark progress on the server", Content: "tail the run log", Project: "solver"},
			{Key: "other", Type: "procedure", Title: "Run policy benchmarks on the server", Content: "screen job", Project: "policy"},
			{Key: "old", Type: "decision", Title: "Fix for the timeout regression: patch C", Importance: 0.9, Project: "solver"},
			{Key: "new", Type: "decision", Title: "Fix for the timeout regression: patch P", Importance: 0.9, Project: "solver"},
			{Key: "tip", Type: "fact", Title: "Read issues with gh api", Project: "solver"},
		},
		Sessions: []Session{{Project: "solver", Queries: []Query{
			{Q: "is the benchmark on the server still running?", Relevant: []string{"progress", "connect"}, Used: []string{"connect"}},
			{Q: "ok", Source: SourcePrompt},                                             // skipped by the prompt hook
			{Q: "<task-notification> server log tail done", Source: SourceNotification}, // unlabeled
			{Q: "fix for the timeout regression", Relevant: []string{"new"}, Stale: []string{"old"}, Add: []string{"tip"}},
		}}},
	}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	r, err := Run(s, Config{Name: "session", Learn: LearnSession, K: 6}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	q := r.Queries
	if !q[1].Skipped || len(q[1].Hits) != 0 || q[2].Skipped || !q[2].Metrics.Unlabeled {
		t.Fatalf("filter/unlabeled: %+v %+v", q[1], q[2])
	}
	if q[0].Metrics.CrossProject == 0 || q[0].Metrics.Irrelevant < q[0].Metrics.CrossProject {
		t.Fatalf("the policy server procedure should count as cross-project noise: %+v", q[0])
	}
	if q[3].Metrics.StaleLabels != 1 || q[3].Metrics.StaleHits != 1 {
		t.Fatalf("stale: %+v", q[3])
	}
	a := r.Passes[0]
	if a.Labeled != 2 || a.StaleQueries != 1 || a.StaleRate != 1 || !near(a.SkippedRate, 0.25) {
		t.Fatalf("aggregate: %+v", a)
	}
	// Unlabeled: the skipped "ok" (no hits) and the notification.
	if want := map[bool]float64{true: 0.5, false: 0}[len(q[2].Hits) > 0]; a.NoiseRate != want {
		t.Fatalf("noise rate %g, want %g", a.NoiseRate, want)
	}
	// "tip" was stored after the last query, linked to the used "connect".
	if r.Graph.Nodes != 6 || r.Graph.Learned == 0 || r.Graph.Spurious == 0 {
		t.Fatalf("mid-session add must create (spurious) context links: %+v", r.Graph)
	}
	bad := &Scenario{Nodes: s.Nodes, Sessions: []Session{{Queries: []Query{{Q: "x", Add: []string{"tip", "tip"}}}}}}
	if err := bad.Validate(); err == nil {
		t.Fatal("double add accepted")
	}
}
