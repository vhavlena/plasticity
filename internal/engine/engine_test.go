package engine

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
)

type fixture struct {
	t   *testing.T
	e   *Engine
	ids map[string]string // short name → node ID
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	e, err := Open(filepath.Join(t.TempDir(), "g.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	f := &fixture{t: t, e: e, ids: map[string]string{}}
	f.add("sqlite", model.TypeDecision, "Use SQLite WAL for storage", "multi-process readers, one writer", "hooks spawn processes")
	f.add("badger", model.TypeFact, "Badger holds an exclusive directory lock", "only one process may open the database", "")
	f.add("hooks", model.TypeFact, "Claude Code hooks run as separate processes", "each hook invocation is a new process", "")
	f.add("spread", model.TypeConcept, "Spreading activation", "activation flows along weighted edges", "")
	f.add("hebb", model.TypeConcept, "Hebbian learning", "neurons that fire together wire together", "")
	f.add("pasta", model.TypeProcedure, "Cooking pasta", "boil water, add salt", "")
	return f
}

func (f *fixture) add(name string, typ model.NodeType, title, content, rationale string) {
	f.t.Helper()
	n, err := f.e.AddNode(&model.Node{Type: typ, Title: title, Content: content, Rationale: rationale})
	if err != nil {
		f.t.Fatal(err)
	}
	f.ids[name] = n.ID
}

func (f *fixture) edge(a, b string) *model.Edge {
	f.t.Helper()
	var ed *model.Edge
	if err := f.e.st.Read(func(tx *store.Tx) error {
		var err error
		ed, err = tx.GetEdge(f.ids[a], f.ids[b], model.KindAssociative)
		return err
	}); err != nil {
		f.t.Fatal(err)
	}
	return ed
}

func (f *fixture) weight(a, b string) float64 {
	ed := f.edge(a, b)
	if ed == nil {
		return 0
	}
	return plasticity.EffectiveWeight(ed, f.e.P)
}

// fire makes each named neuron fire alone n times (as if recalled in n
// recalls without its partners), ageing its synapses by interference.
func (f *fixture) fire(n int, names ...string) {
	f.t.Helper()
	if err := f.e.st.Write(func(tx *store.Tx) error {
		for _, name := range names {
			for i := 0; i < n; i++ {
				if err := tx.Fire([]string{f.ids[name]}); err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		f.t.Fatal(err)
	}
}

// ticks advances the global recall clock by n without firing anything.
func (f *fixture) ticks(n int) {
	f.t.Helper()
	if err := f.e.st.Write(func(tx *store.Tx) error {
		for i := 0; i < n; i++ {
			if _, err := tx.AdvanceTick(); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fixture) recall(q, session string) *RecallResult {
	f.t.Helper()
	r, err := f.e.Recall(RecallRequest{Query: q, Session: session})
	if err != nil {
		f.t.Fatal(err)
	}
	return r
}

func hitIDs(r *RecallResult) map[string]RecallHit {
	m := map[string]RecallHit{}
	for _, h := range r.Hits {
		m[h.Node.ID] = h
	}
	return m
}

func TestCoRecallStrengthensAndSaturates(t *testing.T) {
	f := newFixture(t)
	prev := 0.0
	for i := 0; i < 8; i++ {
		r := f.recall("sqlite badger lock", "s1")
		if r.RecallID == "" || len(r.Hits) < 2 {
			t.Fatalf("recall %d: %+v", i, r)
		}
		rep, err := f.e.ReinforceSession("s1")
		if err != nil || rep.Traces != 1 || rep.Pairs == 0 {
			t.Fatalf("reinforce %d: %v %+v", i, err, rep)
		}
		w := f.weight("sqlite", "badger")
		if w <= prev || w >= 1 {
			t.Fatalf("step %d: weight %g did not grow below 1 (prev %g)", i, w, prev)
		}
		prev = w
	}
	if f.weight("sqlite", "pasta") != 0 {
		t.Fatal("unrelated nodes became associated")
	}
	// Learned-from traces are deleted: a second reinforce learns nothing.
	if rep, _ := f.e.ReinforceSession("s1"); rep.Traces != 0 || rep.Pairs != 0 {
		t.Fatalf("traces not deleted: %+v", rep)
	}
}

// After learning, recalling one concept brings up its associate even though
// the associate does not match the query lexically.
func TestAssociativeRecall(t *testing.T) {
	f := newFixture(t)
	before := hitIDs(f.recall("sqlite", ""))
	if _, ok := before[f.ids["badger"]]; ok {
		t.Fatal("badger should not be recalled before learning")
	}
	if _, err := f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if _, err := f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]}); err != nil {
			t.Fatal(err)
		}
	}
	after := hitIDs(f.recall("sqlite", ""))
	h, ok := after[f.ids["badger"]]
	if !ok {
		t.Fatalf("badger not recalled after learning (w=%g)", f.weight("sqlite", "badger"))
	}
	if h.IsSeed || h.Hops != 1 || len(h.Path) != 2 || h.Path[0] != f.ids["sqlite"] {
		t.Fatalf("unexpected path: %+v", h)
	}
}

func TestStrongerLinksReachDeeper(t *testing.T) {
	f := newFixture(t)
	// strong chain sqlite → hooks → spread, weak link sqlite → hebb
	for _, l := range [][2]string{{"sqlite", "hooks"}, {"hooks", "spread"}} {
		if _, err := f.e.Link(f.ids[l[0]], f.ids[l[1]], model.KindExplicit, 0.95); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.e.Link(f.ids["sqlite"], f.ids["hebb"], model.KindExplicit, 0.12); err != nil {
		t.Fatal(err)
	}
	got := hitIDs(f.recall("sqlite", ""))
	if h, ok := got[f.ids["spread"]]; !ok || h.Hops != 2 {
		t.Fatalf("2-hop node over strong links missing: %+v", h)
	}
	if _, ok := got[f.ids["hebb"]]; ok {
		t.Fatal("1-hop node over a weak link should be below threshold")
	}
}

func TestSpacedReinforcementSurvivesMassedIsForgotten(t *testing.T) {
	f := newFixture(t)
	spaced := []string{f.ids["sqlite"], f.ids["hooks"]}
	massed := []string{f.ids["spread"], f.ids["hebb"]}
	if _, err := f.e.Reinforce(massed); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if _, err := f.e.Reinforce(spaced); err != nil {
			t.Fatal(err)
		}
		f.fire(15, "sqlite") // used elsewhere between reinforcements
	}
	f.fire(60, "sqlite", "hooks", "spread", "hebb")
	rep, err := f.e.Consolidate(false)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Pruned != 1 || rep.EdgesScanned != 2 {
		t.Fatalf("consolidate: %+v", rep)
	}
	if f.edge("spread", "hebb") != nil {
		t.Fatal("unreinforced edge should be pruned")
	}
	if f.edge("sqlite", "hooks") == nil {
		t.Fatal("spaced-reinforced edge should survive")
	}
}

func TestFeedbackLTD(t *testing.T) {
	f := newFixture(t)
	// pre-existing associations of sqlite with badger and hooks
	for i := 0; i < 3; i++ {
		f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]})
		f.e.Reinforce([]string{f.ids["sqlite"], f.ids["hooks"]})
	}
	wBadger, wHooks := f.weight("sqlite", "badger"), f.weight("sqlite", "hooks")
	r := f.recall("sqlite", "s2")
	ids := hitIDs(r)
	if _, ok := ids[f.ids["badger"]]; !ok {
		t.Fatal("badger should be recalled")
	}
	rep, err := f.e.Feedback("s2", []string{f.ids["sqlite"], f.ids["hooks"]})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Depressed == 0 || rep.Traces != 1 {
		t.Fatalf("feedback: %+v", rep)
	}
	if !(f.weight("sqlite", "badger") < wBadger) {
		t.Fatal("recalled-but-unused association should be depressed")
	}
	if !(f.weight("sqlite", "hooks") > wHooks) {
		t.Fatal("used association should be potentiated")
	}
}

func TestHomeostasisBoundsHub(t *testing.T) {
	f := newFixture(t)
	hub := f.ids["hebb"]
	var others []string
	for i := 0; i < 15; i++ {
		f.add("x", model.TypeFact, "fact "+string(rune('a'+i)), "", "")
		others = append(others, f.ids["x"])
	}
	for round := 0; round < 10; round++ {
		for _, o := range others {
			if _, err := f.e.Reinforce([]string{hub, o}); err != nil {
				t.Fatal(err)
			}
		}
	}
	v, err := f.e.GetNode(hub)
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, ed := range v.Edges {
		total += ed.Weight
	}
	if len(v.Edges) != 15 || total > f.e.P.MaxOutWeight+1e-9 {
		t.Fatalf("hub total weight %g over %d edges exceeds %g", total, len(v.Edges), f.e.P.MaxOutWeight)
	}
}

func TestDormancyAndReactivation(t *testing.T) {
	f := newFixture(t)
	if rep, _ := f.e.Consolidate(false); rep.Dormant != 0 {
		t.Fatalf("fresh nodes must not be dormant: %+v", rep)
	}
	f.ticks(4000) // ≈ e^8 recalls elsewhere: B = −4
	rep, err := f.e.Consolidate(false)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Dormant != 6 {
		t.Fatalf("expected all isolated nodes dormant: %+v", rep)
	}
	r := f.recall("pasta", "")
	if len(r.Hits) != 1 || r.Hits[0].Node.Dormant {
		t.Fatalf("dormant node should be searchable and woken: %+v", r.Hits)
	}
	v, _ := f.e.GetNode(f.ids["pasta"])
	if v.Dormant {
		t.Fatal("recall should clear dormancy")
	}
}

func TestTypedLinksAndNodeView(t *testing.T) {
	f := newFixture(t)
	if _, err := f.e.Link(f.ids["badger"], f.ids["sqlite"], model.KindSupports, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := f.e.Link(f.ids["badger"], "nope", model.KindSupports, 0); err == nil {
		t.Fatal("linking unknown node must fail")
	}
	if _, err := f.e.Link(f.ids["badger"], f.ids["badger"], model.KindSupports, 0); err == nil {
		t.Fatal("self loop must fail")
	}
	v, err := f.e.GetNode(f.ids["sqlite"])
	if err != nil {
		t.Fatal(err)
	}
	if len(v.Edges) != 1 || v.Edges[0].Direction != "in" || v.Edges[0].Weight != f.e.P.DefaultTypedWeight ||
		v.Edges[0].OtherTitle != "Badger holds an exclusive directory lock" {
		t.Fatalf("node view: %+v", v.Edges)
	}
	// typed edges decay to the floor and are not pruned
	f.fire(5000, "sqlite")
	if rep, _ := f.e.Consolidate(false); rep.Pruned != 0 {
		t.Fatalf("typed edge pruned: %+v", rep)
	}
	v, _ = f.e.GetNode(f.ids["sqlite"])
	if w := v.Edges[0].Weight; w < f.e.P.TypedWeightFloor-1e-9 {
		t.Fatalf("typed weight %g below floor", w)
	}
}

func TestUpdateDeleteAndParams(t *testing.T) {
	f := newFixture(t)
	title := "Cooking spaghetti"
	if _, err := f.e.UpdateNode(f.ids["pasta"], NodePatch{Title: &title}); err != nil {
		t.Fatal(err)
	}
	if hits, _ := f.e.Search("spaghetti", 5); len(hits) != 1 {
		t.Fatalf("search after update: %v", hits)
	}
	bad := 2.0
	if _, err := f.e.UpdateNode(f.ids["pasta"], NodePatch{Importance: &bad}); err == nil {
		t.Fatal("invalid importance accepted")
	}
	if err := f.e.DeleteNode(f.ids["pasta"]); err != nil {
		t.Fatal(err)
	}
	if hits, _ := f.e.Search("spaghetti", 5); len(hits) != 0 {
		t.Fatal("deleted node still searchable")
	}
	if err := f.e.SetParam("hop_decay", "0.5"); err != nil || f.e.P.HopDecay != 0.5 {
		t.Fatalf("set param: %v", err)
	}
	if err := f.e.SetParam("hop_decay", "7"); err == nil {
		t.Fatal("out-of-range param accepted")
	}
	if err := f.e.SetParam("bogus", "1"); err == nil {
		t.Fatal("unknown param accepted")
	}
	if err := f.e.ResetParam("hop_decay"); err != nil || f.e.P.HopDecay != model.DefaultParams().HopDecay {
		t.Fatalf("reset param: %v", err)
	}
}

func TestMatchExprIsSafe(t *testing.T) {
	cases := map[string]string{
		`How does the SQLite lock work?`: `"sqlite" OR "lock" OR "work"`,
		`NEAR(a b) "x" OR * -y`:          `"near"`,
		`the a of`:                       ``,
		`Hebbian hebbian HEBBIAN`:        `"hebbian"`,
	}
	for in, want := range cases {
		if got := MatchExpr(in); got != want {
			t.Errorf("MatchExpr(%q) = %q, want %q", in, got, want)
		}
	}
	f := newFixture(t)
	for in := range cases {
		if _, err := f.e.Recall(RecallRequest{Query: in, NoRecord: true}); err != nil {
			t.Errorf("recall %q: %v", in, err)
		}
	}
}

func TestPPRRerankAndExport(t *testing.T) {
	f := newFixture(t)
	f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"], f.ids["hooks"]})
	r, err := f.e.Recall(RecallRequest{Query: "sqlite", Rerank: "ppr", NoRecord: true})
	if err != nil || len(r.Hits) != 3 {
		t.Fatalf("ppr recall: %v %+v", err, r)
	}
	var buf bytes.Buffer
	if err := f.e.Export(&buf, "json"); err != nil {
		t.Fatal(err)
	}
	var out struct {
		Nodes []model.Node      `json:"nodes"`
		Edges []json.RawMessage `json:"edges"`
	}
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil || len(out.Nodes) != 6 || len(out.Edges) != 3 {
		t.Fatalf("json export: %v nodes=%d edges=%d", err, len(out.Nodes), len(out.Edges))
	}
	buf.Reset()
	if err := f.e.Export(&buf, "dot"); err != nil || !strings.HasPrefix(buf.String(), "graph plasticity {") {
		t.Fatalf("dot export: %v", err)
	}
}

func TestQueryConfidenceCalibratesSeeds(t *testing.T) {
	f := newFixture(t)
	strong := f.recall("spreading activation", "")
	weak := f.recall("kubernetes deploy rollout spreading", "")
	if strong.Confidence != 1 || len(strong.Hits) == 0 {
		t.Fatalf("fully matched query: %+v", strong)
	}
	if !(weak.Confidence < strong.Confidence) || weak.Confidence >= f.e.P.LearnMinConfidence {
		t.Fatalf("partially matched query confidence %g", weak.Confidence)
	}
	if len(weak.Hits) != 1 || !weak.Hits[0].IsSeed || weak.Hits[0].Activation >= strong.Hits[0].Activation {
		t.Fatalf("weak match must be a weaker seed: %+v vs %+v", weak.Hits, strong.Hits)
	}
	unmatched := 0
	for _, ti := range weak.Terms {
		if ti.DocFreq == 0 && !ti.Matched {
			unmatched++
		}
	}
	if unmatched != 3 {
		t.Fatalf("term report: %+v", weak.Terms)
	}
}

func TestNoGoodMatchRecallsNothing(t *testing.T) {
	f := newFixture(t)
	r := f.recall("terraform helm chart ingress certificate rotation pasta", "s")
	if r.Seeds != 0 || len(r.Hits) != 0 || r.RecallID != "" {
		t.Fatalf("a single weak term among many unknown ones must not seed: %+v", r)
	}
	if r.Confidence <= 0 || r.Confidence >= f.e.P.MinSeedStrength {
		t.Fatalf("confidence %g", r.Confidence)
	}
}

func TestMultiTopicQueryKeepsAllSeeds(t *testing.T) {
	f := newFixture(t)
	r := f.recall("sqlite badger", "")
	ids := hitIDs(r)
	if r.Confidence != 1 || !ids[f.ids["sqlite"]].IsSeed || !ids[f.ids["badger"]].IsSeed {
		t.Fatalf("each topic's match should be a seed: %+v", r)
	}
}

func TestLowConfidenceRecallDoesNotLearn(t *testing.T) {
	f := newFixture(t)
	for i := 0; i < 3; i++ {
		r := f.recall("kubernetes canary deploy rollout spreading hebbian", "s")
		if r.RecallID != "" || r.TraceSkipped == "" || len(r.Hits) < 2 {
			t.Fatalf("low-confidence recall must not store a trace: %+v", r)
		}
	}
	rep, err := f.e.ReinforceSession("s")
	if err != nil || rep.Traces != 0 || f.weight("spread", "hebb") != 0 {
		t.Fatalf("learned from a low-confidence recall: %v %+v", err, rep)
	}
}

func TestWeaklyActivatedNodesStayOutOfTrace(t *testing.T) {
	f := newFixture(t)
	// sqlite → hebb carries activation 0.8·0.2 = 0.16: recalled (≥ θ) but
	// below learn_min_activation.
	if _, err := f.e.Link(f.ids["sqlite"], f.ids["hebb"], model.KindExplicit, 0.2); err != nil {
		t.Fatal(err)
	}
	r := f.recall("sqlite", "s")
	if _, ok := hitIDs(r)[f.ids["hebb"]]; !ok || r.RecallID == "" {
		t.Fatalf("hebb should be recalled and a trace stored: %+v", r)
	}
	rep, err := f.e.ReinforceSession("s")
	if err != nil || rep.Traces != 1 || rep.Pairs != 0 {
		t.Fatalf("weakly activated node entered learning: %v %+v", err, rep)
	}
}

// Without recalls nothing ages: consolidation is a no-op on memory.
func TestNoRecallsNoForgetting(t *testing.T) {
	f := newFixture(t)
	f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]})
	w := f.weight("sqlite", "badger")
	for i := 0; i < 3; i++ {
		if rep, _ := f.e.Consolidate(false); rep.Pruned != 0 || f.weight("sqlite", "badger") != w {
			t.Fatalf("memory changed without recalls: %+v w=%g", rep, f.weight("sqlite", "badger"))
		}
	}
}

// Pending traces that nobody learns from are dropped after TraceRetention
// recalls; learning deletes them immediately.
func TestTraceRetention(t *testing.T) {
	f := newFixture(t)
	if err := f.e.SetParam("trace_retention", "3"); err != nil {
		t.Fatal(err)
	}
	f.recall("sqlite badger", "old")
	f.recall("sqlite badger", "learned")
	if rep, _ := f.e.ReinforceSession("learned"); rep.Traces != 1 {
		t.Fatalf("reinforce: %+v", rep)
	}
	if st, _ := f.e.Stats(); st.PendingTraces != 1 {
		t.Fatalf("learned trace not deleted: %+v", st)
	}
	for i := 0; i < 3; i++ {
		f.recall("pasta", "")
	}
	rep, err := f.e.Consolidate(false)
	if err != nil || rep.TracesPurged != 1 {
		t.Fatalf("stale trace not purged: %v %+v", err, rep)
	}
	if r, _ := f.e.ReinforceSession("old"); r.Traces != 0 {
		t.Fatal("purged trace still learnable")
	}
}

// Recalling one endpoint without the other ages the synapse; recalling both
// together does not.
func TestSoloRecallAgesCoRecallRefreshes(t *testing.T) {
	f := newFixture(t)
	f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]})
	f.e.Reinforce([]string{f.ids["sqlite"], f.ids["pasta"]})
	w0 := f.weight("sqlite", "badger")
	for i := 0; i < 5; i++ {
		// both fire together; θ=0.3 keeps the weak sqlite–pasta link
		// (0.8·0.2 = 0.16) from pulling pasta into the result
		if _, err := f.e.Recall(RecallRequest{Query: "sqlite badger", Threshold: 0.3}); err != nil {
			t.Fatal(err)
		}
	}
	if ab := f.edge("sqlite", "badger"); ab.Age() != 0 || f.weight("sqlite", "badger") != w0 {
		t.Fatalf("co-recall aged the synapse: age=%d", ab.Age())
	}
	if ap := f.edge("sqlite", "pasta"); ap.Age() != 5 {
		t.Fatalf("sqlite–pasta should age by sqlite's 5 solo firings, age=%d", ap.Age())
	}
	// A high threshold keeps badger out of the result: sqlite fires alone.
	r, err := f.e.Recall(RecallRequest{Query: "sqlite", Threshold: 0.9})
	if err != nil || len(r.Hits) != 1 {
		t.Fatalf("solo recall: %v %+v", err, r)
	}
	if ab := f.edge("sqlite", "badger"); ab.Age() != 1 || !(f.weight("sqlite", "badger") < w0) {
		t.Fatalf("solo recall should age the synapse: age=%d", ab.Age())
	}
}

// A node pulled into a recall only through a link must not strengthen that
// link (or create a learned one next to it): that would be circular.
func TestReachedThroughLinkIsNotLearned(t *testing.T) {
	f := newFixture(t)
	if _, err := f.e.Link(f.ids["sqlite"], f.ids["hooks"], model.KindExplicit, 0.9); err != nil {
		t.Fatal(err)
	}
	r := f.recall("sqlite", "s")
	if h, ok := hitIDs(r)[f.ids["hooks"]]; !ok || h.IsSeed || r.RecallID == "" {
		t.Fatalf("hooks should be reached via the link and traced: %+v", r)
	}
	rep, err := f.e.ReinforceSession("s")
	if err != nil || rep.Pairs != 0 || rep.Dependent != 1 {
		t.Fatalf("dependent pair learned: %v %+v", err, rep)
	}
	if f.edge("sqlite", "hooks") != nil {
		t.Fatal("no associative synapse may be created from a dependent pair")
	}
}

// Same-seed descendants are explained by existing structure (no triangle
// closure); nodes from different seeds are independent evidence.
func TestOnlyDifferentSeedsAreLearned(t *testing.T) {
	f := newFixture(t)
	// badger → hooks and badger → spread: both descend from seed badger.
	for _, d := range []string{"hooks", "spread"} {
		if _, err := f.e.Link(f.ids["badger"], f.ids[d], model.KindExplicit, 0.9); err != nil {
			t.Fatal(err)
		}
	}
	r := f.recall("sqlite badger", "s")
	ids := hitIDs(r)
	if !ids[f.ids["sqlite"]].IsSeed || !ids[f.ids["badger"]].IsSeed || ids[f.ids["hooks"]].IsSeed || ids[f.ids["spread"]].IsSeed {
		t.Fatalf("unexpected seeds: %+v", r.Hits)
	}
	rep, err := f.e.ReinforceSession("s")
	if err != nil {
		t.Fatal(err)
	}
	// Independent: sqlite–badger, sqlite–hooks, sqlite–spread.
	// Dependent:   badger–hooks, badger–spread (on path), hooks–spread (same seed).
	if rep.Pairs != 3 || rep.Dependent != 3 {
		t.Fatalf("pairs/dependent: %+v", rep)
	}
	for _, p := range [][2]string{{"sqlite", "badger"}, {"sqlite", "hooks"}, {"sqlite", "spread"}} {
		if f.edge(p[0], p[1]) == nil {
			t.Errorf("independent pair %v not learned", p)
		}
	}
	for _, p := range [][2]string{{"badger", "hooks"}, {"badger", "spread"}, {"hooks", "spread"}} {
		if f.edge(p[0], p[1]) != nil {
			t.Errorf("dependent pair %v learned", p)
		}
	}
}

// A learned synapse cannot feed itself: once sqlite–badger exists, recalling
// only "sqlite" (badger reached through the synapse) does not strengthen it.
func TestLearnedLinkDoesNotSelfReinforce(t *testing.T) {
	f := newFixture(t)
	for i := 0; i < 3; i++ {
		f.recall("sqlite badger", "s")
		f.e.ReinforceSession("s")
	}
	w := f.weight("sqlite", "badger")
	for i := 0; i < 5; i++ {
		r := f.recall("sqlite", "s")
		if _, ok := hitIDs(r)[f.ids["badger"]]; !ok {
			t.Fatal("badger should be recalled via the learned synapse")
		}
		f.e.ReinforceSession("s")
	}
	if got := f.weight("sqlite", "badger"); got != w {
		t.Fatalf("synapse reinforced itself: %g → %g", w, got)
	}
}
