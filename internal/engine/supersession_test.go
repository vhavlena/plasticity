package engine

import (
	"slices"
	"strings"
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// supersede marks fixture neuron old as replaced by successor.
func (f *fixture) supersede(old, successor string) {
	f.t.Helper()
	succ := f.ids[successor]
	if _, err := f.e.UpdateNode(f.ids[old], NodePatch{SupersededBy: &succ}); err != nil {
		f.t.Fatal(err)
	}
}

// With ExcludeSuperseded, a query worded like the superseded neuron finds
// its successor in its place; explicit recall keeps it, scaled down.
func TestSupersededRedirectsToSuccessor(t *testing.T) {
	f := newFixture(t)
	f.add("pg", model.TypeDecision, "Use Postgres for the graph", "server database instead of a file", "")
	f.supersede("sqlite", "pg")
	q := RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true}
	r, err := f.e.Recall(q)
	if err != nil {
		t.Fatal(err)
	}
	ids := hitIDs(r)
	if _, ok := ids[f.ids["sqlite"]]; ok {
		t.Fatalf("superseded neuron injected: %+v", r.Hits)
	}
	h, ok := ids[f.ids["pg"]]
	if !ok || !slices.Equal(h.Supersedes, []string{"Use SQLite WAL for storage"}) || h.Path[len(h.Path)-1] != f.ids["pg"] || h.Path[0] != f.ids["sqlite"] {
		t.Fatalf("successor not redirected to: %+v", h)
	}
	q.ExcludeSuperseded = false
	r, _ = f.e.Recall(q)
	old, ok := hitIDs(r)[f.ids["sqlite"]]
	if !ok || old.Node.SuccessorTitle != "Use Postgres for the graph" {
		t.Fatalf("explicit recall should show the superseded neuron with its successor: %+v", r.Hits)
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SuccessorTitle != "Use Postgres for the graph" {
		t.Fatalf("get: %+v", v.Node)
	}
}

// Redirects follow chains to the current neuron; supersede refuses cycles,
// and a cycle in stored data neither hangs recall nor hides its neurons.
func TestSupersessionChainsAndCycles(t *testing.T) {
	f := newFixture(t)
	f.add("v2", model.TypeDecision, "Storage v2", "", "")
	f.add("v3", model.TypeDecision, "Storage v3", "", "")
	f.supersede("sqlite", "v2")
	f.supersede("v2", "v3")
	r, _ := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true})
	ids := hitIDs(r)
	if _, ok := ids[f.ids["v3"]]; !ok {
		t.Fatalf("chain end not reached: %+v", r.Hits)
	}
	if _, ok := ids[f.ids["v2"]]; ok {
		t.Fatal("intermediate superseded neuron injected")
	}
	succ := f.ids["sqlite"]
	if _, err := f.e.UpdateNode(f.ids["v3"], NodePatch{SupersededBy: &succ}); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("cycle accepted: %v", err)
	}
	if err := f.e.st.Write(func(tx *store.Tx) error { return tx.SetSupersededBy(f.ids["v3"], f.ids["sqlite"]) }); err != nil {
		t.Fatal(err)
	}
	// No neuron of a cyclic chain has a current successor, so none is
	// hidden (or redirected), and recall terminates.
	r, err := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := hitIDs(r)[f.ids["sqlite"]]; !ok {
		t.Fatalf("neuron of a cyclic chain hidden: %+v", r.Hits)
	}
	// Clearing makes the neuron current again.
	none := ""
	if _, err := f.e.UpdateNode(f.ids["v3"], NodePatch{SupersededBy: &none}); err != nil {
		t.Fatal(err)
	}
	if r, _ := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true}); hitIDs(r)[f.ids["v3"]].Node == nil {
		t.Fatalf("cleared neuron not current: %+v", r.Hits)
	}
}

// add --supersedes marks the old neurons in the insert's transaction: an
// unknown ID aborts the whole add.
func TestAddSupersedesIsAtomic(t *testing.T) {
	f := newFixture(t)
	before, _ := f.e.Stats()
	_, err := f.e.AddNodeChecked(&model.Node{Type: model.TypeDecision, Title: "Use Postgres"},
		AddOptions{Supersedes: []string{f.ids["sqlite"], "01NOPE"}})
	if err == nil {
		t.Fatal("unknown superseded ID accepted")
	}
	if after, _ := f.e.Stats(); after.Nodes != before.Nodes {
		t.Fatal("node inserted although the add failed")
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SupersededBy != "" {
		t.Fatal("partial supersession left behind")
	}
	res, err := f.e.AddNodeChecked(&model.Node{Type: model.TypeDecision, Title: "Use Postgres"},
		AddOptions{Supersedes: []string{f.ids["sqlite"]}})
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SupersededBy != res.Node.ID {
		t.Fatalf("not superseded: %+v", v.Node)
	}
}

// A corrected neuron usually looks like a duplicate of the one it replaces;
// --supersedes must still insert it and mark the old one.
func TestSupersedesBypassesDuplicateOfReplaced(t *testing.T) {
	f := newFixture(t, WithEmbedder(conceptEmbedder{}))
	// Same title as the sqlite neuron.
	res, err := f.e.AddNodeChecked(&model.Node{Type: model.TypeDecision, Title: "Use SQLite WAL for storage", Content: "WAL checkpoints tuned"},
		AddOptions{Supersedes: []string{f.ids["sqlite"]}})
	if err != nil || res.Node == nil || res.Duplicate != nil {
		t.Fatalf("same-title correction refused: %v %+v", err, res)
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SupersededBy != res.Node.ID {
		t.Fatalf("not superseded: %+v", v.Node)
	}
	// Semantic near-duplicate of the pasta neuron (same concept vector).
	res, err = f.e.AddNodeChecked(&model.Node{Type: model.TypeProcedure, Title: "Kitchen meal", Content: "food"},
		AddOptions{Supersedes: []string{f.ids["pasta"]}})
	if err != nil || res.Node == nil {
		t.Fatalf("semantic correction refused: %v %+v", err, res)
	}
	// Without --supersedes the duplicate check still applies.
	if res, _ := f.e.AddNodeChecked(&model.Node{Type: model.TypeFact, Title: "Hebbian learning"}, AddOptions{}); res.Duplicate == nil {
		t.Fatal("duplicate check bypassed")
	}
}

// Deleting a successor re-points its predecessors to its own successor, or
// makes them current: they never stay hidden behind a dangling mark.
func TestDeletingSuccessorRepointsPredecessors(t *testing.T) {
	f := newFixture(t)
	f.add("v2", model.TypeDecision, "Storage v2", "", "")
	f.add("v3", model.TypeDecision, "Storage v3", "", "")
	f.supersede("sqlite", "v2")
	f.supersede("v2", "v3")
	if err := f.e.DeleteNode(f.ids["v2"]); err != nil {
		t.Fatal(err)
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SupersededBy != f.ids["v3"] {
		t.Fatalf("predecessor should now be superseded by v3: %+v", v.Node)
	}
	if err := f.e.DeleteNode(f.ids["v3"]); err != nil {
		t.Fatal(err)
	}
	if v, _ := f.e.GetNode(f.ids["sqlite"]); v.Node.SupersededBy != "" {
		t.Fatalf("predecessor should be current again: %+v", v.Node)
	}
	r, _ := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true})
	if _, ok := hitIDs(r)[f.ids["sqlite"]]; !ok {
		t.Fatalf("predecessor hidden after its successors were deleted: %+v", r.Hits)
	}
}

// A successor that is a seed is its own source: feedback weakens no link
// for it even when its strongest path ran through the old neuron. A
// successor reached only through the redirect never has the redirect hop
// old→successor weakened either.
func TestRedirectKeepsOwnPathAndSkipsLTD(t *testing.T) {
	f := newFixture(t)
	f.add("pg", model.TypeDecision, "Postgres storage", "server database", "") // a seed itself ("storage")
	for i := 0; i < 4; i++ {
		f.e.Reinforce([]string{f.ids["sqlite"], f.ids["pg"]})
	}
	f.supersede("sqlite", "pg")
	r, _ := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", Session: "s", ExcludeSuperseded: true})
	if h := hitIDs(r)[f.ids["pg"]]; !h.IsSeed {
		t.Fatalf("successor should be a seed: %+v", h)
	}
	// (The link still ages: the successor fired without the excluded old
	// neuron. Only LTD is checked here.)
	w := f.weight("sqlite", "pg")
	if rep, err := f.e.Feedback("s", []string{f.ids["hooks"]}); err != nil || f.weight("sqlite", "pg") < w {
		t.Fatalf("link weakened for an unused seed: %v %+v", err, rep)
	}

	g := newFixture(t)
	g.add("pg", model.TypeDecision, "Relational server", "networked", "") // reached only via the redirect
	for i := 0; i < 4; i++ {
		g.e.Reinforce([]string{g.ids["sqlite"], g.ids["pg"]})
	}
	g.supersede("sqlite", "pg")
	r, _ = g.e.Recall(RecallRequest{Query: "sqlite WAL", Session: "s", ExcludeSuperseded: true})
	h := hitIDs(r)[g.ids["pg"]]
	if len(h.Path) < 2 || h.Path[len(h.Path)-2] != g.ids["sqlite"] {
		t.Fatalf("redirected successor path: %+v", h)
	}
	w = g.weight("sqlite", "pg")
	if rep, err := g.e.Feedback("s", []string{g.ids["hooks"]}); err != nil || g.weight("sqlite", "pg") < w {
		t.Fatalf("redirect hop weakened: %v %+v", err, rep)
	}
}
