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
// and a cycle in stored data does not hang recall.
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
	r, err := f.e.Recall(RecallRequest{Query: "sqlite WAL storage", NoRecord: true, ExcludeSuperseded: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"sqlite", "v2", "v3"} {
		if _, ok := hitIDs(r)[f.ids[k]]; ok {
			t.Fatalf("neuron %s of a cyclic chain injected", k)
		}
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
