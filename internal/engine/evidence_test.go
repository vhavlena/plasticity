package engine

import (
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// chain builds sqlite —strong— badger —strong— hooks plus a tentative
// sqlite–hooks link that cannot carry activation, so a recall of "sqlite"
// reaches hooks along [sqlite, badger, hooks].
func chain(t *testing.T) *fixture {
	f := newFixture(t)
	for i := 0; i < 4; i++ {
		f.e.Reinforce([]string{f.ids["sqlite"], f.ids["badger"]})
		f.e.Reinforce([]string{f.ids["badger"], f.ids["hooks"]})
	}
	if err := f.e.AddLinks([]LinkSpec{{Src: f.ids["sqlite"], Dst: f.ids["hooks"], Kind: model.KindAssociative, Weight: 0.1}}); err != nil {
		t.Fatal(err)
	}
	return f
}

// Feedback weakens only the last hop of the unused neuron's activation path;
// ltd_all_links = 1 restores weakening every link to a used neuron.
func TestLTDWeakensLastHopOnly(t *testing.T) {
	for _, all := range []bool{false, true} {
		f := chain(t)
		if all {
			if err := f.e.SetParam("ltd_all_links", "1"); err != nil {
				t.Fatal(err)
			}
		}
		r := f.recall("sqlite WAL", "s")
		h, ok := hitIDs(r)[f.ids["hooks"]]
		if !ok || len(h.Path) != 3 || h.Path[1] != f.ids["badger"] {
			t.Fatalf("hooks should be reached via badger: %+v", r.Hits)
		}
		direct, lastHop := f.edge("sqlite", "hooks").Weight, f.weight("badger", "hooks")
		if _, err := f.e.Feedback("s", []string{f.ids["sqlite"], f.ids["badger"]}); err != nil {
			t.Fatal(err)
		}
		if !(f.weight("badger", "hooks") < lastHop) {
			t.Fatalf("all=%v: the last hop must be weakened", all)
		}
		if changed := f.edge("sqlite", "hooks").Weight != direct; changed != all {
			t.Fatalf("all=%v: the link that carried nothing changed: %v", all, changed)
		}
	}
}

// An unused seed was found on its own evidence: no link is weakened for it.
// A path without any used neuron weakens its last hop at the indirect rate.
func TestLTDSeedsAndIndirectPaths(t *testing.T) {
	f := chain(t)
	f.recall("sqlite badger", "s")
	before := f.weight("sqlite", "badger")
	if _, err := f.e.Feedback("s", []string{f.ids["sqlite"]}); err != nil {
		t.Fatal(err)
	}
	if w := f.weight("sqlite", "badger"); w != before {
		t.Fatalf("seed-only unused neuron weakened its link: %g -> %g", before, w)
	}

	// Full vs indirect rate on the same last hop (badger–hooks).
	drop := func(used string) float64 {
		g := chain(t)
		g.recall("sqlite WAL", "s")
		w := g.weight("badger", "hooks")
		if _, err := g.e.Feedback("s", []string{g.ids[used]}); err != nil {
			t.Fatal(err)
		}
		return w - g.weight("badger", "hooks")
	}
	full, indirect := drop("sqlite"), drop("pasta") // pasta: used, but not on the path
	if !(indirect > 0 && indirect < full) {
		t.Fatalf("indirect drop %g should be positive and below the full drop %g", indirect, full)
	}
}

// A recall below learn_min_confidence fires nothing and records no access;
// a confident one fires its hits and records light accesses.
func TestOnlyConfidentRecallsFireAndAccess(t *testing.T) {
	f := newFixture(t)
	state := func(name string) (int64, []model.Access) {
		t.Helper()
		var fires int64
		var acc []model.Access
		_ = f.e.st.Read(func(tx *store.Tx) error {
			n, err := tx.GetNode(f.ids[name])
			if err != nil {
				t.Fatal(err)
			}
			m, _ := tx.Accesses([]string{n.ID})
			fires, acc = n.Fires, m[n.ID]
			return nil
		})
		return fires, acc
	}
	if _, acc := state("hebb"); len(acc) != 1 || acc[0].Weight != 1 {
		t.Fatalf("creation should be one access of weight 1: %v", acc)
	}
	// 2 matched terms, 7 unknown: confidence ≈ 0.36 < learn_min_confidence.
	r := f.recall("kubernetes canary deploy rollout helm ingress terraform spreading hebbian", "s")
	if len(r.Hits) == 0 || r.Confidence >= f.e.P.LearnMinConfidence {
		t.Fatalf("expected a low-confidence recall with hits: %+v", r)
	}
	if fires, acc := state("hebb"); fires != 0 || len(acc) != 1 {
		t.Fatalf("low-confidence recall fired (%d) or accessed (%v)", fires, acc)
	}
	f.recall("hebbian learning", "s")
	fires, acc := state("hebb")
	if fires != 1 || len(acc) != 2 || acc[1].Weight != f.e.P.RecallAccessWeight {
		t.Fatalf("confident recall: fires %d, accesses %v", fires, acc)
	}
}
