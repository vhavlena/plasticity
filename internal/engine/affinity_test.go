package engine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/paths"
	"github.com/vhavlena/plasticity/internal/store"
)

func TestForeignByProjects(t *testing.T) {
	for _, c := range []struct {
		name string
		ps   []store.NodeProject
		want bool
	}{
		{"legacy (no records)", nil, false},
		{"own: created here", []store.NodeProject{{Project: "/me", Created: true}}, false},
		{"own: used here", []store.NodeProject{{Project: "/other", Created: true}, {Project: "/me", Uses: 1}}, false},
		{"shared: used in 2 projects", []store.NodeProject{{Project: "/a", Created: true, Uses: 1}, {Project: "/b", Uses: 3}}, false},
		{"foreign: created elsewhere", []store.NodeProject{{Project: "/a", Created: true}}, true},
		{"foreign: used in 1 other project", []store.NodeProject{{Project: "/a", Created: true}, {Project: "/b", Uses: 1}}, true},
	} {
		if got := foreignByProjects(c.ps, "/me", 2); got != c.want {
			t.Errorf("%s: foreign = %v, want %v", c.name, got, c.want)
		}
	}
}

// A foreign neuron's seed strength is scaled by foreign_project_factor for
// hook recalls and by the explicit factor for Claude's; global-tagged and
// own neurons are not.
func TestProjectAffinityScalesForeignSeeds(t *testing.T) {
	f := newFixture(t)
	f.add("glob", model.TypeFact, "Badger tip for every repository", "badger lock advice", "")
	for name, project := range map[string]string{"sqlite": "/me", "badger": "/other", "glob": "/other"} {
		if err := f.e.NoteNodeProject(f.ids[name], project, true, 0); err != nil {
			t.Fatal(err)
		}
	}
	tags := []string{GlobalTag}
	if _, err := f.e.UpdateNode(f.ids["glob"], NodePatch{Tags: &tags}); err != nil {
		t.Fatal(err)
	}
	seeds := func(req RecallRequest) map[string]float64 {
		t.Helper()
		var sd *seeding
		if err := f.e.st.Read(func(tx *store.Tx) error {
			terms := queryTerms(req.Query)
			hits, err := tx.Search(MatchExpr(req.Query), f.e.bm25Weights(), 10)
			if err != nil {
				return err
			}
			if sd, err = f.e.computeSeeds(tx, terms, hits, nil, nil); err != nil {
				return err
			}
			return f.e.applyProjectAffinity(tx, &req, sd.seeds)
		}); err != nil {
			t.Fatal(err)
		}
		return sd.seeds
	}
	base := seeds(RecallRequest{Query: "sqlite badger lock"})
	auto := seeds(RecallRequest{Query: "sqlite badger lock", Project: "/me", Auto: true})
	explicit := seeds(RecallRequest{Query: "sqlite badger lock", Project: "/me"})
	approx := func(a, b float64) bool { return a-b < 1e-9 && b-a < 1e-9 }
	if !approx(auto[f.ids["badger"]], base[f.ids["badger"]]*f.e.P.ForeignProjectFactor) ||
		!approx(explicit[f.ids["badger"]], base[f.ids["badger"]]*f.e.P.ForeignProjectFactorExplicit) {
		t.Fatalf("foreign seed: base %g auto %g explicit %g", base[f.ids["badger"]], auto[f.ids["badger"]], explicit[f.ids["badger"]])
	}
	for _, k := range []string{"sqlite", "glob", "hooks"} { // own, global, no records
		if auto[f.ids[k]] != base[f.ids[k]] {
			t.Errorf("%s scaled: %g -> %g", k, base[f.ids[k]], auto[f.ids[k]])
		}
	}
	// The project defaults to the session's.
	f.e.StartSession("S", "/me", "startup")
	if s := seeds(RecallRequest{Query: "sqlite badger lock", Session: "S", Auto: true}); !approx(s[f.ids["badger"]], auto[f.ids["badger"]]) {
		t.Fatalf("session project not used: %g", s[f.ids["badger"]])
	}
}

// Project records written under a path that now resolves differently (a
// symlink to the repository) still count as the recall's project.
func TestStoredProjectPathsAreNormalized(t *testing.T) {
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(repo, link); err != nil {
		t.Fatal(err)
	}
	f := newFixture(t)
	if err := f.e.NoteNodeProject(f.ids["badger"], link, true, 0); err != nil { // recorded before resolution
		t.Fatal(err)
	}
	var foreign map[string]bool
	_ = f.e.st.Read(func(tx *store.Tx) error {
		var err error
		foreign, err = f.e.foreignSet(tx, paths.Project(repo), []string{f.ids["badger"]})
		return err
	})
	if foreign[f.ids["badger"]] {
		t.Fatal("a neuron recorded under a symlinked path is foreign to its own repository")
	}
}
