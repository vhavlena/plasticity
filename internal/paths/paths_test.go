package paths

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolution(t *testing.T) {
	home := t.TempDir()
	t.Setenv("PLASTICITY_HOME", home)
	t.Setenv("PLASTICITY_DB", "")
	t.Setenv("PLASTICITY_SCOPE", "")
	t.Setenv("CLAUDE_PLUGIN_OPTION_SCOPE", "")

	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(repo, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := Project(sub); got != repo {
		t.Fatalf("project of %s = %s, want %s", sub, got, repo)
	}
	if got := DB(sub); got != filepath.Join(home, "graph.db") {
		t.Fatalf("global db: %s", got)
	}
	t.Setenv("CLAUDE_PLUGIN_OPTION_SCOPE", "project")
	p1, p2 := DB(sub), DB(repo)
	if p1 != p2 || filepath.Dir(filepath.Dir(p1)) != filepath.Join(home, "projects") {
		t.Fatalf("project db: %s vs %s", p1, p2)
	}
	if DB(t.TempDir()) == p1 {
		t.Fatal("different projects must get different databases")
	}
	t.Setenv("PLASTICITY_DB", "/x/y.db")
	if DB(sub) != "/x/y.db" {
		t.Fatal("PLASTICITY_DB must win")
	}
	if ModelsDir() != filepath.Join(home, "models") {
		t.Fatal("models dir")
	}
}
