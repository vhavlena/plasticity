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
	realRepo, _ := filepath.EvalSymlinks(repo)
	if got := Project(sub); got != realRepo {
		t.Fatalf("project of %s = %s, want %s", sub, got, realRepo)
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

// A git worktree belongs to the project of its main repository, also when
// reached through a symlink.
func TestWorktreeResolvesToMainRepository(t *testing.T) {
	root := t.TempDir()
	if r, err := filepath.EvalSymlinks(root); err == nil {
		root = r // macOS: /var -> /private/var
	}
	main := filepath.Join(root, "main")
	wtGit := filepath.Join(main, ".git", "worktrees", "feature")
	if err := os.MkdirAll(wtGit, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wtGit, "commondir"), []byte("../..\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wt := filepath.Join(root, "feature")
	if err := os.MkdirAll(filepath.Join(wt, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: "+wtGit+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Project(filepath.Join(wt, "src")); got != main {
		t.Fatalf("worktree project = %s, want %s", got, main)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(wt, link); err != nil {
		t.Fatal(err)
	}
	if got := Project(link); got != main {
		t.Fatalf("symlinked worktree project = %s, want %s", got, main)
	}
	// A .git file without commondir (a submodule) is its own project.
	sub := filepath.Join(root, "sub")
	os.MkdirAll(filepath.Join(root, "modgit"), 0o755)
	os.MkdirAll(sub, 0o755)
	os.WriteFile(filepath.Join(sub, ".git"), []byte("gitdir: ../modgit\n"), 0o644)
	if got := Project(sub); got != sub {
		t.Fatalf("submodule project = %s", got)
	}
}
