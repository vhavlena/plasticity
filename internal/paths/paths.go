// Package paths resolves where plasticity keeps its state: databases,
// downloaded models and the built binary all live under one home directory
// (PLASTICITY_HOME, default ~/.plasticity), independent of the Claude Code
// plugin directories, which change on every plugin update.
package paths

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
)

// Scopes of the memory.
const (
	ScopeGlobal  = "global"  // one graph shared by all projects (default)
	ScopeProject = "project" // one graph per git repository
)

// Home returns the state directory.
func Home() string {
	if h := os.Getenv("PLASTICITY_HOME"); h != "" {
		return h
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".plasticity")
	}
	return ".plasticity"
}

// ModelsDir is where embedding models are downloaded (shared by all graphs).
func ModelsDir() string { return filepath.Join(Home(), "models") }

// Scope returns the configured scope: PLASTICITY_SCOPE, else the plugin
// option CLAUDE_PLUGIN_OPTION_SCOPE, else global.
func Scope() string {
	for _, k := range []string{"PLASTICITY_SCOPE", "CLAUDE_PLUGIN_OPTION_SCOPE", "CLAUDE_PLUGIN_OPTION_scope"} {
		if v := strings.ToLower(strings.TrimSpace(os.Getenv(k))); v == ScopeGlobal || v == ScopeProject {
			return v
		}
	}
	return ScopeGlobal
}

// gitRoot returns the nearest ancestor of dir containing .git, else dir
// itself (absolute), without resolving symlinks or worktrees.
func gitRoot(dir string) string {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return dir
	}
	for d := abs; ; {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return d
		}
		parent := filepath.Dir(d)
		if parent == d {
			return abs
		}
		d = parent
	}
}

// Project returns the project directory containing dir: the nearest
// ancestor with a .git entry (or dir itself). Symlinks are resolved, and a
// git worktree (where .git is a file pointing into the main repository's
// .git/worktrees) resolves to its main repository, so all worktrees of a
// repository are one project.
func Project(dir string) string {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return dir
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	}
	for d := abs; ; {
		if fi, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			if !fi.IsDir() {
				if main := worktreeMain(d); main != "" {
					return main
				}
			}
			return d
		}
		parent := filepath.Dir(d)
		if parent == d {
			return abs
		}
		d = parent
	}
}

// worktreeMain returns the main repository of the worktree at dir, whose
// .git file reads "gitdir: <main>/.git/worktrees/<name>"; "" if it is not
// such a file. The worktree's gitdir holds "commondir", the path (relative
// to it) of the main repository's .git.
func worktreeMain(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, ".git"))
	if err != nil {
		return ""
	}
	gitdir, ok := strings.CutPrefix(strings.TrimSpace(string(b)), "gitdir:")
	if !ok {
		return ""
	}
	gitdir = strings.TrimSpace(gitdir)
	if !filepath.IsAbs(gitdir) {
		gitdir = filepath.Join(dir, gitdir)
	}
	c, err := os.ReadFile(filepath.Join(gitdir, "commondir"))
	if err != nil {
		return "" // a submodule: its own project
	}
	common := strings.TrimSpace(string(c))
	if !filepath.IsAbs(common) {
		common = filepath.Join(gitdir, common)
	}
	common = filepath.Clean(common)
	if real, err := filepath.EvalSymlinks(common); err == nil {
		common = real
	}
	if filepath.Base(common) != ".git" {
		return "" // a bare repository: keep the worktree
	}
	return filepath.Dir(common)
}

// DB returns the database path: PLASTICITY_DB if set, else by scope — the
// global graph, or the graph of the project containing dir.
func DB(dir string) string {
	if p := os.Getenv("PLASTICITY_DB"); p != "" {
		return p
	}
	if Scope() == ScopeProject {
		// The database path keeps the plain .git-ancestor resolution (no
		// symlink or worktree resolution, unlike Project), so existing
		// per-project databases stay where they are.
		h := sha256.Sum256([]byte(gitRoot(dir)))
		return filepath.Join(Home(), "projects", hex.EncodeToString(h[:6]), "graph.db")
	}
	return filepath.Join(Home(), "graph.db")
}
