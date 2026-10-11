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

// Project returns the project a directory belongs to: the nearest ancestor
// containing .git, else the directory itself (absolute).
func Project(dir string) string {
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

// DB returns the database path: PLASTICITY_DB if set, else by scope — the
// global graph, or the graph of the project containing dir.
func DB(dir string) string {
	if p := os.Getenv("PLASTICITY_DB"); p != "" {
		return p
	}
	if Scope() == ScopeProject {
		h := sha256.Sum256([]byte(Project(dir)))
		return filepath.Join(Home(), "projects", hex.EncodeToString(h[:6]), "graph.db")
	}
	return filepath.Join(Home(), "graph.db")
}
