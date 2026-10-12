package hooks

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"

	_ "modernc.org/sqlite"
)

func newEngine(t *testing.T) *engine.Engine {
	t.Helper()
	e, err := engine.Open(filepath.Join(t.TempDir(), "g.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

func add(t *testing.T, e *engine.Engine, title, content string) string {
	t.Helper()
	n, err := e.AddNode(&model.Node{Type: model.TypeDecision, Title: title, Content: content, Rationale: "because"})
	if err != nil {
		t.Fatal(err)
	}
	return n.ID
}

// transcript writes a JSONL transcript with the given tool calls.
func transcript(t *testing.T, tools ...[2]string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "t.jsonl")
	var lines []string
	lines = append(lines, `{"type":"user","message":{"content":"hello"}}`, `not json`)
	for _, tl := range tools {
		blk := map[string]any{"type": "tool_use", "name": tl[0], "input": map[string]string{"file_path": tl[1]}}
		raw, _ := json.Marshal(map[string]any{"type": "assistant", "message": map[string]any{"content": []any{map[string]string{"type": "text", "text": "x"}, blk}}})
		lines = append(lines, string(raw))
	}
	lines = append(lines, `{"type":"assistant","message":{"content":"plain text reply"}}`)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestConfigFromEnv(t *testing.T) {
	for _, k := range []string{"PLASTICITY_RECALL_LIMIT", "PLASTICITY_REFLECT", "PLASTICITY_EMBEDDINGS", "PLASTICITY_SCOPE", "PLASTICITY_LOG", "PLASTICITY_EPISODES"} {
		t.Setenv(k, "")
	}
	t.Setenv("CLAUDE_PLUGIN_OPTION_RECALL_LIMIT", "4")
	t.Setenv("CLAUDE_PLUGIN_OPTION_reflect", "false")
	t.Setenv("PLASTICITY_EMBEDDINGS", "off")
	c := ConfigFromEnv()
	if c.RecallLimit != 4 || c.Reflect || c.Embeddings != "off" || !c.Log || c.Scope != "global" || c.Episodes {
		t.Fatalf("config: %+v", c)
	}
	t.Setenv("CLAUDE_PLUGIN_OPTION_EPISODES", "true")
	if !ConfigFromEnv().Episodes {
		t.Fatal("episodes option ignored")
	}
}

func TestSessionStartAndPrompt(t *testing.T) {
	e := newEngine(t)
	cfg := Config{Scope: "global", RecallLimit: 5, Reflect: true, Log: true, Embeddings: "off", ContextBudget: 3000, DB: "/h/it's/graph.db"}
	cli := `plasticity --db '/h/it'\''s/graph.db'`
	sqlite := add(t, e, "Use SQLite WAL for storage", "many processes read while one writes")
	add(t, e, "Cooking pasta", "boil water")

	out, err := SessionStart(e, Input{SessionID: "S1", Source: "startup"}, "/repo", cfg)
	if err != nil || out.Specific.HookEventName != "SessionStart" {
		t.Fatalf("session start: %v %+v", err, out)
	}
	ctx := out.Specific.AdditionalContext
	for _, want := range []string{"2 neurons", "S1", cli + " recall", cli + " add", cli + " used --session S1"} {
		if !strings.Contains(ctx, want) {
			t.Fatalf("session-start context lacks %q:\n%s", want, ctx)
		}
	}

	out, err = Prompt(e, Input{SessionID: "S1", Prompt: "why did we pick sqlite storage?"}, "/repo", cfg)
	if err != nil || out == nil || out.Specific.HookEventName != "UserPromptSubmit" {
		t.Fatalf("prompt: %v %+v", err, out)
	}
	ctx = out.Specific.AdditionalContext
	if !strings.Contains(ctx, sqlite) || !strings.Contains(ctx, "Why: because") || strings.Contains(ctx, "pasta") || !strings.Contains(ctx, cli+" used --session S1") {
		t.Fatalf("injected context:\n%s", ctx)
	}
	if out, _ := Prompt(e, Input{SessionID: "S1", Prompt: "/compact"}, "/repo", cfg); out != nil {
		t.Fatal("slash commands must not trigger recall")
	}
	if out, _ := Prompt(e, Input{SessionID: "S1", Prompt: "kubernetes helm charts deployment"}, "/repo", cfg); out != nil {
		t.Fatalf("unmatched prompt must inject nothing: %+v", out)
	}
	s, _ := e.Session("S1")
	if s.Prompts != 3 || s.FirstPrompt != "why did we pick sqlite storage?" || s.Project != "/repo" {
		t.Fatalf("session: %+v", s)
	}
	recalls, _ := e.Events(store.EventFilter{Session: "S1", Kind: engine.EventRecall})
	if len(recalls) != 2 || !strings.Contains(string(recalls[0].Payload), sqlite) {
		t.Fatalf("recall events: %+v", recalls)
	}
}

func TestFormatMemoriesBudget(t *testing.T) {
	r := &engine.RecallResult{}
	for i := 0; i < 50; i++ {
		r.Hits = append(r.Hits, engine.RecallHit{IsSeed: true, Node: &model.Node{ID: strings.Repeat("X", 26), Type: model.TypeFact,
			Title: "a fairly long title for budget testing", Content: strings.Repeat("content ", 40)}})
	}
	text := FormatMemories(r, "S", "plasticity", 1500)
	if len(text) > 1500 || !strings.HasSuffix(text, "</plasticity-memories>") || strings.Count(text, "- [") < 2 {
		t.Fatalf("budget not respected (%d chars):\n%s", len(text), text)
	}
	if FormatMemories(&engine.RecallResult{}, "S", "plasticity", 1500) != "" {
		t.Fatal("no hits must give no context")
	}
}

func TestStopNudge(t *testing.T) {
	e := newEngine(t)
	cfg := Config{Reflect: true, Log: true, DB: "/h/graph.db"}
	e.StartSession("S", "", "startup")
	few := transcript(t, [2]string{"Read", "a.go"})
	if out, _ := Stop(e, Input{SessionID: "S", TranscriptPath: few}, cfg); out != nil {
		t.Fatal("nudged after one tool call")
	}
	var tools [][2]string
	for i := 0; i < 6; i++ {
		tools = append(tools, [2]string{"Bash", ""})
	}
	many := transcript(t, tools...)
	if out, _ := Stop(e, Input{SessionID: "S", TranscriptPath: many, StopHookActive: true}, cfg); out != nil {
		t.Fatal("must not nudge while already continuing from a stop hook")
	}
	out, err := Stop(e, Input{SessionID: "S", TranscriptPath: many}, cfg)
	if err != nil || out == nil || out.Decision != "block" || !strings.Contains(out.Reason, "plasticity --db '/h/graph.db' add") || !strings.Contains(out.Reason, "--session S") {
		t.Fatalf("expected nudge: %v %+v", err, out)
	}
	if !strings.Contains(nudgeReason("S", Config{DB: "/h/graph.db"}.CLI()), "plasticity --db '/h/graph.db' --no-log add") {
		t.Fatal("log=false must pass --no-log to the suggested commands")
	}
	if out, _ := Stop(e, Input{SessionID: "S", TranscriptPath: many}, cfg); out != nil {
		t.Fatal("nudged twice")
	}
	e.StartSession("T", "", "startup")
	if out, _ := Stop(e, Input{SessionID: "T", TranscriptPath: many}, Config{Reflect: false}); out != nil {
		t.Fatal("reflect=false must disable the nudge")
	}
}

func TestTranscriptParsingAndSessionEnd(t *testing.T) {
	path := transcript(t, [2]string{"Edit", "/repo/a.go"}, [2]string{"Write", "/repo/sub/b.go"}, [2]string{"Read", "/repo/c.go"},
		[2]string{"Edit", "/repo/a.go"}, [2]string{"Write", "/elsewhere/x.txt"})
	if n, err := CountToolCalls(path); err != nil || n != 5 {
		t.Fatalf("tool calls: %d %v", n, err)
	}
	files, _ := EditedFiles(path, "/repo")
	if strings.Join(files, ",") != "/elsewhere/x.txt,a.go,sub/b.go" {
		t.Fatalf("edited files: %v", files)
	}
	if _, err := CountToolCalls(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing transcript must error")
	}

	e := newEngine(t)
	id := add(t, e, "Use SQLite WAL for storage", "x")
	e.StartSession("S", "/repo", "startup")
	e.NotePrompt("S", "/repo", "refactor the storage layer please")
	e.Used("S", []string{id})
	rep, err := SessionEnd(e, Input{SessionID: "S", TranscriptPath: path}, "/repo", episodes)
	if err != nil || rep.Episode == nil || !strings.Contains(rep.Episode.Content, "sub/b.go") || rep.Links != 1 {
		t.Fatalf("session end: %v %+v", err, rep)
	}
	rep, err = SessionEnd(e, Input{SessionID: "unknown", TranscriptPath: "/nope"}, "/repo", episodes)
	if err != nil || rep != nil {
		t.Fatalf("unknown session: %v %+v", err, rep)
	}
}

var episodes = Config{Episodes: true}

func TestEpisodesOff(t *testing.T) {
	e := newEngine(t)
	id := add(t, e, "Use SQLite WAL for storage", "x")
	e.StartSession("S", "/repo", "startup")
	e.NotePrompt("S", "/repo", "refactor the sqlite storage layer please")
	e.Used("S", []string{id})
	rep, err := SessionEnd(e, Input{SessionID: "S"}, "/repo", Config{})
	if err != nil || rep.Episode != nil {
		t.Fatalf("episode created although disabled: %v %+v", err, rep)
	}
	if s, _ := e.Session("S"); s.EndedAt.IsZero() || s.EpisodeID != "" {
		t.Fatalf("session not closed: %+v", s)
	}
	ss, err := e.Sessions("", 10)
	if err != nil || len(ss) != 1 || len(ss[0].Used) != 1 || ss[0].Used[0].ID != id || ss[0].FirstPrompt == "" {
		t.Fatalf("session history: %v %+v", err, ss)
	}

	// Episodes that exist (created while enabled) are never injected.
	e.StartSession("T", "/repo", "startup")
	e.NotePrompt("T", "/repo", "sqlite storage layer refactoring")
	if rep, err := SessionEnd(e, Input{SessionID: "T"}, "/repo", episodes); err != nil || rep.Episode == nil {
		t.Fatalf("episode: %v %+v", err, rep)
	}
	out, err := Prompt(e, Input{SessionID: "U", Prompt: "sqlite storage layer refactoring"}, "/repo", Config{RecallLimit: 5, ContextBudget: 3000})
	if err != nil || out == nil {
		t.Fatalf("prompt: %v", err)
	}
	if ctx := out.Specific.AdditionalContext; strings.Contains(ctx, "episode:") || !strings.Contains(ctx, id) {
		t.Fatalf("episode injected:\n%s", ctx)
	}
}

type fakeEmbedder struct{}

func (fakeEmbedder) ID() string   { return "fake:setup" }
func (fakeEmbedder) Close() error { return nil }
func (fakeEmbedder) Embed(context.Context, string, embed.Kind) ([]float32, error) {
	return []float32{1, 0}, nil
}

func init() {
	embed.Register("fakesetup", embed.Provider{Factory: func(context.Context, embed.Config) (embed.Embedder, error) {
		return fakeEmbedder{}, nil
	}})
	embed.Register("broken", embed.Provider{Factory: func(context.Context, embed.Config) (embed.Embedder, error) {
		return nil, os.ErrNotExist
	}})
}

func TestSetup(t *testing.T) {
	ctx := context.Background()
	home := t.TempDir()
	path := filepath.Join(t.TempDir(), "g.db")
	e, err := engine.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	add(t, e, "A neuron", "x")
	if s, _ := Setup(ctx, e, Config{Embeddings: "off"}, home); s != "embeddings disabled" {
		t.Fatalf("off: %q", s)
	}

	// A running setup (fresh lock) is respected.
	lock := filepath.Join(home, "setup.lock")
	os.WriteFile(lock, []byte("1\n"), 0o644)
	if s, _ := Setup(ctx, e, Config{Embeddings: "fakesetup"}, home); s != "setup already running" {
		t.Fatalf("lock: %q", s)
	}
	// A stale lock is taken over.
	old := time.Now().Add(-2 * staleLock)
	os.Chtimes(lock, old, old)
	s, err := Setup(ctx, e, Config{Embeddings: "fakesetup"}, home)
	if err != nil || !strings.Contains(s, "indexed 1") {
		t.Fatalf("setup: %q %v", s, err)
	}
	if _, err := os.Stat(lock); !os.IsNotExist(err) {
		t.Fatal("lock not released")
	}
	// A neuron added while embeddings were unavailable leaves a gap.
	lexical, err := engine.Open(path, engine.WithoutEmbeddings())
	if err != nil {
		t.Fatal(err)
	}
	add(t, lexical, "Another neuron", "y")
	lexical.Close()
	if s, _ := Setup(ctx, e, Config{Embeddings: "fakesetup"}, home); s != "indexed 1 neurons" {
		t.Fatalf("gap filling: %q", s)
	}
	if s, _ := Setup(ctx, e, Config{Embeddings: "fakesetup"}, home); s != "up to date" {
		t.Fatalf("up to date: %q", s)
	}
	// "auto" keeps an embedder configured by hand.
	if s, _ := Setup(ctx, e, Config{Embeddings: "auto"}, home); s != "up to date" {
		t.Fatalf("auto with a configured embedder: %q", s)
	}
	// Switching to another provider replaces it (here: fails, and backs off).
	if _, err := Setup(ctx, e, Config{Embeddings: "broken"}, home); err == nil {
		t.Fatal("switching provider must try the new one")
	}
	os.Remove(filepath.Join(home, "setup.failed"))
	// "off" turns a configured embedder off, also for later processes.
	if s, _ := Setup(ctx, e, Config{Embeddings: "off"}, home); s != "embeddings turned off" {
		t.Fatalf("off after setup: %q", s)
	}
	later, err := engine.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	st, _ := later.EmbedStatus()
	later.Close()
	if st.Config != nil || st.Active != "" {
		t.Fatalf("embedder still active after off: %+v", st)
	}
	if s, _ := Setup(ctx, e, Config{Embeddings: "off"}, home); s != "embeddings disabled" {
		t.Fatalf("off again: %q", s)
	}

	// Failures back off for a day.
	e2 := newEngine(t)
	home2 := t.TempDir()
	if _, err := Setup(ctx, e2, Config{Embeddings: "broken"}, home2); err == nil {
		t.Fatal("broken provider must fail")
	}
	if s, err := Setup(ctx, e2, Config{Embeddings: "broken"}, home2); err != nil || !strings.Contains(s, "retrying later") {
		t.Fatalf("backoff: %q %v", s, err)
	}
}

func TestCloseAbandonedSkipsFailures(t *testing.T) {
	path := filepath.Join(t.TempDir(), "g.db")
	e, err := engine.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	for _, id := range []string{"broken", "fine"} {
		e.StartSession(id, "/repo", "startup", "")
		e.NotePrompt(id, "/repo", "a prompt about sqlite in "+id)
	}
	// Closing "broken" always fails.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TRIGGER fail BEFORE UPDATE OF ended_at ON sessions WHEN NEW.id = 'broken'
		BEGIN SELECT RAISE(ABORT, 'simulated failure'); END`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	cfg := Config{Embeddings: "off", SessionIdle: -time.Hour, Episodes: true}
	out, err := SessionStart(e, Input{SessionID: "new", Source: "startup"}, "/repo", cfg)
	if err != nil || out == nil || !strings.Contains(out.Specific.AdditionalContext, "This session's ID is new") {
		t.Fatalf("session start must not depend on closing old sessions: %v %+v", err, out)
	}
	n, err := CloseAbandoned(e, cfg, 5)
	if err != nil || n != 2 { // "fine" and the empty "new"
		t.Fatalf("closed %d: %v", n, err)
	}
	if s, _ := e.Session("fine"); s.EndedAt.IsZero() || s.EpisodeID == "" {
		t.Fatalf("session after a failing one not closed: %+v", s)
	}
	evs, _ := e.Events(store.EventFilter{Session: "broken", Kind: engine.EventError})
	if len(evs) != 1 || !strings.Contains(string(evs[0].Payload), `"op":"close-session"`) {
		t.Fatalf("failure not logged: %+v", evs)
	}
}

func TestSessionStartClosesAbandonedSessions(t *testing.T) {
	e := newEngine(t)
	id := add(t, e, "Use SQLite WAL for storage", "x")
	tr := transcript(t, [2]string{"Edit", "/repo/x.go"})
	cfg := Config{Embeddings: "off", Episodes: true}
	SessionStart(e, Input{SessionID: "old", TranscriptPath: tr}, "/repo", cfg)
	Prompt(e, Input{SessionID: "old", Prompt: "tell me about sqlite storage"}, "/repo", Config{RecallLimit: 5, ContextBudget: 3000})
	e.Used("old", []string{id})
	// "old" never got a SessionEnd. A later session start closes it.
	idle := cfg
	idle.SessionIdle = -time.Hour
	if _, err := SessionStart(e, Input{SessionID: "new"}, "/repo", idle); err != nil {
		t.Fatal(err)
	}
	if _, err := CloseAbandoned(e, idle, 3); err != nil {
		t.Fatal(err)
	}
	s, _ := e.Session("old")
	if s.EndedAt.IsZero() || s.EpisodeID == "" {
		t.Fatalf("abandoned session not closed: %+v", s)
	}
	ep, err := e.GetNode(s.EpisodeID)
	if err != nil || !strings.Contains(ep.Content, "x.go") {
		t.Fatalf("episode of abandoned session: %v %+v", err, ep)
	}

	// Days later the user resumes "old": it reopens, and its real end
	// refreshes the same episode with the later prompt and neuron.
	if _, err := e.Reinforce([]string{id, s.EpisodeID}); err != nil {
		t.Fatal(err)
	}
	learned := edgeWeight(t, e, id, s.EpisodeID)
	SessionStart(e, Input{SessionID: "old", Source: "resume"}, "/repo", cfg)
	Prompt(e, Input{SessionID: "old", Prompt: "now add the vacuum schedule"}, "/repo", Config{RecallLimit: 5, ContextBudget: 3000})
	if s, _ := e.Session("old"); !s.EndedAt.IsZero() {
		t.Fatalf("resumed session still closed: %+v", s)
	}
	later := add(t, e, "Vacuum weekly", "y")
	e.Used("old", []string{later})
	rep, err := SessionEnd(e, Input{SessionID: "old", TranscriptPath: tr}, "/repo", cfg)
	if err != nil || rep.Episode == nil || !rep.Updated || rep.Episode.ID != s.EpisodeID {
		t.Fatalf("resumed session end: %v %+v", err, rep)
	}
	if !strings.Contains(rep.Episode.Content, "vacuum schedule") || rep.Links != 1 {
		t.Fatalf("episode not refreshed (links %d): %s", rep.Links, rep.Episode.Content)
	}
	if w := edgeWeight(t, e, id, s.EpisodeID); w != learned {
		t.Fatalf("refresh reset a learned link: %v -> %v", learned, w)
	}
	if s, _ := e.Session("old"); s.EndedAt.IsZero() {
		t.Fatal("resumed session not closed again")
	}
	// Ending it twice changes nothing.
	if rep, err := SessionEnd(e, Input{SessionID: "old"}, "/repo", cfg); err != nil || rep.Episode != nil {
		t.Fatalf("second end: %v %+v", err, rep)
	}
}

func edgeWeight(t *testing.T, e *engine.Engine, a, b string) float64 {
	t.Helper()
	v, err := e.GetNode(a)
	if err != nil {
		t.Fatal(err)
	}
	for _, ed := range v.Edges {
		if ed.Other == b {
			return ed.Weight
		}
	}
	t.Fatalf("no edge %s–%s", a, b)
	return 0
}
