package hooks

import (
	"context"
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
	for _, k := range []string{"PLASTICITY_RECALL_LIMIT", "PLASTICITY_REFLECT", "PLASTICITY_EMBEDDINGS", "PLASTICITY_SCOPE", "PLASTICITY_LOG"} {
		t.Setenv(k, "")
	}
	t.Setenv("CLAUDE_PLUGIN_OPTION_RECALL_LIMIT", "4")
	t.Setenv("CLAUDE_PLUGIN_OPTION_reflect", "false")
	t.Setenv("PLASTICITY_EMBEDDINGS", "off")
	c := ConfigFromEnv()
	if c.RecallLimit != 4 || c.Reflect || c.Embeddings != "off" || !c.Log || c.Scope != "global" {
		t.Fatalf("config: %+v", c)
	}
}

func TestSessionStartAndPrompt(t *testing.T) {
	e := newEngine(t)
	cfg := Config{Scope: "global", RecallLimit: 5, Reflect: true, Log: true, Embeddings: "off", ContextBudget: 3000}
	sqlite := add(t, e, "Use SQLite WAL for storage", "many processes read while one writes")
	add(t, e, "Cooking pasta", "boil water")

	out, err := SessionStart(e, Input{SessionID: "S1", Source: "startup"}, "/repo", cfg)
	if err != nil || out.Specific.HookEventName != "SessionStart" {
		t.Fatalf("session start: %v %+v", err, out)
	}
	ctx := out.Specific.AdditionalContext
	for _, want := range []string{"2 neurons", "S1", "plasticity add", "plasticity used --session S1"} {
		if !strings.Contains(ctx, want) {
			t.Fatalf("session-start context lacks %q:\n%s", want, ctx)
		}
	}

	out, err = Prompt(e, Input{SessionID: "S1", Prompt: "why did we pick sqlite storage?"}, "/repo", cfg)
	if err != nil || out == nil || out.Specific.HookEventName != "UserPromptSubmit" {
		t.Fatalf("prompt: %v %+v", err, out)
	}
	ctx = out.Specific.AdditionalContext
	if !strings.Contains(ctx, sqlite) || !strings.Contains(ctx, "Why: because") || strings.Contains(ctx, "pasta") {
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
	text := FormatMemories(r, "S", 1500)
	if len(text) > 1500 || !strings.HasSuffix(text, "</plasticity-memories>") || strings.Count(text, "- [") < 2 {
		t.Fatalf("budget not respected (%d chars):\n%s", len(text), text)
	}
	if FormatMemories(&engine.RecallResult{}, "S", 1500) != "" {
		t.Fatal("no hits must give no context")
	}
}

func TestStopNudge(t *testing.T) {
	e := newEngine(t)
	cfg := Config{Reflect: true}
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
	if err != nil || out == nil || out.Decision != "block" || !strings.Contains(out.Reason, "--session S") {
		t.Fatalf("expected nudge: %v %+v", err, out)
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
	rep, err := SessionEnd(e, Input{SessionID: "S", TranscriptPath: path}, "/repo")
	if err != nil || rep.Episode == nil || !strings.Contains(rep.Episode.Content, "sub/b.go") || rep.Links != 1 {
		t.Fatalf("session end: %v %+v", err, rep)
	}
	rep, err = SessionEnd(e, Input{SessionID: "unknown", TranscriptPath: "/nope"}, "/repo")
	if err != nil || rep != nil {
		t.Fatalf("unknown session: %v %+v", err, rep)
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

func TestSessionStartClosesAbandonedSessions(t *testing.T) {
	e := newEngine(t)
	id := add(t, e, "Use SQLite WAL for storage", "x")
	tr := transcript(t, [2]string{"Edit", "/repo/x.go"})
	cfg := Config{Embeddings: "off"}
	SessionStart(e, Input{SessionID: "old", TranscriptPath: tr}, "/repo", cfg)
	Prompt(e, Input{SessionID: "old", Prompt: "tell me about sqlite storage"}, "/repo", Config{RecallLimit: 5, ContextBudget: 3000})
	e.Used("old", []string{id})
	// "old" never got a SessionEnd. A later session start closes it.
	defer func(d time.Duration) { abandonedAfter = d }(abandonedAfter)
	abandonedAfter = -time.Hour
	if _, err := SessionStart(e, Input{SessionID: "new"}, "/repo", cfg); err != nil {
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
}
