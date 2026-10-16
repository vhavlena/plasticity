package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var binary string

// TestMain builds the CLI once so tests can run it as separate processes,
// exactly as Claude Code hooks would.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "plasticity-bin")
	if err != nil {
		panic(err)
	}
	binary = filepath.Join(dir, "plasticity")
	if out, err := exec.Command("go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		panic(fmt.Sprintf("build: %v\n%s", err, out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

type cli struct {
	t  *testing.T
	db string
}

func newCLI(t *testing.T) *cli {
	return &cli{t: t, db: filepath.Join(t.TempDir(), "g.db")}
}

func (c *cli) run(args ...string) ([]byte, error) {
	cmd := exec.Command(binary, append([]string{"--db", c.db}, args...)...)
	cmd.Stderr = nil
	return cmd.Output()
}

// json runs the command and decodes its JSON output into v.
func (c *cli) json(v any, args ...string) {
	c.t.Helper()
	out, err := c.run(args...)
	if err != nil {
		msg := ""
		if ee, ok := err.(*exec.ExitError); ok {
			msg = string(ee.Stderr)
		}
		c.t.Fatalf("plasticity %v: %v %s", args, err, msg)
	}
	if v != nil {
		if err := json.Unmarshal(out, v); err != nil {
			c.t.Fatalf("plasticity %v: bad JSON %q: %v", args, out, err)
		}
	}
}

func (c *cli) add(title, content string, extra ...string) string {
	var r struct{ Node struct{ ID string } }
	c.json(&r, append([]string{"add", "--title", title, "--content", content}, extra...)...)
	if r.Node.ID == "" {
		c.t.Fatalf("add %q inserted nothing", title)
	}
	return r.Node.ID
}

// hook runs a hook handler with the given input JSON and returns stdout.
func (c *cli) hook(event string, input map[string]any) string {
	c.t.Helper()
	raw, _ := json.Marshal(input)
	cmd := exec.Command(binary, "--db", c.db, "hook", event)
	cmd.Stdin = strings.NewReader(string(raw))
	cmd.Env = append(os.Environ(), "PLASTICITY_EMBEDDINGS=off")
	out, err := cmd.Output()
	if err != nil {
		c.t.Fatalf("hook %s: %v", event, err)
	}
	return string(out)
}

type edgeView struct {
	Other         string
	Kind          string
	Weight        float64
	Coactivations int
}

func (c *cli) edges(id string) map[string]edgeView {
	var v struct{ Edges []edgeView }
	c.json(&v, "get", id)
	m := map[string]edgeView{}
	for _, e := range v.Edges {
		m[e.Other+"/"+e.Kind] = e
	}
	return m
}

func TestLearnRecallForget(t *testing.T) {
	c := newCLI(t)
	c.json(nil, "params", "set", "initial_stability", "2") // forget fast
	a := c.add("SQLite WAL", "concurrent readers")
	b := c.add("Badger lock", "single process")
	c.add("Pasta", "boil water")

	prev := 0.0
	for i := 0; i < 5; i++ {
		c.json(nil, "recall", "sqlite", "badger", "--session", "s")
		var rep struct{ Traces, Pairs int }
		c.json(&rep, "reinforce", "--session", "s")
		if rep.Traces != 1 || rep.Pairs != 1 {
			t.Fatalf("reinforce %d: %+v", i, rep)
		}
		w := c.edges(a)[b+"/associative"].Weight
		if w <= prev || w >= 1 {
			t.Fatalf("step %d: weight %g did not increase below 1", i, w)
		}
		prev = w
	}

	var r struct {
		Hits []struct {
			Node struct{ ID string }
			Hops int
		}
	}
	c.json(&r, "recall", "sqlite", "--no-record")
	if len(r.Hits) != 2 || r.Hits[1].Node.ID != b || r.Hits[1].Hops != 1 {
		t.Fatalf("associative recall failed: %+v", r)
	}

	// Recalling sqlite alone (θ high enough to keep badger out) is
	// interference: the sqlite–badger synapse fades to the floor weight but
	// is never removed.
	for i := 0; i < 15; i++ {
		c.json(nil, "recall", "sqlite", "--threshold", "0.9")
	}
	c.json(nil, "consolidate")
	e, ok := c.edges(a)[b+"/associative"]
	if !ok || e.Weight < 0.05-1e-9 || e.Weight > 0.1 {
		t.Fatalf("interfered association should fade to the floor: %v", c.edges(a))
	}
}

func TestErrors(t *testing.T) {
	c := newCLI(t)
	for _, args := range [][]string{
		{"get", "missing"},
		{"add", "--title", "x", "--type", "bogus"},
		{"add", "--title", ""},
		{"reinforce"},
		{"params", "set", "hop_decay", "2"},
		{"recall", "x", "--rerank", "foo"},
		{"export", "--format", "xml"},
		{"add", "--title", "x", "--link", "nope:supports"},
	} {
		if _, err := c.run(args...); err == nil {
			t.Errorf("plasticity %v: expected failure", args)
		}
	}
}

func TestViz(t *testing.T) {
	c := newCLI(t)
	c.add("SQLite storage", "single file")
	file := filepath.Join(t.TempDir(), "g.html")
	var res struct {
		File   string `json:"file"`
		Opened bool   `json:"opened"`
	}
	c.json(&res, "viz", "--no-open", "-o", file)
	b, err := os.ReadFile(file)
	if err != nil || res.File != file || res.Opened {
		t.Fatalf("viz: %v %+v", err, res)
	}
	if !strings.Contains(string(b), "SQLite storage") || !strings.HasPrefix(string(b), "<!doctype html>") {
		t.Fatal("viz page does not contain the graph")
	}
	out, err := c.run("export", "-f", "html")
	if err != nil || !strings.Contains(string(out), "SQLite storage") {
		t.Fatalf("html export: %v", err)
	}
}

func TestStdinAndLinks(t *testing.T) {
	c := newCLI(t)
	a := c.add("Decision", "use WAL")
	cmd := exec.Command(binary, "--db", c.db, "add", "--stdin", "--link", a+":supports:0.7")
	cmd.Stdin = strings.NewReader(`{"type":"rationale","title":"Why WAL","content":"hooks are processes","tags":["db"]}`)
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	var r struct{ Node struct{ ID, Type string } }
	_ = json.Unmarshal(out, &r)
	n := r.Node
	if n.Type != "rationale" {
		t.Fatalf("stdin node: %s", out)
	}
	e := c.edges(a)[n.ID+"/supports"]
	if e.Weight != 0.7 {
		t.Fatalf("link from --link missing: %v", c.edges(a))
	}
}

// N separate processes recall and reinforce the same pair concurrently; with
// SQLite WAL + busy timeout all must succeed and no update may be lost.
func TestConcurrentProcesses(t *testing.T) {
	c := newCLI(t)
	a := c.add("Alpha neuron", "shared")
	b := c.add("Beta neuron", "shared")
	const n = 12
	var wg sync.WaitGroup
	errs := make(chan error, 2*n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if out, err := c.run("recall", "shared", "neuron", "--session", fmt.Sprint("s", i)); err != nil {
				errs <- fmt.Errorf("recall %d: %v %s", i, err, out)
			}
			if out, err := c.run("reinforce", a, b); err != nil {
				errs <- fmt.Errorf("reinforce %d: %v %s", i, err, out)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got := c.edges(a)[b+"/associative"].Coactivations; got != n {
		t.Fatalf("coactivations = %d, want %d (lost updates)", got, n)
	}
	var st struct {
		PendingTraces int `json:"pending_traces"`
	}
	c.json(&st, "stats")
	if st.PendingTraces != n {
		t.Fatalf("pending traces = %d, want %d", st.PendingTraces, n)
	}
}

func TestEmbedCommands(t *testing.T) {
	c := newCLI(t)
	var st struct {
		Active    string
		Providers []string
	}
	c.json(&st, "embed", "status")
	if st.Active != "" || len(st.Providers) < 2 {
		t.Fatalf("embeddings must be off by default: %+v", st)
	}
	for _, args := range [][]string{
		{"embed", "enable", "nope"},
		{"embed", "reindex"},
		{"embed", "enable", "ollama", "--host", "http://127.0.0.1:1"},
		{"embed", "enable", "ollama", "--opt", "novalue"},
	} {
		if _, err := c.run(args...); err == nil {
			t.Errorf("plasticity %v: expected failure", args)
		}
	}
	c.json(&st, "embed", "status")
	if st.Active != "" {
		t.Fatal("a failed enable must not change the configuration")
	}

	dir := os.Getenv("PLASTICITY_TEST_MODEL")
	if dir == "" {
		t.Skip("set PLASTICITY_TEST_MODEL for the real-model part")
	}
	a := c.add("Use SQLite WAL for storage", "multiple processes can read concurrently while one writes")
	c.add("Cooking pasta", "boil water and add salt")
	var rep struct{ Embedded int }
	c.json(&rep, "embed", "enable", "hugot", "--path", dir)
	if rep.Embedded != 2 {
		t.Fatalf("enable: %+v", rep)
	}
	var r struct {
		Hits []struct {
			Node       struct{ ID string }
			Similarity *float64
		}
	}
	c.json(&r, "recall", "which", "engine", "lets", "several", "programs", "write", "at", "once", "--no-record")
	if len(r.Hits) == 0 || r.Hits[0].Node.ID != a || r.Hits[0].Similarity == nil {
		t.Fatalf("semantic recall via CLI: %+v", r)
	}
	c.json(&r, "recall", "which", "engine", "lets", "several", "programs", "write", "at", "once", "--no-record", "--no-embed")
	for _, h := range r.Hits {
		if h.Similarity != nil {
			t.Fatalf("--no-embed still used vectors: %+v", r)
		}
	}
}

func TestEvalCommands(t *testing.T) {
	c := newCLI(t)
	dir := t.TempDir()
	scen, rep := filepath.Join(dir, "s.json"), filepath.Join(dir, "r.json")
	var g struct{ Nodes, Queries int }
	c.json(&g, "eval", "gen", "--nodes", "60", "--topics", "3", "--sessions", "3", "--queries", "5", "-o", scen)
	if g.Nodes != 60 || g.Queries != 15 {
		t.Fatalf("gen: %+v", g)
	}
	out, err := c.run("eval", "run", scen, "--learn", "none,explicit", "--passes", "2", "--json", rep)
	if err != nil {
		t.Fatalf("eval run: %v\n%s", err, out)
	}
	for _, want := range []string{"none/lexical", "explicit/lexical", "assoc-recall", "recall p50"} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("report lacks %q:\n%s", want, out)
		}
	}
	raw, err := os.ReadFile(rep)
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Results []struct {
			Learn  string
			Passes []struct{ Queries int }
		}
	}
	if err := json.Unmarshal(raw, &r); err != nil || len(r.Results) != 2 || len(r.Results[1].Passes) != 2 || r.Results[1].Passes[0].Queries != 15 {
		t.Fatalf("json report: %v %s", err, raw)
	}
	// The hand-made scenario must stay valid (if present: scenarios/ may be untracked).
	if _, err := os.Stat("../../scenarios/plasticity-design.json"); err == nil {
		if out, err := c.run("eval", "run", "../../scenarios/plasticity-design.json", "--learn", "none", "--passes", "1"); err != nil {
			t.Fatalf("plasticity-design scenario: %v\n%s", err, out)
		}
	}
	if _, err := c.run("eval", "run", scen, "--learn", "bogus"); err == nil {
		t.Fatal("unknown learning mode accepted")
	}
}

func TestAddDuplicateAndLinkValidation(t *testing.T) {
	c := newCLI(t)
	c.add("Use SQLite WAL", "x")
	var r struct {
		Node      *struct{ ID string }
		Duplicate *struct{ ID string } `json:"duplicate_of"`
	}
	c.json(&r, "add", "--title", "use sqlite  WAL", "-c", "y")
	if r.Node != nil || r.Duplicate == nil {
		t.Fatalf("duplicate not reported: %+v", r)
	}
	c.json(&r, "add", "--title", "use sqlite  WAL", "-c", "y", "--force")
	if r.Node == nil {
		t.Fatal("--force must insert")
	}
	if _, err := c.run("add", "--title", "Orphan", "--link", "nope:supports"); err == nil {
		t.Fatal("bad --link target accepted")
	}
	var st struct{ Nodes int }
	c.json(&st, "stats")
	if st.Nodes != 2 {
		t.Fatalf("a failed --link must insert nothing: %d nodes", st.Nodes)
	}
}

// A whole Claude Code session driven through the hook handlers.
func TestHookLifecycle(t *testing.T) {
	c := newCLI(t)
	sqlite := c.add("Use SQLite WAL for storage", "many processes read while one writes", "-r", "hooks are processes")
	dir := t.TempDir()
	tr := filepath.Join(dir, "t.jsonl")
	var lines []string
	for i := 0; i < 6; i++ {
		lines = append(lines, `{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Edit","input":{"file_path":"`+dir+`/main.go"}}]}}`)
	}
	os.WriteFile(tr, []byte(strings.Join(lines, "\n")), 0o644)
	in := func(extra map[string]any) map[string]any {
		m := map[string]any{"session_id": "S1", "cwd": dir, "transcript_path": tr}
		for k, v := range extra {
			m[k] = v
		}
		return m
	}

	var out struct {
		Specific         struct{ HookEventName, AdditionalContext string } `json:"hookSpecificOutput"`
		Decision, Reason string
	}
	json.Unmarshal([]byte(c.hook("session-start", in(map[string]any{"source": "startup"}))), &out)
	if out.Specific.HookEventName != "SessionStart" || !strings.Contains(out.Specific.AdditionalContext, "S1") {
		t.Fatalf("session-start: %+v", out)
	}
	json.Unmarshal([]byte(c.hook("prompt", in(map[string]any{"prompt": "why did we choose sqlite storage?"}))), &out)
	if !strings.Contains(out.Specific.AdditionalContext, sqlite) {
		t.Fatalf("prompt context: %+v", out)
	}
	if got := c.hook("prompt", in(map[string]any{"prompt": "ok thanks"})); got != "" {
		t.Fatalf("trivial prompt produced output: %q", got)
	}

	out = struct {
		Specific         struct{ HookEventName, AdditionalContext string } `json:"hookSpecificOutput"`
		Decision, Reason string
	}{}
	json.Unmarshal([]byte(c.hook("stop", in(nil))), &out)
	if out.Decision != "block" {
		t.Fatalf("expected a reflection nudge: %+v", out)
	}
	if got := c.hook("stop", in(map[string]any{"stop_hook_active": true})); got != "" {
		t.Fatalf("stop while active: %q", got)
	}

	// Claude follows the nudge: stores a decision and reports what it used.
	var added struct {
		Node         struct{ ID string }
		ContextLinks int `json:"context_links"`
	}
	// Claude recalls by itself during the session; the recall is logged.
	var own struct {
		Hits []struct{ Node struct{ ID string } }
	}
	c.json(&own, "recall", "keep", "sqlite", "writes", "short", "--session", "S1")
	if len(own.Hits) == 0 {
		t.Fatalf("Claude's recall found nothing")
	}
	c.json(nil, "used", "--session", "S1", sqlite)
	c.json(&added, "add", "-t", "decision", "--title", "Keep writes short", "-r", "avoid lock waits", "--session", "S1")
	if added.ContextLinks != 1 {
		t.Fatalf("new neuron not linked to the session context: %+v", added)
	}
	c.hook("session-end", in(map[string]any{"reason": "prompt_input_exit"}))

	var evs []struct{ Kind string }
	c.json(&evs, "log", "--session", "S1")
	kinds := map[string]int{}
	for _, e := range evs {
		kinds[e.Kind]++
	}
	for _, k := range []string{"session_start", "prompt", "recall", "nudge", "used", "add"} {
		if kinds[k] == 0 {
			t.Errorf("no %s event: %v", k, kinds)
		}
	}
	if kinds["episode"] != 0 {
		t.Errorf("episode created although episodes are off by default: %v", kinds)
	}
	// The session's history is kept outside the graph.
	var ss []struct {
		ID            string
		FirstPrompt   string `json:"first_prompt"`
		Used, Created []struct{ ID, Title string }
	}
	c.json(&ss, "sessions")
	if len(ss) != 1 || len(ss[0].Used) != 1 || ss[0].Used[0].ID != sqlite ||
		len(ss[0].Created) != 1 || ss[0].Created[0].ID != added.Node.ID || !strings.Contains(ss[0].FirstPrompt, "sqlite") {
		t.Fatalf("sessions: %+v", ss)
	}
	// Bad input never fails the hook (exit 0, message on stderr).
	cmd := exec.Command(binary, "--db", c.db, "hook", "prompt")
	cmd.Stdin = strings.NewReader("not json")
	if err := cmd.Run(); err != nil {
		t.Fatalf("hook must exit 0 on bad input: %v", err)
	}

	// The session is visible in the usage report ...
	var u struct {
		Sessions, Prompts, Recalls, Created, Nudges, Episodes int
		UsedOfInj                                             int                              `json:"used_of_injected"`
		NudgeResponse                                         float64                          `json:"nudge_response"`
		BySource                                              map[string]struct{ Recalls int } `json:"by_source"`
	}
	c.json(&u, "eval", "report", "--json")
	if u.Sessions != 1 || u.Prompts != 2 || u.Recalls != 2 || u.Created != 1 || u.UsedOfInj != 1 ||
		u.Nudges != 1 || u.NudgeResponse != 1 || u.Episodes != 0 ||
		u.BySource["prompt"].Recalls != 1 || u.BySource["claude"].Recalls != 1 {
		t.Fatalf("usage report: %+v", u)
	}
	if out, err := c.run("eval", "report"); err != nil || !strings.Contains(string(out), "usage rate") {
		t.Fatalf("text report: %v\n%s", err, out)
	}
	// ... and exports as a scenario that replays.
	scen := filepath.Join(dir, "sessions.json")
	var ex struct {
		Stats struct{ Queries int }
	}
	c.json(&ex, "eval", "export", "-o", scen)
	if ex.Stats.Queries != 2 { // the prompt recall and Claude's recall
		t.Fatalf("export: %+v", ex)
	}
	if out, err := c.run("eval", "run", scen, "--learn", "none", "--passes", "1"); err != nil || !strings.Contains(string(out), "none") {
		t.Fatalf("replaying the export: %v\n%s", err, out)
	}
}

// The plugin's hooks must call existing hook subcommands, and every handler
// must exit 0 even on minimal input.
func TestPluginHooksAreWired(t *testing.T) {
	raw, err := os.ReadFile("../../hooks/hooks.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Hooks map[string][]struct {
			Hooks []struct{ Type, Command string }
		}
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	c := newCLI(t)
	seen := 0
	for event, groups := range cfg.Hooks {
		for _, g := range groups {
			for _, h := range g.Hooks {
				i := strings.Index(h.Command, " hook ")
				if h.Type != "command" || i < 0 || !strings.Contains(h.Command, "${CLAUDE_PLUGIN_ROOT}/bin/plasticity") {
					t.Fatalf("%s: unexpected hook %+v", event, h)
				}
				name := strings.Fields(h.Command[i+len(" hook "):])[0]
				cmd := exec.Command(binary, "--db", c.db, "hook", name)
				cmd.Stdin = strings.NewReader(`{"session_id":"wired","cwd":"` + t.TempDir() + `","prompt":"is everything wired correctly?"}`)
				cmd.Env = append(os.Environ(), "PLASTICITY_EMBEDDINGS=off", "PLASTICITY_HOME="+t.TempDir())
				var stderr strings.Builder
				cmd.Stderr = &stderr
				if err := cmd.Run(); err != nil {
					t.Fatalf("%s → hook %s: %v %s", event, name, err, stderr.String())
				}
				if strings.Contains(stderr.String(), "unknown command") {
					t.Fatalf("%s → hook %s does not exist: %s", event, name, stderr.String())
				}
				seen++
			}
		}
	}
	if seen < 5 {
		t.Fatalf("only %d hooks wired", seen)
	}
	for _, f := range []string{"../../.claude-plugin/plugin.json", "../../.claude-plugin/marketplace.json"} {
		raw, err := os.ReadFile(f)
		var v map[string]any
		if err != nil || json.Unmarshal(raw, &v) != nil || v["name"] != "plasticity" {
			t.Fatalf("%s: invalid manifest (%v)", f, err)
		}
	}
	if fi, err := os.Stat("../../bin/plasticity"); err != nil || fi.Mode()&0o111 == 0 {
		t.Fatal("bin/plasticity shim missing or not executable")
	}
}

func TestNotesAndChangePeriods(t *testing.T) {
	c := newCLI(t)
	id := c.add("Use SQLite WAL for storage", "many readers")
	in := func(prompt string) map[string]any {
		return map[string]any{"session_id": "S1", "cwd": t.TempDir(), "prompt": prompt}
	}
	c.hook("prompt", in("why sqlite storage for the graph"))
	c.json(nil, "note", "-k", "helped", "-s", "S1", "--id", id, "the WAL decision was right there")
	if _, err := c.run("note", "-k", "bogus", "x"); err == nil {
		t.Fatal("unknown note kind accepted")
	}
	c.json(nil, "params", "set", "min_seed_strength", "0.3")
	c.hook("prompt", in("what storage engine do we use"))
	c.json(nil, "note", "-k", "miss", "-s", "S1", "nothing about the engine choice")

	var ps []struct {
		Change *struct{ Key, Old, New string }
		Usage  struct {
			Recalls int
			Notes   map[string]int
		}
	}
	c.json(&ps, "eval", "report", "--by-change", "--json")
	if len(ps) != 2 || ps[0].Change != nil || ps[0].Usage.Notes["helped"] != 1 ||
		ps[1].Change == nil || ps[1].Change.New != "0.3" || ps[1].Usage.Recalls != 1 || ps[1].Usage.Notes["miss"] != 1 {
		t.Fatalf("periods: %+v", ps)
	}
	out, err := c.run("eval", "report")
	if err != nil || !strings.Contains(string(out), "1 miss") || !strings.Contains(string(out), "min_seed_strength 0.25→0.3") {
		t.Fatalf("text report: %v\n%s", err, out)
	}
}

func TestNoLog(t *testing.T) {
	events := func(c *cli) int {
		var evs []json.RawMessage
		c.json(&evs, "log", "--limit", "100")
		return len(evs)
	}
	c := newCLI(t)
	c.add("Logged", "x", "--session", "S")
	if events(c) == 0 {
		t.Fatal("add --session logged nothing")
	}
	c = newCLI(t)
	c.add("Not logged", "x", "--session", "S", "--no-log")
	c.json(nil, "recall", "not logged", "--session", "S", "--no-log")
	if n := events(c); n != 0 {
		t.Fatalf("--no-log: %d events", n)
	}
	t.Setenv("PLASTICITY_LOG", "false") // inherited by the child processes
	c = newCLI(t)
	c.add("Not logged", "x", "--session", "S")
	if n := events(c); n != 0 {
		t.Fatalf("PLASTICITY_LOG=false: %d events", n)
	}
}
