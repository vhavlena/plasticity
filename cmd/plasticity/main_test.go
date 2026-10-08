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
	var n struct{ ID string }
	c.json(&n, append([]string{"add", "--title", title, "--content", content}, extra...)...)
	return n.ID
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

	// Without interfering recalls, consolidation forgets nothing.
	var cons struct{ Pruned int }
	c.json(&cons, "consolidate")
	if cons.Pruned != 0 {
		t.Fatalf("consolidate pruned an unaged synapse: %+v", cons)
	}

	// Recalling sqlite alone (θ high enough to keep badger out) is
	// interference: the sqlite–badger synapse fades and is pruned.
	for i := 0; i < 15; i++ {
		c.json(nil, "recall", "sqlite", "--threshold", "0.9")
	}
	c.json(&cons, "consolidate")
	if cons.Pruned != 1 || len(c.edges(a)) != 0 {
		t.Fatalf("interfered association not forgotten: %+v %v", cons, c.edges(a))
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

func TestStdinAndLinks(t *testing.T) {
	c := newCLI(t)
	a := c.add("Decision", "use WAL")
	cmd := exec.Command(binary, "--db", c.db, "add", "--stdin", "--link", a+":supports:0.7")
	cmd.Stdin = strings.NewReader(`{"type":"rationale","title":"Why WAL","content":"hooks are processes","tags":["db"]}`)
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	var n struct{ ID, Type string }
	_ = json.Unmarshal(out, &n)
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
