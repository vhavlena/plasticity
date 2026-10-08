package retrieval

import (
	"fmt"
	"math"
	"testing"
)

// mapGraph is an undirected in-memory test graph.
type mapGraph struct {
	adj   map[string][]Neighbor
	calls int
}

func newGraph() *mapGraph { return &mapGraph{adj: map[string][]Neighbor{}} }

func (g *mapGraph) link(a, b string, w float64) *mapGraph {
	g.adj[a] = append(g.adj[a], Neighbor{ID: b, Weight: w})
	g.adj[b] = append(g.adj[b], Neighbor{ID: a, Weight: w})
	return g
}

func (g *mapGraph) Neighbors(id string) ([]Neighbor, int, error) {
	g.calls++
	return g.adj[id], len(g.adj[id]), nil
}

func cfg() SpreadConfig {
	return SpreadConfig{HopDecay: 0.8, Threshold: 0.1, FanK0: 8, FanBeta: 0.5, MaxNodes: 1000, MaxHops: 50}
}

// chain builds seed - p1 - p2 - ... - pn with equal weights.
func chain(g *mapGraph, prefix string, n int, w float64) {
	prev := "seed"
	for i := 1; i <= n; i++ {
		id := fmt.Sprintf("%s%d", prefix, i)
		g.link(prev, id, w)
		prev = id
	}
}

func depth(res *Result, prefix string, n int) int {
	d := 0
	for i := 1; i <= n; i++ {
		if _, ok := res.Activations[fmt.Sprintf("%s%d", prefix, i)]; ok {
			d = i
		}
	}
	return d
}

func TestStrongConnectionsReachDeeper(t *testing.T) {
	g := newGraph()
	chain(g, "strong", 20, 0.9)
	chain(g, "mid", 20, 0.6)
	chain(g, "weak", 20, 0.3)
	res, err := Spread(g, map[string]float64{"seed": 1}, cfg())
	if err != nil {
		t.Fatal(err)
	}
	// The seed has degree 3 < K0, so no fan penalty. Per-hop factor is
	// 0.8·w; but interior chain nodes have degree 2, also no penalty.
	want := map[string]int{
		"strong": int(math.Floor(math.Log(0.1) / math.Log(0.72))), // 7
		"mid":    int(math.Floor(math.Log(0.1) / math.Log(0.48))), // 3
		"weak":   int(math.Floor(math.Log(0.1) / math.Log(0.24))), // 1
	}
	for prefix, w := range want {
		if got := depth(res, prefix, 20); got != w {
			t.Errorf("%s chain reached depth %d, want %d", prefix, got, w)
		}
	}
	if !(depth(res, "strong", 20) > depth(res, "mid", 20) && depth(res, "mid", 20) > depth(res, "weak", 20)) {
		t.Error("depth must increase with connection strength")
	}
}

func TestThresholdIsExact(t *testing.T) {
	// a = 1·0.8·w; with w = 0.125 → exactly 0.1 (reached), w = 0.124 → not.
	g := newGraph().link("seed", "in", 0.125).link("seed", "out", 0.124)
	res, _ := Spread(g, map[string]float64{"seed": 1}, cfg())
	if _, ok := res.Activations["in"]; !ok {
		t.Error("node at exactly θ must be reached")
	}
	if _, ok := res.Activations["out"]; ok {
		t.Error("node below θ must not be reached")
	}
}

func TestStrongestPathWins(t *testing.T) {
	// seed -0.5- x directly, or seed -0.9- m -0.9- x: 0.8·0.9·0.8·0.9 = 0.5184 > 0.4
	g := newGraph().link("seed", "x", 0.5).link("seed", "m", 0.9).link("m", "x", 0.9)
	res, _ := Spread(g, map[string]float64{"seed": 1}, cfg())
	x := res.Activations["x"]
	if x == nil || x.Hops != 2 || len(x.Path) != 3 || x.Path[1] != "m" {
		t.Fatalf("expected strongest 2-hop path via m, got %+v", x)
	}
	if math.Abs(x.Score-0.5184) > 1e-9 {
		t.Fatalf("activation %g, want 0.5184", x.Score)
	}
}

func TestNoisyOrMultiSeedBoost(t *testing.T) {
	g := newGraph().link("s1", "shared", 0.5).link("s2", "shared", 0.5).link("s1", "only1", 0.5)
	res, _ := Spread(g, map[string]float64{"s1": 1, "s2": 1}, cfg())
	shared, only := res.Activations["shared"].Score, res.Activations["only1"].Score
	if !(shared > only) {
		t.Fatalf("node reached from two seeds (%g) should beat one seed (%g)", shared, only)
	}
	if want := 1 - (1-0.4)*(1-0.4); math.Abs(shared-want) > 1e-9 {
		t.Fatalf("noisy-OR %g, want %g", shared, want)
	}
	if !res.Activations["s1"].IsSeed || res.Activations["shared"].IsSeed {
		t.Fatal("IsSeed flags wrong")
	}
}

func TestHubFanEffect(t *testing.T) {
	g := newGraph().link("seed", "hub", 0.9).link("seed", "lean", 0.9)
	for i := 0; i < 64; i++ {
		g.link("hub", fmt.Sprintf("h%d", i), 0.9)
	}
	g.link("lean", "leanchild", 0.9)
	res, _ := Spread(g, map[string]float64{"seed": 1}, cfg())
	hc, lc := res.Activations["h0"], res.Activations["leanchild"]
	if lc == nil {
		t.Fatal("lean child should be reached")
	}
	if hc != nil && !(hc.Score < lc.Score) {
		t.Fatalf("hub children (%g) should be dampened below lean child (%g)", hc.Score, lc.Score)
	}
}

func TestBudgetLimitsExpansion(t *testing.T) {
	g := newGraph()
	chain(g, "c", 100, 1.0)
	c := cfg()
	c.HopDecay = 1
	c.MaxNodes = 5
	res, _ := Spread(g, map[string]float64{"seed": 1}, c)
	if g.calls > 5 || res.Expanded > 5 {
		t.Fatalf("expanded %d nodes (%d calls), budget 5", res.Expanded, g.calls)
	}
	c.MaxNodes = 1000
	c.MaxHops = 3
	res, _ = Spread(newGraphChain(), map[string]float64{"seed": 1}, c)
	if _, ok := res.Activations["c4"]; ok {
		t.Fatal("max hops not enforced")
	}
}

func newGraphChain() *mapGraph {
	g := newGraph()
	chain(g, "c", 10, 1.0)
	return g
}

func TestWeakSeedIgnored(t *testing.T) {
	g := newGraph().link("s", "x", 1)
	res, _ := Spread(g, map[string]float64{"s": 0.05}, cfg())
	if len(res.Activations) != 0 {
		t.Fatalf("seed below θ should not fire: %v", res.Activations)
	}
}

func TestPPR(t *testing.T) {
	nodes := []string{"a", "b", "c", "d"}
	edges := []WeightedEdge{{"a", "b", 1}, {"b", "a", 1}, {"b", "c", 0.1}, {"c", "d", 1}}
	r := PersonalizedPageRank(nodes, edges, map[string]float64{"a": 1}, 0.15, 1e-10, 1000)
	sum := 0.0
	for _, v := range r {
		sum += v
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("PPR mass %g != 1", sum)
	}
	if !(r["a"] > r["b"] && r["b"] > r["c"] && r["b"] > r["d"]) {
		t.Fatalf("unexpected PPR order: %v", r)
	}
	if len(PersonalizedPageRank(nil, nil, nil, 0.15, 1e-6, 10)) != 0 {
		t.Fatal("empty graph")
	}
}
