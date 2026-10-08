// Package retrieval implements weight-gated spreading activation and
// personalized PageRank over an abstract, lazily-loaded graph.
package retrieval

import (
	"container/heap"
	"math"
	"sort"
)

// Neighbor is an edge seen from one endpoint; Weight is the effective
// (already decayed) weight in [0,1].
type Neighbor struct {
	ID     string
	Weight float64
	Kind   string
}

// Graph provides lazy access to the neighborhood of a node. Degree is the
// total number of edges of the node (used for the fan effect), which may be
// larger than len(neighbors) if the caller truncated the list.
type Graph interface {
	Neighbors(id string) (neighbors []Neighbor, degree int, err error)
}

// SpreadConfig parameterizes spreading activation.
type SpreadConfig struct {
	HopDecay  float64 // δ: per-hop attenuation
	Threshold float64 // θ: minimum activation for a node to be reached
	FanK0     float64 // fan effect: divide by max(1, degree/K0)^β
	FanBeta   float64
	MaxNodes  int // max distinct nodes expanded (neighborhoods loaded)
	MaxHops   int // safety cap on path length
}

// Activation is the result for one reached node.
type Activation struct {
	ID string `json:"id"`
	// Score is the noisy-OR combination over seeds: 1 − ∏(1 − a_seed).
	Score float64 `json:"activation"`
	// Seed is the seed whose strongest path reached this node with the
	// highest activation; Path is that path (seed first, node last).
	Seed string   `json:"seed"`
	Path []string `json:"path"`
	Hops int      `json:"hops"`
	// IsSeed reports whether the node was itself a lexical seed.
	IsSeed bool `json:"is_seed"`
}

// Result of spreading activation.
type Result struct {
	Activations map[string]*Activation
	// Edges are all edges observed while expanding nodes, as
	// (from, to, weight) with from an expanded node. Used for PPR reranking.
	Edges []WeightedEdge
	// Expanded is the number of neighborhoods loaded.
	Expanded int
}

// WeightedEdge is a directed weighted edge.
type WeightedEdge struct {
	From, To string
	Weight   float64
}

// memoGraph caches neighborhoods and enforces the expansion budget.
type memoGraph struct {
	g      Graph
	budget int
	cache  map[string]memoEntry
}

type memoEntry struct {
	nbrs []Neighbor
	deg  int
}

// get returns the neighborhood of id, or ok=false if the budget is exhausted.
func (m *memoGraph) get(id string) (memoEntry, bool, error) {
	if e, ok := m.cache[id]; ok {
		return e, true, nil
	}
	if len(m.cache) >= m.budget {
		return memoEntry{}, false, nil
	}
	nbrs, deg, err := m.g.Neighbors(id)
	if err != nil {
		return memoEntry{}, false, err
	}
	if deg < len(nbrs) {
		deg = len(nbrs)
	}
	e := memoEntry{nbrs: nbrs, deg: deg}
	m.cache[id] = e
	return e, true, nil
}

// Spread runs weight-gated spreading activation from the given seeds (seed
// id → initial activation in [0,1]).
//
// For each seed, a best-first search (Dijkstra on −ln a) computes the
// strongest path to every node, where traversing an edge of weight w from a
// node of degree k multiplies the activation by
//
//	δ · w / max(1, k/K0)^β.
//
// Because every factor is ≤ 1, activation is monotonically non-increasing
// along a path, so the search can stop as soon as it would fall below θ:
// strongly connected regions are explored deeply, weakly connected ones only
// one hop or not at all. Per-seed activations are combined by noisy-OR.
func Spread(g Graph, seeds map[string]float64, cfg SpreadConfig) (*Result, error) {
	mg := &memoGraph{g: g, budget: cfg.MaxNodes, cache: map[string]memoEntry{}}

	type perSeed struct {
		a      map[string]float64
		parent map[string]string
		hops   map[string]int
	}
	runs := map[string]*perSeed{}

	// Deterministic seed order: strongest first, so a limited budget is
	// spent on the best seeds.
	order := make([]string, 0, len(seeds))
	for id, s := range seeds {
		if s >= cfg.Threshold {
			order = append(order, id)
		}
	}
	sort.Slice(order, func(i, j int) bool {
		if seeds[order[i]] != seeds[order[j]] {
			return seeds[order[i]] > seeds[order[j]]
		}
		return order[i] < order[j]
	})

	for _, seed := range order {
		run := &perSeed{a: map[string]float64{}, parent: map[string]string{}, hops: map[string]int{}}
		runs[seed] = run
		s := math.Min(1, seeds[seed])
		run.a[seed] = s
		pq := &maxHeap{{id: seed, a: s}}
		for pq.Len() > 0 {
			it := heap.Pop(pq).(item)
			if it.a < run.a[it.id] { // stale entry
				continue
			}
			if run.hops[it.id] >= cfg.MaxHops {
				continue
			}
			e, ok, err := mg.get(it.id)
			if err != nil {
				return nil, err
			}
			if !ok {
				continue // budget exhausted: node stays a leaf
			}
			fan := math.Pow(math.Max(1, float64(e.deg)/cfg.FanK0), cfg.FanBeta)
			for _, nb := range e.nbrs {
				a := it.a * cfg.HopDecay * nb.Weight / fan
				if a < cfg.Threshold || a <= run.a[nb.ID] {
					continue
				}
				run.a[nb.ID] = a
				run.parent[nb.ID] = it.id
				run.hops[nb.ID] = run.hops[it.id] + 1
				heap.Push(pq, item{id: nb.ID, a: a})
			}
		}
	}

	res := &Result{Activations: map[string]*Activation{}, Expanded: len(mg.cache)}
	best := map[string]float64{}
	for _, seed := range order {
		run := runs[seed]
		for id, a := range run.a {
			act := res.Activations[id]
			if act == nil {
				act = &Activation{ID: id}
				res.Activations[id] = act
			}
			act.Score = 1 - (1-act.Score)*(1-a)
			if a > best[id] {
				best[id] = a
				act.Seed = seed
				act.Hops = run.hops[id]
				act.Path = tracePath(run.parent, seed, id)
			}
		}
	}
	for id := range seeds {
		if act := res.Activations[id]; act != nil {
			act.IsSeed = true
		}
	}

	ids := make([]string, 0, len(mg.cache))
	for id := range mg.cache {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		for _, nb := range mg.cache[id].nbrs {
			res.Edges = append(res.Edges, WeightedEdge{From: id, To: nb.ID, Weight: nb.Weight})
		}
	}
	return res, nil
}

func tracePath(parent map[string]string, seed, id string) []string {
	path := []string{id}
	for cur := id; cur != seed; {
		cur = parent[cur]
		path = append(path, cur)
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

type item struct {
	id string
	a  float64
}

type maxHeap []item

func (h maxHeap) Len() int { return len(h) }
func (h maxHeap) Less(i, j int) bool {
	if h[i].a != h[j].a {
		return h[i].a > h[j].a
	}
	return h[i].id < h[j].id
}
func (h maxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)   { *h = append(*h, x.(item)) }
func (h *maxHeap) Pop() any {
	old := *h
	it := old[len(old)-1]
	*h = old[:len(old)-1]
	return it
}
