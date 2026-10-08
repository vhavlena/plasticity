package retrieval

import "math"

// PersonalizedPageRank computes PPR over the subgraph induced by nodes, with
// transition probabilities proportional to edge weights (edges are treated as
// undirected) and teleport distribution proportional to teleport (as in
// HippoRAG). restart is the teleport probability α. The result sums to 1.
func PersonalizedPageRank(nodes []string, edges []WeightedEdge, teleport map[string]float64, restart float64, tol float64, maxIter int) map[string]float64 {
	idx := make(map[string]int, len(nodes))
	for i, id := range nodes {
		idx[id] = i
	}
	n := len(nodes)
	if n == 0 {
		return map[string]float64{}
	}

	type arc struct {
		to int
		w  float64
	}
	adj := make([][]arc, n)
	outW := make([]float64, n)
	seen := map[[2]int]bool{}
	for _, e := range edges {
		i, ok1 := idx[e.From]
		j, ok2 := idx[e.To]
		if !ok1 || !ok2 || i == j || e.Weight <= 0 {
			continue
		}
		a, b := i, j
		if a > b {
			a, b = b, a
		}
		if seen[[2]int{a, b}] { // the same undirected edge seen from both ends
			continue
		}
		seen[[2]int{a, b}] = true
		adj[i] = append(adj[i], arc{j, e.Weight})
		adj[j] = append(adj[j], arc{i, e.Weight})
		outW[i] += e.Weight
		outW[j] += e.Weight
	}

	tel := make([]float64, n)
	tsum := 0.0
	for id, v := range teleport {
		if i, ok := idx[id]; ok && v > 0 {
			tel[i] = v
			tsum += v
		}
	}
	if tsum == 0 {
		for i := range tel {
			tel[i] = 1
		}
		tsum = float64(n)
	}
	for i := range tel {
		tel[i] /= tsum
	}

	r := append([]float64(nil), tel...)
	next := make([]float64, n)
	for it := 0; it < maxIter; it++ {
		dangling := 0.0
		for i := range next {
			next[i] = 0
		}
		for i := 0; i < n; i++ {
			if outW[i] == 0 {
				dangling += r[i]
				continue
			}
			for _, a := range adj[i] {
				next[a.to] += (1 - restart) * r[i] * a.w / outW[i]
			}
		}
		diff := 0.0
		for i := 0; i < n; i++ {
			// Dangling mass returns to the teleport distribution.
			next[i] += (restart + (1-restart)*dangling) * tel[i]
			diff += math.Abs(next[i] - r[i])
		}
		r, next = next, r
		if diff < tol {
			break
		}
	}
	out := make(map[string]float64, n)
	for i, id := range nodes {
		out[id] = r[i]
	}
	return out
}
