package engine

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
	"github.com/vhavlena/plasticity/internal/version"
	"github.com/vhavlena/plasticity/internal/viz"
)

// batchSize keeps each consolidation write transaction short so concurrent
// recalls are never blocked for long.
const batchSize = 500

// ConsolidateReport summarizes a consolidation ("sleep") pass.
type ConsolidateReport struct {
	EdgesScanned int   `json:"edges_scanned"`
	Scaled       int   `json:"scaled_nodes"`
	Dormant      int   `json:"marked_dormant"`
	Woken        int   `json:"woken"`
	TracesPurged int64 `json:"traces_purged"`
}

// Consolidate applies homeostatic scaling, updates node dormancy and purges
// old traces. Decay itself needs no sweep: it is a function of the
// endpoints' fire counts. Synapses are never removed: weakening stops at
// their weight floor.
func (e *Engine) Consolidate() (*ConsolidateReport, error) {
	rep := &ConsolidateReport{}

	// 1. One pass over all edges: record each node's degree and summed
	// associative weight above the floors, so that steps 2 and 3 need no
	// per-node queries. (Edges written concurrently during consolidation may
	// be missed; dormancy and scaling are advisory and catch up on the next
	// run.)
	degree := map[string]int{}
	assocSum := map[string]float64{}
	var after store.EdgeKey
	for {
		var page []*model.Edge
		err := e.st.Read(func(tx *store.Tx) error {
			var err error
			if page, err = tx.EdgesAfter(after, batchSize); err != nil {
				return err
			}
			for _, ed := range page {
				degree[ed.Src]++
				degree[ed.Dst]++
				if !ed.Kind.Typed() {
					w := plasticity.Excess(ed, e.P)
					assocSum[ed.Src] += w
					assocSum[ed.Dst] += w
				}
			}
			return nil
		})
		if err != nil {
			return rep, err
		}
		rep.EdgesScanned += len(page)
		if len(page) < batchSize {
			break
		}
		last := page[len(page)-1]
		after = store.EdgeKey{Src: last.Src, Dst: last.Dst, Kind: last.Kind}
	}

	// 2. Homeostasis, only for nodes over the limit.
	var over []string
	for id, sum := range assocSum {
		if sum > e.P.MaxOutWeight {
			over = append(over, id)
		}
	}
	sort.Strings(over)
	for start := 0; start < len(over); start += batchSize {
		batch := over[start:min(start+batchSize, len(over))]
		if err := e.st.Write(func(tx *store.Tx) error {
			n, err := e.homeostasis(tx, batch)
			rep.Scaled += n
			return err
		}); err != nil {
			return rep, err
		}
	}

	// 3. Dormancy, paged over all nodes.
	afterID := ""
	for {
		var ids []string
		err := e.st.Write(func(tx *store.Tx) error {
			var err error
			if ids, err = tx.NodeIDsAfter(afterID, batchSize); err != nil || len(ids) == 0 {
				return err
			}
			return e.updateDormancy(tx, ids, degree, rep)
		})
		if err != nil {
			return rep, err
		}
		if len(ids) < batchSize {
			break
		}
		afterID = ids[len(ids)-1]
	}

	// 4. Trace retention: forget pending traces nobody learned from.
	err := e.st.Write(func(tx *store.Tx) error {
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		rep.TracesPurged, err = tx.PurgeTraces(tick - int64(e.P.TraceRetention))
		return err
	})
	return rep, err
}

// updateDormancy marks isolated, rarely accessed nodes dormant (creation
// counts as an access) and wakes dormant nodes that regained synapses.
// degree holds the nodes' edge counts.
func (e *Engine) updateDormancy(tx *store.Tx, ids []string, degree map[string]int, rep *ConsolidateReport) error {
	now, err := tx.Tick()
	if err != nil {
		return err
	}
	nodes, err := tx.GetNodes(ids)
	if err != nil {
		return err
	}
	acc, err := tx.Accesses(ids)
	if err != nil {
		return err
	}
	for _, id := range ids {
		n := nodes[id]
		b := plasticity.BaseLevel(append(acc[id], n.CreatedTick), now, e.P.ActrDecay)
		dormant := degree[id] == 0 && b < e.P.DormantBelow
		if dormant == n.Dormant {
			continue
		}
		if err := tx.SetDormant(id, dormant); err != nil {
			return err
		}
		if dormant {
			rep.Dormant++
		} else {
			rep.Woken++
		}
	}
	return nil
}

// ---------------------------------------------------------------- export

// ExportEdge is an edge with its effective weight at export time.
type ExportEdge struct {
	model.Edge
	EffectiveWeight float64 `json:"effective_weight"`
}

// Export writes the whole graph as "json", "dot" or "html" (an interactive,
// self-contained visualization).
func (e *Engine) Export(w io.Writer, format string) error {
	if format != "json" && format != "dot" && format != "html" {
		return fmt.Errorf("unknown export format %q (json|dot|html)", format)
	}
	now := timestamp()
	nodes, edges, err := e.Snapshot()
	if err != nil {
		return err
	}
	if format == "json" {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"version": version.String(), "exported_at": now, "nodes": nodes, "edges": edges})
	}
	if format == "html" {
		// An associative edge below firing_threshold / hop_decay cannot carry
		// activation on its own; the page draws such tentative links dashed.
		return viz.Write(w, map[string]any{"version": version.String(), "exported_at": now, "nodes": nodes, "edges": edges,
			"propagation_weight": e.P.FiringThreshold / e.P.HopDecay})
	}
	return writeDOT(w, nodes, edges)
}

// Snapshot returns all nodes and edges (with their effective weights).
func (e *Engine) Snapshot() ([]*model.Node, []ExportEdge, error) {
	var nodes []*model.Node
	var edges []ExportEdge
	err := e.st.Read(func(tx *store.Tx) error {
		after := ""
		for {
			ids, err := tx.NodeIDsAfter(after, batchSize)
			if err != nil {
				return err
			}
			m, err := tx.GetNodes(ids)
			if err != nil {
				return err
			}
			for _, id := range ids {
				nodes = append(nodes, m[id])
			}
			if len(ids) < batchSize {
				break
			}
			after = ids[len(ids)-1]
		}
		var ek store.EdgeKey
		for {
			page, err := tx.EdgesAfter(ek, batchSize)
			if err != nil {
				return err
			}
			for _, ed := range page {
				edges = append(edges, ExportEdge{Edge: *ed, EffectiveWeight: plasticity.EffectiveWeight(ed, e.P)})
			}
			if len(page) < batchSize {
				break
			}
			l := page[len(page)-1]
			ek = store.EdgeKey{Src: l.Src, Dst: l.Dst, Kind: l.Kind}
		}
		return nil
	})
	if nodes == nil {
		nodes = []*model.Node{}
	}
	if edges == nil {
		edges = []ExportEdge{}
	}
	return nodes, edges, err
}

var nodeShapes = map[model.NodeType]string{
	model.TypeConcept: "ellipse", model.TypeFact: "box", model.TypeDecision: "diamond",
	model.TypeRationale: "note", model.TypeProcedure: "component", model.TypeEpisode: "folder",
	model.TypeQuestion: "octagon",
}

func writeDOT(w io.Writer, nodes []*model.Node, edges []ExportEdge) error {
	var b strings.Builder
	b.WriteString("graph plasticity {\n  node [fontname=\"Helvetica\"];\n")
	for _, n := range nodes {
		style := ""
		if n.Dormant {
			style = ", style=dashed"
		}
		fmt.Fprintf(&b, "  %q [label=%q, shape=%s%s];\n", n.ID, n.Title, nodeShapes[n.Type], style)
	}
	sort.SliceStable(edges, func(i, j int) bool { return edges[i].EffectiveWeight > edges[j].EffectiveWeight })
	for _, ed := range edges {
		attrs := fmt.Sprintf("penwidth=%.2f, label=\"%.2f\"", 0.5+4*ed.EffectiveWeight, ed.EffectiveWeight)
		if ed.Kind.Typed() {
			attrs += fmt.Sprintf(", dir=forward, style=bold, xlabel=%q", string(ed.Kind))
		} else {
			attrs += ", color=gray40"
		}
		fmt.Fprintf(&b, "  %q -- %q [%s];\n", ed.Src, ed.Dst, attrs)
	}
	b.WriteString("}\n")
	_, err := io.WriteString(w, b.String())
	return err
}

func sortEdgeViews(v []EdgeView) {
	sort.SliceStable(v, func(i, j int) bool {
		if v[i].Weight != v[j].Weight {
			return v[i].Weight > v[j].Weight
		}
		return v[i].Other < v[j].Other
	})
}
