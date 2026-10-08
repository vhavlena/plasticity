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
)

// batchSize keeps each consolidation write transaction short so concurrent
// recalls are never blocked for long.
const batchSize = 500

// ConsolidateReport summarizes a consolidation ("sleep") pass.
type ConsolidateReport struct {
	EdgesScanned int   `json:"edges_scanned"`
	Pruned       int   `json:"pruned"`
	Scaled       int   `json:"scaled_nodes"`
	Dormant      int   `json:"marked_dormant"`
	Woken        int   `json:"woken"`
	TracesPurged int64 `json:"traces_purged"`
}

// Consolidate prunes synapses weakened by interference, applies homeostatic
// scaling, updates node dormancy and purges old traces. Decay itself needs
// no sweep: it is a function of the endpoints' fire counts.
func (e *Engine) Consolidate(pruneTyped bool) (*ConsolidateReport, error) {
	rep := &ConsolidateReport{}

	// 1. Pruning, paged over all edges.
	var after store.EdgeKey
	for {
		var page []*model.Edge
		err := e.st.Write(func(tx *store.Tx) error {
			var err error
			if page, err = tx.EdgesAfter(after, batchSize); err != nil {
				return err
			}
			for _, ed := range page {
				if plasticity.ShouldPrune(ed, e.P, pruneTyped) {
					if _, err := tx.DeleteEdge(ed.Src, ed.Dst, ed.Kind); err != nil {
						return err
					}
					rep.Pruned++
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

	// 2. Homeostasis + dormancy, paged over all nodes.
	afterID := ""
	for {
		var ids []string
		err := e.st.Write(func(tx *store.Tx) error {
			var err error
			if ids, err = tx.NodeIDsAfter(afterID, batchSize); err != nil || len(ids) == 0 {
				return err
			}
			n, err := e.homeostasis(tx, ids)
			if err != nil {
				return err
			}
			rep.Scaled += n
			return e.updateDormancy(tx, ids, rep)
		})
		if err != nil {
			return rep, err
		}
		if len(ids) < batchSize {
			break
		}
		afterID = ids[len(ids)-1]
	}

	// 3. Trace retention: forget pending traces nobody learned from.
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
func (e *Engine) updateDormancy(tx *store.Tx, ids []string, rep *ConsolidateReport) error {
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
		deg, err := tx.Degree(id)
		if err != nil {
			return err
		}
		b := plasticity.BaseLevel(append(acc[id], n.CreatedTick), now, e.P.ActrDecay)
		dormant := deg == 0 && b < e.P.DormantBelow
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

// Export writes the whole graph as "json" or "dot".
func (e *Engine) Export(w io.Writer, format string) error {
	if format != "json" && format != "dot" {
		return fmt.Errorf("unknown export format %q (json|dot)", format)
	}
	now := timestamp()
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
	if err != nil {
		return err
	}
	if nodes == nil {
		nodes = []*model.Node{}
	}
	if edges == nil {
		edges = []ExportEdge{}
	}
	if format == "json" {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(map[string]any{"exported_at": now, "nodes": nodes, "edges": edges})
	}
	return writeDOT(w, nodes, edges)
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
