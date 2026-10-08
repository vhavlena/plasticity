package engine

import (
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
)

// LearnReport summarizes a learning step.
type LearnReport struct {
	Traces      int `json:"traces"`
	Pairs       int `json:"pairs"`
	Created     int `json:"created"`
	Potentiated int `json:"potentiated"`
	Depressed   int `json:"depressed"`
	// Dependent counts co-activated pairs not learned from because one
	// node's activation was caused by the other or by the same seed.
	Dependent int `json:"skipped_dependent"`
	Scaled    int `json:"scaled_nodes"`
}

// activated is a node with its activation in a co-activation event. path is
// the activation path from its seed (nil for explicitly reported use).
type activated struct {
	id   string
	a    float64
	path []string
}

// independent reports whether the co-activation of a and b is evidence of
// an association, rather than an echo of the graph's existing structure.
// That requires the two nodes to come from different seeds, with neither
// lying on the other's activation path; otherwise one node fired because of
// the other (or both because of a common seed) and learning would be circular.
// Explicitly reported use (no paths) is always independent.
func independent(a, b activated) bool {
	if len(a.path) == 0 || len(b.path) == 0 {
		return true
	}
	return a.path[0] != b.path[0] && !slices.Contains(a.path, b.id) && !slices.Contains(b.path, a.id)
}

// coactivate applies Hebbian LTP to every pair of the given nodes.
func (e *Engine) coactivate(tx *store.Tx, nodes []activated, now time.Time, rep *LearnReport, touched map[string]bool) error {
	ids := make([]string, len(nodes))
	for i, n := range nodes {
		ids[i] = n.id
	}
	fires, err := tx.GetNodes(ids) // current fire counts for new synapses
	if err != nil {
		return err
	}
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			a, b := nodes[i], nodes[j]
			if a.id == b.id {
				continue
			}
			if !independent(a, b) {
				rep.Dependent++
				continue
			}
			ed, err := tx.GetEdge(a.id, b.id, model.KindAssociative)
			if err != nil {
				return err
			}
			if ed == nil {
				fa, fb := fires[a.id], fires[b.id]
				if fa == nil || fb == nil {
					continue // node deleted since the trace was recorded
				}
				ed = plasticity.NewAssociative(a.id, b.id, fa.Fires, fb.Fires, now, e.P)
				rep.Created++
			} else {
				rep.Potentiated++
			}
			plasticity.Potentiate(ed, a.a, b.a, now, e.P)
			if err := tx.UpsertEdge(ed); err != nil {
				return err
			}
			rep.Pairs++
			touched[a.id], touched[b.id] = true, true
		}
	}
	return nil
}

// homeostasis rescales the associative synapses of each touched node whose
// summed effective weight exceeds MaxOutWeight.
func (e *Engine) homeostasis(tx *store.Tx, ids []string) (int, error) {
	scaled := 0
	for _, id := range ids {
		edges, err := tx.EdgesOf(id, 0)
		if err != nil {
			return scaled, err
		}
		var assoc []*model.Edge
		total := 0.0
		for _, ed := range edges {
			if ed.Kind == model.KindAssociative {
				assoc = append(assoc, ed)
				total += plasticity.EffectiveWeight(ed, e.P)
			}
		}
		f := plasticity.ScaleFactor(total, e.P.MaxOutWeight)
		if f >= 1 {
			continue
		}
		for _, ed := range assoc {
			plasticity.Materialize(ed, e.P)
			ed.Weight *= f
			if err := tx.UpsertEdge(ed); err != nil {
				return scaled, err
			}
		}
		scaled++
	}
	return scaled, nil
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// topK keeps the k most activated entries.
func topK(entries []store.TraceEntry, k int) []activated {
	sorted := append([]store.TraceEntry(nil), entries...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Activation > sorted[j].Activation })
	if len(sorted) > k {
		sorted = sorted[:k]
	}
	out := make([]activated, len(sorted))
	for i, en := range sorted {
		out[i] = activated{id: en.NodeID, a: en.Activation, path: en.Path}
	}
	return out
}

// Reinforce applies Hebbian learning to an explicit set of nodes used
// together (each with activation 1).
func (e *Engine) Reinforce(ids []string) (*LearnReport, error) {
	if len(ids) < 2 {
		return nil, fmt.Errorf("reinforce needs at least two nodes")
	}
	rep := &LearnReport{}
	now := timestamp()
	err := e.st.Write(func(tx *store.Tx) error {
		if _, err := mustExist(tx, ids...); err != nil {
			return err
		}
		nodes := make([]activated, len(ids))
		for i, id := range ids {
			nodes[i] = activated{id: id, a: 1}
		}
		touched := map[string]bool{}
		if err := e.coactivate(tx, nodes, now, rep, touched); err != nil {
			return err
		}
		var err error
		rep.Scaled, err = e.homeostasis(tx, sortedKeys(touched))
		return err
	})
	return rep, err
}

// ReinforceSession learns from (and deletes) the session's pending recall traces: the nodes
// independently co-activated in each recall are associated with LTP
// proportional to the product of their activations.
func (e *Engine) ReinforceSession(session string) (*LearnReport, error) {
	rep := &LearnReport{}
	now := timestamp()
	err := e.st.Write(func(tx *store.Tx) error {
		traces, err := tx.PendingTraces(session)
		if err != nil {
			return err
		}
		touched := map[string]bool{}
		ids := make([]string, 0, len(traces))
		for _, tr := range traces {
			if err := e.coactivate(tx, topK(tr.Entries, e.P.ReinforceTopK), now, rep, touched); err != nil {
				return err
			}
			ids = append(ids, tr.RecallID)
		}
		rep.Traces = len(traces)
		if err := tx.DeleteTraces(ids); err != nil {
			return err
		}
		rep.Scaled, err = e.homeostasis(tx, sortedKeys(touched))
		return err
	})
	return rep, err
}

// Feedback tells the graph which nodes were actually useful in the session.
// Used nodes are potentiated with each other (activation 1); nodes recalled
// in the session's pending traces but not used are depressed (LTD) on their
// existing associative links to the used nodes. Pending traces are deleted.
func (e *Engine) Feedback(session string, used []string) (*LearnReport, error) {
	if len(used) == 0 {
		return nil, fmt.Errorf("feedback needs at least one used node")
	}
	rep := &LearnReport{}
	now := timestamp()
	err := e.st.Write(func(tx *store.Tx) error {
		if _, err := mustExist(tx, used...); err != nil {
			return err
		}
		usedSet := map[string]bool{}
		usedNodes := make([]activated, 0, len(used))
		for _, id := range used {
			if !usedSet[id] {
				usedSet[id] = true
				usedNodes = append(usedNodes, activated{id: id, a: 1})
			}
		}

		traces, err := tx.PendingTraces(session)
		if err != nil {
			return err
		}
		unused := map[string]float64{} // recalled but not used → max activation
		recallIDs := make([]string, 0, len(traces))
		for _, tr := range traces {
			recallIDs = append(recallIDs, tr.RecallID)
			for _, en := range tr.Entries {
				if !usedSet[en.NodeID] && en.Activation > unused[en.NodeID] {
					unused[en.NodeID] = en.Activation
				}
			}
		}
		rep.Traces = len(traces)

		touched := map[string]bool{}
		if err := e.coactivate(tx, usedNodes, now, rep, touched); err != nil {
			return err
		}
		for _, r := range sortedKeys(unused) {
			for _, u := range usedNodes {
				ed, err := tx.GetEdge(r, u.id, model.KindAssociative)
				if err != nil {
					return err
				}
				if ed == nil {
					continue
				}
				plasticity.Depress(ed, unused[r], u.a, now, e.P)
				if err := tx.UpsertEdge(ed); err != nil {
					return err
				}
				rep.Depressed++
			}
		}
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		for _, u := range usedNodes {
			if err := tx.RecordAccess(u.id, tick, e.P.MaxAccessRecords); err != nil {
				return err
			}
		}
		if err := tx.DeleteTraces(recallIDs); err != nil {
			return err
		}
		rep.Scaled, err = e.homeostasis(tx, sortedKeys(touched))
		return err
	})
	return rep, err
}
