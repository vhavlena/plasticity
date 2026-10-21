package engine

import (
	"math"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// DemotedEdge is a context link lowered by DemoteUnconfirmed.
type DemotedEdge struct {
	Src      string  `json:"src"`
	Dst      string  `json:"dst"`
	SrcTitle string  `json:"src_title"`
	DstTitle string  `json:"dst_title"`
	From     float64 `json:"from"`
	To       float64 `json:"to"`
}

// DemoteReport lists the links DemoteUnconfirmed changed (or would change).
type DemoteReport struct {
	DryRun  bool          `json:"dry_run"`
	Demoted int           `json:"demoted"`
	Edges   []DemotedEdge `json:"edges"`
}

// DemoteUnconfirmed lowers unconfirmed context links made before they were
// tentative: associative edges never co-used (coactivations = 0) whose stored
// weight is still the old context link weight from, to weight to. With
// dryRun nothing is written.
func (e *Engine) DemoteUnconfirmed(from, to float64, dryRun bool) (*DemoteReport, error) {
	rep := &DemoteReport{DryRun: dryRun, Edges: []DemotedEdge{}}
	write := e.st.Write
	if dryRun {
		write = e.st.Read
	}
	err := write(func(tx *store.Tx) error {
		var found []*model.Edge
		var ek store.EdgeKey
		for {
			page, err := tx.EdgesAfter(ek, batchSize)
			if err != nil {
				return err
			}
			for _, ed := range page {
				if ed.Kind == model.KindAssociative && ed.Coactivations == 0 && math.Abs(ed.Weight-from) < 1e-9 {
					found = append(found, ed)
				}
			}
			if len(page) < batchSize {
				break
			}
			l := page[len(page)-1]
			ek = store.EdgeKey{Src: l.Src, Dst: l.Dst, Kind: l.Kind}
		}
		ids := make([]string, 0, 2*len(found))
		for _, ed := range found {
			ids = append(ids, ed.Src, ed.Dst)
		}
		nodes, err := tx.GetNodes(ids)
		if err != nil {
			return err
		}
		title := func(id string) string {
			if n := nodes[id]; n != nil {
				return n.Title
			}
			return ""
		}
		for _, ed := range found {
			rep.Edges = append(rep.Edges, DemotedEdge{Src: ed.Src, Dst: ed.Dst, SrcTitle: title(ed.Src), DstTitle: title(ed.Dst), From: ed.Weight, To: to})
			if dryRun {
				continue
			}
			ed.Weight = to
			if err := tx.UpsertEdge(ed); err != nil {
				return err
			}
		}
		rep.Demoted = len(found)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if !dryRun && rep.Demoted > 0 {
		e.Log("", EventMaintain, map[string]any{"op": "demote-unconfirmed", "from": from, "to": to, "demoted": rep.Demoted})
	}
	return rep, nil
}
