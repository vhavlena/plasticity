// Package engine orchestrates storage, retrieval and plasticity: it is the
// API used by the CLI.
package engine

import (
	"fmt"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
)

// Engine is a knowledge graph with plasticity.
type Engine struct {
	st *store.Store
	P  model.Params
}

// Open opens the graph database at path.
func Open(path string) (*Engine, error) {
	st, err := store.Open(path)
	if err != nil {
		return nil, err
	}
	e := &Engine{st: st}
	if err := st.Read(func(tx *store.Tx) error {
		p, err := tx.LoadParams()
		e.P = p
		return err
	}); err != nil {
		st.Close()
		return nil, fmt.Errorf("load params: %w", err)
	}
	return e, nil
}

// Close closes the database.
func (e *Engine) Close() error { return e.st.Close() }

// timestamp is the wall-clock time used for IDs and informational
// timestamps. Memory dynamics never depend on it: they run on recall counts.
func timestamp() time.Time { return time.Now().UTC() }

// ---------------------------------------------------------------- nodes

// AddNode validates and stores a new node, assigning ID and timestamps.
func (e *Engine) AddNode(n *model.Node) (*model.Node, error) {
	now := timestamp()
	n.ID = ulid.MustNew(ulid.Timestamp(now), ulid.DefaultEntropy()).String()
	n.CreatedAt, n.UpdatedAt = now, now
	n.Dormant, n.Fires = false, 0
	if err := n.Validate(); err != nil {
		return nil, err
	}
	return n, e.st.Write(func(tx *store.Tx) error {
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		n.CreatedTick = tick
		return tx.InsertNode(n)
	})
}

// NodePatch holds optional updates; nil fields are left unchanged.
type NodePatch struct {
	Type       *model.NodeType `json:"type,omitempty"`
	Title      *string         `json:"title,omitempty"`
	Content    *string         `json:"content,omitempty"`
	Rationale  *string         `json:"rationale,omitempty"`
	Tags       *[]string       `json:"tags,omitempty"`
	Source     *string         `json:"source,omitempty"`
	Importance *float64        `json:"importance,omitempty"`
}

// UpdateNode applies a patch.
func (e *Engine) UpdateNode(id string, p NodePatch) (*model.Node, error) {
	var out *model.Node
	err := e.st.Write(func(tx *store.Tx) error {
		n, err := tx.GetNode(id)
		if err != nil {
			return err
		}
		if p.Type != nil {
			n.Type = *p.Type
		}
		if p.Title != nil {
			n.Title = *p.Title
		}
		if p.Content != nil {
			n.Content = *p.Content
		}
		if p.Rationale != nil {
			n.Rationale = *p.Rationale
		}
		if p.Tags != nil {
			n.Tags = *p.Tags
		}
		if p.Source != nil {
			n.Source = *p.Source
		}
		if p.Importance != nil {
			n.Importance = *p.Importance
		}
		if err := n.Validate(); err != nil {
			return err
		}
		n.UpdatedAt = timestamp()
		out = n
		return tx.UpdateNode(n)
	})
	return out, err
}

// DeleteNode removes a node and its synapses.
func (e *Engine) DeleteNode(id string) error {
	return e.st.Write(func(tx *store.Tx) error { return tx.DeleteNode(id) })
}

// EdgeView is an edge seen from one node, with its current effective weight.
type EdgeView struct {
	Other         string         `json:"other"`
	OtherTitle    string         `json:"other_title"`
	Kind          model.EdgeKind `json:"kind"`
	Direction     string         `json:"direction"` // "out", "in" or "both"
	Weight        float64        `json:"weight"`
	Age           int64          `json:"age"` // solo endpoint firings since last update
	Stability     float64        `json:"stability"`
	Coactivations int            `json:"coactivations"`
	LastUpdate    time.Time      `json:"last_update"`
}

// NodeView is a node with its neighborhood.
type NodeView struct {
	*model.Node
	BaseLevel *float64   `json:"base_level,omitempty"`
	Accesses  int        `json:"accesses"`
	Edges     []EdgeView `json:"edges"`
}

// GetNode returns a node with its synapses (strongest first).
func (e *Engine) GetNode(id string) (*NodeView, error) {
	var v *NodeView
	err := e.st.Read(func(tx *store.Tx) error {
		n, err := tx.GetNode(id)
		if err != nil {
			return err
		}
		now, err := tx.Tick()
		if err != nil {
			return err
		}
		v = &NodeView{Node: n, Edges: []EdgeView{}}
		acc, err := tx.Accesses([]string{id})
		if err != nil {
			return err
		}
		v.Accesses = len(acc[id])
		if b := plasticity.BaseLevel(acc[id], now, e.P.ActrDecay); len(acc[id]) > 0 {
			v.BaseLevel = &b
		}
		edges, err := tx.EdgesOf(id, 0)
		if err != nil {
			return err
		}
		others := make([]string, len(edges))
		for i, ed := range edges {
			others[i] = ed.Other(id)
		}
		titles, err := tx.GetNodes(others)
		if err != nil {
			return err
		}
		for _, ed := range edges {
			dir := "both"
			if ed.Kind.Typed() {
				dir = map[bool]string{true: "out", false: "in"}[ed.Src == id]
			}
			ev := EdgeView{Other: ed.Other(id), Kind: ed.Kind, Direction: dir,
				Weight: plasticity.EffectiveWeight(ed, e.P), Age: ed.Age(), Stability: ed.Stability,
				Coactivations: ed.Coactivations, LastUpdate: ed.LastUpdate}
			if o := titles[ev.Other]; o != nil {
				ev.OtherTitle = o.Title
			}
			v.Edges = append(v.Edges, ev)
		}
		sortEdgeViews(v.Edges)
		return nil
	})
	return v, err
}

// ---------------------------------------------------------------- edges

// Link creates or overwrites an explicit edge src → dst. For associative
// edges the direction is irrelevant. weight ≤ 0 selects the default.
func (e *Engine) Link(src, dst string, kind model.EdgeKind, weight float64) (*model.Edge, error) {
	if !kind.Valid() {
		return nil, fmt.Errorf("invalid edge kind %q", kind)
	}
	if src == dst {
		return nil, fmt.Errorf("self-loops are not allowed")
	}
	if weight > 1 {
		return nil, fmt.Errorf("weight must be in (0,1], got %g", weight)
	}
	now := timestamp()
	var ed *model.Edge
	err := e.st.Write(func(tx *store.Tx) error {
		nodes, err := mustExist(tx, src, dst)
		if err != nil {
			return err
		}
		ed = plasticity.NewTyped(src, dst, kind, weight, nodes[src].Fires, nodes[dst].Fires, now, e.P)
		if kind == model.KindAssociative {
			ed.Stability = e.P.InitialStability // UpsertEdge canonicalizes the order
		}
		if old, err := tx.GetEdge(ed.Src, ed.Dst, kind); err != nil {
			return err
		} else if old != nil {
			ed.Stability = max(ed.Stability, old.Stability)
			ed.Coactivations = old.Coactivations
		}
		return tx.UpsertEdge(ed)
	})
	return ed, err
}

// Unlink removes an edge.
func (e *Engine) Unlink(src, dst string, kind model.EdgeKind) (bool, error) {
	var ok bool
	err := e.st.Write(func(tx *store.Tx) error {
		var err error
		ok, err = tx.DeleteEdge(src, dst, kind)
		return err
	})
	return ok, err
}

// mustExist loads the given nodes, failing if any of them does not exist.
func mustExist(tx *store.Tx, ids ...string) (map[string]*model.Node, error) {
	found, err := tx.GetNodes(ids)
	if err != nil {
		return nil, err
	}
	var missing []string
	for _, id := range ids {
		if found[id] == nil {
			missing = append(missing, id)
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("unknown node(s) %s: %w", strings.Join(missing, ", "), store.ErrNotFound)
	}
	return found, nil
}

// ---------------------------------------------------------------- params

// SetParam validates and persists a parameter override.
func (e *Engine) SetParam(key, value string) error {
	p, err := e.P.With(map[string]string{key: value})
	if err != nil {
		return err
	}
	if err := e.st.Write(func(tx *store.Tx) error { return tx.SetParam(key, value) }); err != nil {
		return err
	}
	e.P = p
	return nil
}

// ResetParam removes an override, restoring the default.
func (e *Engine) ResetParam(key string) error {
	if _, ok := e.P.ToMap()[key]; !ok {
		return fmt.Errorf("unknown parameter %q", key)
	}
	return e.st.Write(func(tx *store.Tx) error {
		if err := tx.ResetParam(key); err != nil {
			return err
		}
		p, err := tx.LoadParams()
		e.P = p
		return err
	})
}

// Stats returns graph statistics.
func (e *Engine) Stats() (*store.Stats, error) {
	var s *store.Stats
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		s, err = tx.Stats()
		return err
	})
	return s, err
}
