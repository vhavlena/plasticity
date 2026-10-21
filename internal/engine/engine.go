// Package engine orchestrates storage, retrieval and plasticity: it is the
// API used by the CLI.
package engine

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
)

// Engine is a knowledge graph with plasticity.
type Engine struct {
	st   *store.Store
	path string
	P    model.Params

	embCfg   *embed.Config  // stored configuration; nil: none
	embID    string         // ID reported by the configured embedder
	emb      embed.Embedder // active embedder, loaded lazily from embCfg
	embOwned bool           // the engine created emb and must close it
	injected bool           // emb was supplied by WithEmbedder
	noEmbed  bool           // WithoutEmbeddings: lexical seeding only
	noLog    bool           // WithoutLog: no event log

	// afterEpisode, if set, runs in EndSession right after the episode is
	// stored (tests: simulates the hook being killed at that point).
	afterEpisode func() error

	// Warnings collects non-fatal problems (e.g. an unreachable embedder)
	// for the caller to report.
	Warnings []string
}

// Option customizes an engine at Open.
type Option func(*Engine)

// WithEmbedder makes the engine use emb for semantic search and indexing,
// overriding the embedder configured in the database. The caller keeps
// ownership: the engine does not close it. Use it to plug in custom or
// experimental embedders without registering them.
func WithEmbedder(emb embed.Embedder) Option {
	return func(e *Engine) { e.emb, e.injected = emb, true }
}

// WithoutEmbeddings disables semantic search and indexing for this engine,
// regardless of the stored configuration (e.g. for lexical-only baselines).
func WithoutEmbeddings() Option {
	return func(e *Engine) { e.noEmbed = true }
}

// Open opens the graph database at path.
func Open(path string, opts ...Option) (*Engine, error) {
	st, err := store.Open(path)
	if err != nil {
		return nil, err
	}
	e := &Engine{st: st, path: path}
	for _, o := range opts {
		o(e)
	}
	if err := st.Read(func(tx *store.Tx) error {
		p, err := tx.LoadParams()
		if err != nil {
			return err
		}
		e.P = p
		return e.loadEmbedConfig(tx)
	}); err != nil {
		st.Close()
		return nil, fmt.Errorf("load settings: %w", err)
	}
	return e, nil
}

// Close closes the database and releases the embedder.
func (e *Engine) Close() error {
	e.releaseEmbedder()
	return e.st.Close()
}

// timestamp is the wall-clock time used for IDs and informational
// timestamps. Memory dynamics never depend on it: they run on recall counts.
func timestamp() time.Time { return time.Now().UTC() }

// ---------------------------------------------------------------- nodes

// AddNode validates and stores a new node, assigning ID and timestamps, and
// embeds it if embeddings are enabled.
func (e *Engine) AddNode(n *model.Node) (*model.Node, error) {
	if err := e.AddNodes([]*model.Node{n}); err != nil {
		return nil, err
	}
	return n, nil
}

// AddNodes adds several nodes in one transaction (all or none), assigning
// IDs and timestamps and embedding them if embeddings are enabled.
func (e *Engine) AddNodes(nodes []*model.Node) error {
	if err := prepareNodes(nodes); err != nil {
		return err
	}
	vecs := make([]*nodeVector, len(nodes))
	for i, n := range nodes {
		vecs[i] = e.embedNode(context.Background(), n) // outside the write transaction
	}
	return e.insertNodes(nodes, vecs, nil)
}

// prepareNodes assigns IDs and timestamps and validates.
func prepareNodes(nodes []*model.Node) error {
	now := timestamp()
	for i, n := range nodes {
		n.ID = ulid.MustNew(ulid.Timestamp(now), ulid.DefaultEntropy()).String()
		n.CreatedAt, n.UpdatedAt = now, now
		n.Dormant, n.Fires = false, 0
		if err := n.Validate(); err != nil {
			return fmt.Errorf("node %d (%q): %w", i, n.Title, err)
		}
	}
	return nil
}

// insertNodes stores prepared nodes and their (optional) vectors in one
// transaction, together with then (if not nil).
func (e *Engine) insertNodes(nodes []*model.Node, vecs []*nodeVector, then func(*store.Tx) error) error {
	return e.st.Write(func(tx *store.Tx) error {
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		for i, n := range nodes {
			n.CreatedTick = tick
			if err := tx.InsertNode(n); err != nil {
				return err
			}
			if v := vecs[i]; v != nil {
				if err := tx.PutVector(n.ID, v.model, v.hash, v.blob); err != nil {
					return err
				}
			}
		}
		if then != nil {
			return then(tx)
		}
		return nil
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
	// SupersededBy marks the neuron as replaced by another ("" clears it).
	SupersededBy *string `json:"superseded_by,omitempty"`
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
		if p.SupersededBy != nil {
			if err := supersede(tx, id, *p.SupersededBy); err != nil {
				return err
			}
			n.SupersededBy = *p.SupersededBy
		}
		n.UpdatedAt = timestamp()
		out = n
		if err := tx.UpdateNode(n); err != nil {
			return err
		}
		return fillSuccessorTitles(tx, n)
	})
	if err != nil {
		return nil, err
	}
	return out, e.refreshVector(out)
}

// refreshVector re-embeds a node whose text changed since it was embedded.
func (e *Engine) refreshVector(n *model.Node) error {
	model := e.activeModel()
	if model == "" {
		return nil
	}
	var stored string
	if err := e.st.Read(func(tx *store.Tx) error {
		var err error
		stored, err = tx.VectorHash(n.ID, model)
		return err
	}); err != nil {
		return err
	}
	if stored == textHash(embed.Truncate(nodeText(n), e.P.EmbedMaxWords)) {
		return nil
	}
	vec := e.embedNode(context.Background(), n)
	if vec == nil {
		return nil
	}
	return e.st.Write(func(tx *store.Tx) error { return tx.PutVector(n.ID, vec.model, vec.hash, vec.blob) })
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
		if err := fillSuccessorTitles(tx, n); err != nil {
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
	var ed *model.Edge
	err := e.st.Write(func(tx *store.Tx) error {
		if err := e.linkTx(tx, src, dst, kind, weight, timestamp()); err != nil {
			return err
		}
		var err error
		ed, err = tx.GetEdge(src, dst, kind)
		return err
	})
	return ed, err
}

// linkTx validates and upserts one explicit edge inside a transaction.
func (e *Engine) linkTx(tx *store.Tx, src, dst string, kind model.EdgeKind, weight float64, now time.Time) error {
	if !kind.Valid() {
		return fmt.Errorf("invalid edge kind %q", kind)
	}
	if src == dst {
		return fmt.Errorf("self-loops are not allowed")
	}
	if weight > 1 {
		return fmt.Errorf("weight must be in (0,1], got %g", weight)
	}
	nodes, err := mustExist(tx, src, dst)
	if err != nil {
		return err
	}
	ed := plasticity.NewTyped(src, dst, kind, weight, nodes[src].Fires, nodes[dst].Fires, now, e.P)
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
}

// LinkSpec describes an edge for AddLinks.
type LinkSpec struct {
	Src, Dst string
	Kind     model.EdgeKind
	Weight   float64 // ≤ 0: default for the kind
}

// AddLinks creates or overwrites several edges in one transaction.
func (e *Engine) AddLinks(links []LinkSpec) error {
	now := timestamp()
	return e.st.Write(func(tx *store.Tx) error {
		for _, l := range links {
			if err := e.linkTx(tx, l.Src, l.Dst, l.Kind, l.Weight, now); err != nil {
				return fmt.Errorf("link %s→%s: %w", l.Src, l.Dst, err)
			}
		}
		return nil
	})
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
	old := e.P.ToMap()[key]
	e.P = p
	e.logChange(key, old, p.ToMap()[key])
	return nil
}

// logChange logs a change of a parameter or of the embedder, so that usage
// reports can compare the periods before and after it.
func (e *Engine) logChange(key, old, new string) {
	if old != new {
		e.Log("", EventParam, map[string]string{"key": key, "old": old, "new": new})
	}
}

// ResetParam removes an override, restoring the default.
func (e *Engine) ResetParam(key string) error {
	if _, ok := e.P.ToMap()[key]; !ok {
		return fmt.Errorf("unknown parameter %q", key)
	}
	old := e.P.ToMap()[key]
	if err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.ResetParam(key); err != nil {
			return err
		}
		p, err := tx.LoadParams()
		e.P = p
		return err
	}); err != nil {
		return err
	}
	e.logChange(key, old, e.P.ToMap()[key])
	return nil
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
