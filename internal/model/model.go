// Package model defines neurons (nodes), synapses (edges) and the tunable
// parameters of the plasticity engine.
package model

import (
	"fmt"
	"slices"
	"time"
)

// NodeType classifies the knowledge held by a neuron.
type NodeType string

const (
	TypeConcept   NodeType = "concept"
	TypeFact      NodeType = "fact"
	TypeDecision  NodeType = "decision"
	TypeRationale NodeType = "rationale"
	TypeProcedure NodeType = "procedure"
	TypeEpisode   NodeType = "episode"
	TypeQuestion  NodeType = "question"
)

// NodeTypes lists all valid node types.
var NodeTypes = []NodeType{TypeConcept, TypeFact, TypeDecision, TypeRationale, TypeProcedure, TypeEpisode, TypeQuestion}

// Valid reports whether t is a known node type.
func (t NodeType) Valid() bool { return slices.Contains(NodeTypes, t) }

// EdgeKind classifies a synapse. Associative edges are learned by co-activation
// and are symmetric; all other kinds are explicit, directed and decay slower.
type EdgeKind string

const (
	KindAssociative EdgeKind = "associative"
	KindSupports    EdgeKind = "supports"
	KindDerivesFrom EdgeKind = "derives_from"
	KindContradicts EdgeKind = "contradicts"
	KindPartOf      EdgeKind = "part_of"
	KindExplicit    EdgeKind = "explicit"
)

// EdgeKinds lists all valid edge kinds.
var EdgeKinds = []EdgeKind{KindAssociative, KindSupports, KindDerivesFrom, KindContradicts, KindPartOf, KindExplicit}

// Valid reports whether k is a known edge kind.
func (k EdgeKind) Valid() bool { return slices.Contains(EdgeKinds, k) }

// Typed reports whether the edge is an explicit (non-learned) relation.
func (k EdgeKind) Typed() bool { return k != KindAssociative }

// Node is a neuron: a unit of knowledge together with its rationale.
type Node struct {
	ID         string   `json:"id"`
	Type       NodeType `json:"type"`
	Title      string   `json:"title"`
	Content    string   `json:"content,omitempty"`
	Rationale  string   `json:"rationale,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	Source     string   `json:"source,omitempty"`
	Importance float64  `json:"importance"`
	Dormant    bool     `json:"dormant,omitempty"`
	// Fires counts the recalls in which this neuron was activated. It is the
	// neuron's own logical clock: synapses age by their endpoints' firings.
	Fires int64 `json:"fires"`
	// CreatedTick is the global recall tick at creation (for ACT-R).
	CreatedTick int64     `json:"created_tick"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Validate checks the node's invariants.
func (n *Node) Validate() error {
	if !n.Type.Valid() {
		return fmt.Errorf("invalid node type %q", n.Type)
	}
	if n.Title == "" {
		return fmt.Errorf("node title must not be empty")
	}
	if n.Importance < 0 || n.Importance > 1 {
		return fmt.Errorf("importance must be in [0,1], got %g", n.Importance)
	}
	return nil
}

// Edge is a weighted synapse.
//
// Forgetting is driven by interference, not wall-clock time: the age of a
// synapse is the number of times one of its endpoints fired without the
// other since the synapse was last updated,
//
//	age = (SrcFires − SrcMark) + (DstFires − DstMark),
//
// and Weight is the value materialized at age 0 (see package plasticity).
type Edge struct {
	Src       string   `json:"src"`
	Dst       string   `json:"dst"`
	Kind      EdgeKind `json:"kind"`
	Weight    float64  `json:"weight"`
	Stability float64  `json:"stability"` // in endpoint firings
	// SrcMark and DstMark are the endpoints' fire counts at the last update.
	SrcMark int64 `json:"src_mark"`
	DstMark int64 `json:"dst_mark"`
	// SrcFires and DstFires are the endpoints' current fire counts; they are
	// filled in by the store when reading and never persisted on the edge.
	SrcFires      int64     `json:"-"`
	DstFires      int64     `json:"-"`
	LastUpdate    time.Time `json:"last_update"` // wall clock, informational
	Coactivations int       `json:"coactivations"`
}

// Age is the number of solo firings of the endpoints since the last update.
func (e *Edge) Age() int64 {
	return max(0, e.SrcFires-e.SrcMark) + max(0, e.DstFires-e.DstMark)
}

// Other returns the endpoint of e opposite to id.
func (e *Edge) Other(id string) string {
	if e.Src == id {
		return e.Dst
	}
	return e.Src
}

// CanonicalPair orders two node IDs so that associative (symmetric) edges are
// stored exactly once.
func CanonicalPair(a, b string) (string, string) {
	if a <= b {
		return a, b
	}
	return b, a
}
