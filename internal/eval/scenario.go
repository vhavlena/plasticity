// Package eval replays scenarios through the engine to measure retrieval
// quality, learning, graph health and performance, so that algorithm and
// parameter changes can be compared offline and reproducibly.
package eval

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/vhavlena/plasticity/internal/model"
)

// Scenario is a knowledge base plus a sequence of sessions with ground truth.
type Scenario struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Nodes       []ScenarioNode `json:"nodes"`
	Links       []ScenarioLink `json:"links,omitempty"`
	Sessions    []Session      `json:"sessions"`
}

// ScenarioNode is a neuron; Key is the scenario-local name used by links and
// relevance labels.
type ScenarioNode struct {
	Key        string         `json:"key"`
	Type       model.NodeType `json:"type"`
	Title      string         `json:"title"`
	Content    string         `json:"content,omitempty"`
	Rationale  string         `json:"rationale,omitempty"`
	Tags       []string       `json:"tags,omitempty"`
	Importance float64        `json:"importance,omitempty"`
}

// ScenarioLink is an explicit (authored) edge.
type ScenarioLink struct {
	Src    string         `json:"src"`
	Dst    string         `json:"dst"`
	Kind   model.EdgeKind `json:"kind"`
	Weight float64        `json:"weight,omitempty"`
}

// Session is a group of queries, as in one Claude session.
type Session struct {
	Name    string  `json:"name,omitempty"`
	Queries []Query `json:"queries"`
}

// Query is a recall with the keys of the nodes that are relevant to it.
type Query struct {
	Q        string   `json:"q"`
	Relevant []string `json:"relevant"`
}

// Load reads and validates a scenario file.
func Load(path string) (*Scenario, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s Scenario
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if err := s.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &s, nil
}

// Save writes the scenario as JSON.
func (s *Scenario) Save(path string) error {
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// Validate checks keys, references and node fields.
func (s *Scenario) Validate() error {
	keys := make(map[string]bool, len(s.Nodes))
	for i, n := range s.Nodes {
		if n.Key == "" || keys[n.Key] {
			return fmt.Errorf("node %d: missing or duplicate key %q", i, n.Key)
		}
		keys[n.Key] = true
		mn := model.Node{Type: n.Type, Title: n.Title, Importance: n.Importance}
		if err := mn.Validate(); err != nil {
			return fmt.Errorf("node %q: %w", n.Key, err)
		}
	}
	for i, l := range s.Links {
		if !keys[l.Src] || !keys[l.Dst] {
			return fmt.Errorf("link %d: unknown node %q or %q", i, l.Src, l.Dst)
		}
		if !l.Kind.Valid() {
			return fmt.Errorf("link %d: invalid kind %q", i, l.Kind)
		}
	}
	queries := 0
	for si, sess := range s.Sessions {
		for qi, q := range sess.Queries {
			if q.Q == "" || len(q.Relevant) == 0 {
				return fmt.Errorf("session %d query %d: needs text and at least one relevant node", si, qi)
			}
			for _, k := range q.Relevant {
				if !keys[k] {
					return fmt.Errorf("session %d query %d: unknown relevant node %q", si, qi, k)
				}
			}
			queries++
		}
	}
	if queries == 0 {
		return fmt.Errorf("scenario has no queries")
	}
	return nil
}

// QueryCount returns the number of queries in one pass.
func (s *Scenario) QueryCount() int {
	n := 0
	for _, sess := range s.Sessions {
		n += len(sess.Queries)
	}
	return n
}
