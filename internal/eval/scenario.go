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
	// Project is the project the neuron was created in; hits from another
	// project than the query's count as cross-project noise.
	Project string `json:"project,omitempty"`
	// UsedIn are the projects the neuron had been used in before the
	// scenario (for project affinity: a neuron used in several projects is
	// shared knowledge).
	UsedIn []string `json:"used_in,omitempty"`
	// SupersededBy is the key of the neuron that replaces this one; it is
	// marked as soon as both exist (at load, or when one is added).
	SupersededBy string `json:"superseded_by,omitempty"`
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
	Project string  `json:"project,omitempty"` // project the session works in
	Queries []Query `json:"queries"`
}

// Query sources: who issued the recall. Queries without a source are
// recalled as they are.
const (
	// SourcePrompt is a user prompt, recalled by the prompt hook (which skips
	// slash commands and acknowledgements).
	SourcePrompt = "prompt"
	// SourceNotification is a machine-generated message delivered as a
	// prompt (e.g. a <task-notification> of a finished background task).
	SourceNotification = "notification"
	// SourceClaude is a recall Claude issued itself (no hook filter).
	SourceClaude = "claude"
)

// Query is a recall with the keys of the nodes that are relevant to it.
// An empty Relevant set means nothing should be recalled: every hit is noise.
type Query struct {
	Q        string   `json:"q"`
	Source   string   `json:"source,omitempty"`
	Relevant []string `json:"relevant"`
	// Stale nodes are superseded or wrong for this query: recalling them,
	// worse still above the current answer, misleads.
	Stale []string `json:"stale,omitempty"`
	// Used is what the simulated agent reports as used after the recall (in
	// the session and feedback learning modes), which may be fewer nodes
	// than Relevant, as in real use. Empty: the session mode reports
	// nothing; the feedback mode reports the recalled relevant nodes.
	Used []string `json:"used,omitempty"`
	// OutputTail is the end of the task's output file for a notification
	// (what the prompt hook reads for a failed task).
	OutputTail string `json:"output_tail,omitempty"`
	// Add lists nodes the agent stores after this query (and its used
	// report) within the session; they are absent from the graph until then
	// and get the engine's add-time context links.
	Add []string `json:"add,omitempty"`
}

// deferred returns the keys of nodes added during the sessions.
func (s *Scenario) deferred() map[string]bool {
	out := map[string]bool{}
	for _, sess := range s.Sessions {
		for _, q := range sess.Queries {
			for _, k := range q.Add {
				out[k] = true
			}
		}
	}
	return out
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
	for _, n := range s.Nodes {
		if n.SupersededBy != "" && (!keys[n.SupersededBy] || n.SupersededBy == n.Key) {
			return fmt.Errorf("node %q: superseded_by names an unknown node or itself: %q", n.Key, n.SupersededBy)
		}
	}
	deferred := s.deferred()
	for i, l := range s.Links {
		if !keys[l.Src] || !keys[l.Dst] {
			return fmt.Errorf("link %d: unknown node %q or %q", i, l.Src, l.Dst)
		}
		if deferred[l.Src] || deferred[l.Dst] {
			return fmt.Errorf("link %d: node %q or %q is added during a session", i, l.Src, l.Dst)
		}
		if !l.Kind.Valid() {
			return fmt.Errorf("link %d: invalid kind %q", i, l.Kind)
		}
	}
	queries := 0
	added := map[string]bool{}
	for si, sess := range s.Sessions {
		for qi, q := range sess.Queries {
			if q.Q == "" {
				return fmt.Errorf("session %d query %d: needs text", si, qi)
			}
			switch q.Source {
			case "", SourcePrompt, SourceNotification, SourceClaude:
			default:
				return fmt.Errorf("session %d query %d: unknown source %q", si, qi, q.Source)
			}
			for field, ks := range map[string][]string{"relevant": q.Relevant, "stale": q.Stale, "used": q.Used, "add": q.Add} {
				for _, k := range ks {
					if !keys[k] {
						return fmt.Errorf("session %d query %d: unknown %s node %q", si, qi, field, k)
					}
				}
			}
			for _, k := range q.Add {
				if added[k] {
					return fmt.Errorf("session %d query %d: node %q added twice", si, qi, k)
				}
				added[k] = true
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
