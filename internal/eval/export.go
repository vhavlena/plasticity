package eval

import (
	"fmt"
	"strings"
	"time"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/hooks"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// Labeling strategies for exported queries.
const (
	// LabelTurn: a recall's relevant neurons are those reported used after
	// it and before the session's next recalled user prompt (the same turn).
	LabelTurn = "turn"
	// LabelSession: every recall of a session is labeled with all neurons the
	// session reported used.
	LabelSession = "session"
)

// ExportOptions control ExportSessions.
type ExportOptions struct {
	Name  string
	Label string // LabelTurn (default) or LabelSession
	// Learned also exports associative (learned) links, as preloaded links
	// with their current effective weights. By default only typed links are
	// exported, so the replay starts without the learning it is evaluating.
	Learned bool
}

// ExportStats describes what ExportSessions kept and dropped.
type ExportStats struct {
	Sessions  int `json:"sessions"`
	Recalls   int `json:"recalls"`
	Queries   int `json:"queries"`
	Unlabeled int `json:"unlabeled_recalls"` // recalls with no reported use: not exported
	Dropped   int `json:"dropped_labels"`    // labels on deleted neurons or neurons created after the recall
	Episodes  int `json:"dropped_episodes"`  // episode neurons of the exported sessions
}

// ExportSessions turns logged sessions into a scenario: the graph (nodes
// and typed links) becomes the scenario's knowledge base, each logged recall
// a query, and the neurons Claude reported used its relevance labels.
// Nodes are keyed by their IDs. Every node is loaded from the start of the
// replay, including neurons that sessions created later; such neurons are
// never labels of the recalls before their creation. The episode neurons of
// the exported sessions are left out, since they quote the queries.
func ExportSessions(events []store.Event, nodes []*model.Node, edges []engine.ExportEdge, opt ExportOptions) (*Scenario, *ExportStats, error) {
	if opt.Label == "" {
		opt.Label = LabelTurn
	}
	if opt.Label != LabelTurn && opt.Label != LabelSession {
		return nil, nil, fmt.Errorf("unknown label strategy %q (%s|%s)", opt.Label, LabelTurn, LabelSession)
	}
	logged := map[string]bool{}
	for _, ev := range events {
		logged[ev.Session] = true
	}
	st := &ExportStats{}
	exists := make(map[string]bool, len(nodes))
	s := &Scenario{Name: opt.Name}
	for _, n := range nodes {
		// A session's episode quotes its prompts: it would answer the
		// session's own queries.
		if n.Type == model.TypeEpisode && logged[strings.TrimPrefix(n.Source, "session:")] {
			st.Episodes++
			continue
		}
		exists[n.ID] = true
		s.Nodes = append(s.Nodes, ScenarioNode{Key: n.ID, Type: n.Type, Title: n.Title, Content: n.Content,
			Rationale: n.Rationale, Tags: n.Tags, Importance: n.Importance})
	}
	for _, ed := range edges {
		if (ed.Kind.Typed() || opt.Learned) && exists[ed.Src] && exists[ed.Dst] {
			s.Links = append(s.Links, ScenarioLink{Src: ed.Src, Dst: ed.Dst, Kind: ed.Kind, Weight: ed.EffectiveWeight})
		}
	}

	// Group the events by session, keeping their order.
	type sessionLog struct {
		id     string
		events []store.Event
	}
	var order []*sessionLog
	bySession := map[string]*sessionLog{}
	for _, ev := range events {
		if ev.Session == "" {
			continue
		}
		sl := bySession[ev.Session]
		if sl == nil {
			sl = &sessionLog{id: ev.Session}
			bySession[ev.Session] = sl
			order = append(order, sl)
		}
		sl.events = append(sl.events, ev)
	}

	for _, sl := range order {
		createdAt := map[string]int64{} // neuron -> event ID of its creation in this session
		for _, ev := range sl.events {
			if ev.Kind == engine.EventAdd {
				if p, ok := decode[addPayload](ev); ok && p.ID != "" {
					createdAt[p.ID] = ev.ID
				}
			}
		}
		var all []string // every neuron the session reported used, in order
		seen := map[string]bool{}
		for _, ev := range sl.events {
			if ev.Kind != engine.EventUsed {
				continue
			}
			if p, ok := decode[usedPayload](ev); ok {
				for _, id := range p.IDs {
					if !seen[id] {
						seen[id] = true
						all = append(all, id)
					}
				}
			}
		}

		sess := Session{Name: sl.id}
		for i, ev := range sl.events {
			if ev.Kind != engine.EventRecall {
				continue
			}
			p, ok := decode[recallPayload](ev)
			if !ok || p.query() == "" {
				continue
			}
			st.Recalls++
			var cand []string
			if opt.Label == LabelSession {
				cand = all
			} else {
				turn := map[string]bool{}
				for _, next := range sl.events[i+1:] {
					if next.Kind == engine.EventRecall {
						// The next recalled user prompt starts a new turn; short
						// prompts ("ok, go ahead") are not recalled and continue it.
						if np, ok := decode[recallPayload](next); ok && np.source() == engine.RecallSourcePrompt {
							break
						}
					}
					if next.Kind != engine.EventUsed {
						continue
					}
					if u, ok := decode[usedPayload](next); ok {
						for _, id := range u.IDs {
							if !turn[id] {
								turn[id] = true
								cand = append(cand, id)
							}
						}
					}
				}
			}
			var rel []string
			for _, id := range cand {
				if at, ok := createdAt[id]; !exists[id] || (ok && at > ev.ID) {
					st.Dropped++
					continue
				}
				rel = append(rel, id)
			}
			if len(rel) == 0 {
				st.Unlabeled++
				continue
			}
			q := Query{Q: p.query(), Relevant: rel}
			switch p.Source {
			case "", engine.RecallSourcePrompt:
				if hooks.IsNotification(q.Q) {
					q.Source = SourceNotification // logged raw: replay through the hook's plan
				}
			case engine.RecallSourceNotification:
				q.Source, q.Planned = SourceNotification, true // logged after planning
			}
			sess.Queries = append(sess.Queries, q)
		}
		if len(sess.Queries) > 0 {
			s.Sessions = append(s.Sessions, sess)
			st.Sessions++
			st.Queries += len(sess.Queries)
		}
	}
	if st.Queries == 0 {
		return nil, st, fmt.Errorf("no labeled recalls: of %d logged recalls none was followed by `plasticity used`", st.Recalls)
	}
	s.Description = fmt.Sprintf("exported %s from %d real sessions (%d labeled of %d recalls, labels: %s)",
		time.Now().Format(time.DateOnly), st.Sessions, st.Queries, st.Recalls, opt.Label)
	return s, st, s.Validate()
}
