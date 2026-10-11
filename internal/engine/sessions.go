package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// Session node roles.
const (
	RoleUsed    = "used"
	RoleCreated = "created"
)

// Event kinds of the log.
const (
	EventSessionStart = "session_start"
	EventPrompt       = "prompt"
	EventRecall       = "recall"
	EventUsed         = "used"
	EventAdd          = "add"
	EventNudge        = "nudge"
	EventEpisode      = "episode"
	EventConsolidate  = "consolidate"
	EventError        = "error"
)

// Recall sources, logged with each recall event.
const (
	RecallSourcePrompt = "prompt" // automatic recall for a user prompt (hook)
	RecallSourceClaude = "claude" // recall Claude ran itself during a session
)

// LogRecall logs a recall of a session: its source, query, hits and
// confidence, latency and the characters injected into the context.
func (e *Engine) LogRecall(session, source, query string, r *RecallResult, latency time.Duration, injectedChars int) {
	hits := make([]map[string]any, len(r.Hits))
	for i, h := range r.Hits {
		hits[i] = map[string]any{"id": h.Node.ID, "score": h.Score, "activation": h.Activation, "seed": h.IsSeed, "hops": h.Hops}
	}
	e.Log(session, EventRecall, map[string]any{
		"source": source, "query": truncate(query, 500), "hits": hits, "confidence": r.Confidence,
		"lexical": r.LexicalConfidence, "semantic": r.SemanticConfidence, "seeds": r.Seeds,
		"latency_ms": float64(latency) / float64(time.Millisecond), "injected_chars": injectedChars,
		"recall_id": r.RecallID, "trace_skipped": r.TraceSkipped,
	})
}

// WithoutLog disables the event log for this engine.
func WithoutLog() Option {
	return func(e *Engine) { e.noLog = true }
}

// Log appends an event (payload is marshaled to JSON). Logging never fails
// the caller: problems become warnings.
func (e *Engine) Log(session, kind string, payload any) {
	if e.noLog {
		return
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		e.warn("log %s: %v", kind, err)
		return
	}
	if err := e.st.Write(func(tx *store.Tx) error { return tx.LogEvent(timestamp(), session, kind, raw) }); err != nil {
		e.warn("log %s: %v", kind, err)
	}
}

// Events returns logged events.
func (e *Engine) Events(f store.EventFilter) ([]store.Event, error) {
	var evs []store.Event
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		evs, err = tx.Events(f)
		return err
	})
	if evs == nil {
		evs = []store.Event{}
	}
	return evs, err
}

// Session returns a session (nil if unknown).
func (e *Engine) Session(id string) (*store.Session, error) {
	var s *store.Session
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		s, err = tx.GetSession(id)
		return err
	})
	return s, err
}

// StartSession registers a session (idempotent) with its transcript path
// and logs it.
func (e *Engine) StartSession(id, project, source string, transcript ...string) error {
	if err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.TouchSession(id, project, timestamp()); err != nil {
			return err
		}
		if len(transcript) > 0 {
			return tx.SetTranscript(id, transcript[0])
		}
		return nil
	}); err != nil {
		return err
	}
	e.Log(id, EventSessionStart, map[string]string{"project": project, "source": source})
	return nil
}

// NotePrompt records a user prompt of a session.
func (e *Engine) NotePrompt(id, project, prompt string) error {
	err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.TouchSession(id, project, timestamp()); err != nil {
			return err
		}
		return tx.CountPrompt(id, truncate(prompt, 200))
	})
	if err != nil {
		return err
	}
	e.Log(id, EventPrompt, map[string]string{"prompt": truncate(prompt, 2000)})
	return nil
}

// Used records that Claude used these neurons together. Within a session it
// is feedback (LTP among the used neurons, LTD towards neurons recalled in
// the session but not used); without a session it is plain reinforcement.
func (e *Engine) Used(session string, ids []string) (*LearnReport, error) {
	var rep *LearnReport
	var err error
	switch {
	case session != "":
		rep, err = e.Feedback(session, ids)
	case len(ids) >= 2:
		rep, err = e.Reinforce(ids)
	default:
		return nil, fmt.Errorf("used without a session needs at least two nodes")
	}
	if err != nil {
		return nil, err
	}
	if session != "" {
		if err := e.st.Write(func(tx *store.Tx) error {
			if err := tx.TouchSession(session, "", timestamp()); err != nil {
				return err
			}
			return tx.AddSessionNodes(session, RoleUsed, ids, timestamp())
		}); err != nil {
			return rep, err
		}
	}
	e.Log(session, EventUsed, map[string]any{"ids": ids, "report": rep})
	return rep, nil
}

// AddOptions control AddNodeChecked.
type AddOptions struct {
	// Session links the new neuron to the neurons the session used or
	// created before it, and records it as created by the session.
	Session string
	// Force inserts even if a near-duplicate exists.
	Force bool
}

// AddResult is the outcome of AddNodeChecked.
type AddResult struct {
	Node *model.Node `json:"node"`
	// Duplicate is set (and nothing was inserted) when an existing neuron is
	// a near-duplicate; update or link it instead, or add with Force.
	Duplicate    *model.Node `json:"duplicate_of,omitempty"`
	Similarity   float64     `json:"similarity,omitempty"`
	Reason       string      `json:"reason,omitempty"`
	ContextLinks int         `json:"context_links"`
}

// AddNodeChecked adds a neuron after a duplicate check (same normalized
// title, or embedding cosine ≥ DedupeCos), and wires it into its session's
// context with associative links of weight ContextLinkWeight.
func (e *Engine) AddNodeChecked(n *model.Node, opt AddOptions) (*AddResult, error) {
	if err := prepareNodes([]*model.Node{n}); err != nil {
		return nil, err
	}
	ctx := context.Background()
	vec := e.embedNode(ctx, n)
	res := &AddResult{Node: n}
	if !opt.Force {
		dup, sim, reason, err := e.findDuplicate(n, vec)
		if err != nil {
			return nil, err
		}
		if dup != nil {
			res.Node, res.Duplicate, res.Similarity, res.Reason = nil, dup, sim, reason
			e.Log(opt.Session, EventAdd, map[string]any{"title": n.Title, "duplicate_of": dup.ID, "reason": reason, "similarity": sim})
			return res, nil
		}
	}
	if err := e.insertNodes([]*model.Node{n}, []*nodeVector{vec}); err != nil {
		return nil, err
	}
	if opt.Session != "" {
		n, err := e.linkToSession(opt.Session, n.ID, RoleCreated)
		if err != nil {
			return res, err
		}
		res.ContextLinks = n
	}
	e.Log(opt.Session, EventAdd, map[string]any{"id": n.ID, "type": n.Type, "title": n.Title, "context_links": res.ContextLinks})
	return res, nil
}

// findDuplicate looks for an existing neuron equivalent to n.
func (e *Engine) findDuplicate(n *model.Node, vec *nodeVector) (*model.Node, float64, string, error) {
	var dup *model.Node
	var sim float64
	var reason string
	err := e.st.Read(func(tx *store.Tx) error {
		hits, err := tx.Search(MatchExpr(n.Title), e.bm25Weights(), 5)
		if err != nil {
			return err
		}
		for _, h := range hits {
			if normTitle(h.Node.Title) == normTitle(n.Title) {
				dup, sim, reason = h.Node, 1, "same title"
				return nil
			}
		}
		if vec == nil || e.emb == nil {
			return nil
		}
		v, err := embed.Decode(vec.blob)
		if err != nil {
			return err
		}
		best, err := e.semanticSearch(tx, v, 1)
		if err != nil || len(best) == 0 || best[0].cos < e.P.DedupeCos {
			return err
		}
		if dup, err = tx.GetNode(best[0].id); err != nil {
			return err
		}
		sim, reason = best[0].cos, "similar embedding"
		return nil
	})
	return dup, sim, reason, err
}

func normTitle(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// linkToSession links node to the session's most recent used/created
// neurons, records it under role, and returns the number of links made.
func (e *Engine) linkToSession(session, node, role string) (int, error) {
	var ctxIDs []string
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		ctxIDs, err = tx.SessionNodes(session, RoleUsed, RoleCreated)
		return err
	})
	if err != nil {
		return 0, err
	}
	var links []LinkSpec
	for i := len(ctxIDs) - 1; i >= 0 && len(links) < e.P.ContextLinks; i-- {
		if ctxIDs[i] != node {
			links = append(links, LinkSpec{Src: node, Dst: ctxIDs[i], Kind: model.KindAssociative, Weight: e.P.ContextLinkWeight})
		}
	}
	if err := e.AddLinks(links); err != nil {
		return 0, err
	}
	err = e.st.Write(func(tx *store.Tx) error {
		if err := tx.TouchSession(session, "", timestamp()); err != nil {
			return err
		}
		return tx.AddSessionNodes(session, role, []string{node}, timestamp())
	})
	return len(links), err
}

// ShouldNudge decides whether the Stop hook should ask Claude to store or
// report memories: at most once per session, only after toolCalls ≥
// NudgeMinToolCalls, and only if the session neither stored nor reported
// anything. A positive decision is recorded.
func (e *Engine) ShouldNudge(session string, toolCalls int) (bool, error) {
	nudge := false
	err := e.st.Write(func(tx *store.Tx) error {
		s, err := tx.GetSession(session)
		if err != nil || s == nil || s.Nudged || toolCalls < e.P.NudgeMinToolCalls {
			return err
		}
		ids, err := tx.SessionNodes(session, RoleUsed, RoleCreated)
		if err != nil || len(ids) > 0 {
			return err
		}
		nudge = true
		return tx.SetNudged(session)
	})
	if nudge {
		e.Log(session, EventNudge, map[string]int{"tool_calls": toolCalls})
	}
	return nudge, err
}

// AbandonedSessions returns up to limit sessions that never ended and were
// last seen more than idle ago (e.g. Claude Code exited before SessionEnd).
func (e *Engine) AbandonedSessions(idle time.Duration, limit int) ([]*store.Session, error) {
	var out []*store.Session
	err := e.st.Read(func(tx *store.Tx) error {
		ids, err := tx.OpenSessionsBefore(timestamp().Add(-idle), limit)
		if err != nil {
			return err
		}
		for _, id := range ids {
			s, err := tx.GetSession(id)
			if err != nil {
				return err
			}
			out = append(out, s)
		}
		return nil
	})
	return out, err
}

// EpisodeReport summarizes EndSession.
type EpisodeReport struct {
	Episode      *model.Node        `json:"episode,omitempty"`
	Links        int                `json:"links"`
	Consolidated *ConsolidateReport `json:"consolidated,omitempty"`
}

// EndSession closes a session: it creates the session's episode neuron
// (first prompt as title, prompts and edited files as content), links it to
// the neurons the session used or created, and consolidates every
// ConsolidateEverySessions sessions. Sessions without prompts get no episode.
func (e *Engine) EndSession(session string, files []string) (*EpisodeReport, error) {
	s, err := e.Session(session)
	if err != nil || s == nil {
		return nil, err
	}
	rep := &EpisodeReport{}
	if s.EpisodeID == "" && s.Prompts > 0 {
		evs, err := e.Events(store.EventFilter{Session: session, Kind: EventPrompt, Limit: 20})
		if err != nil {
			return nil, err
		}
		var prompts []string
		for _, ev := range evs {
			var p struct{ Prompt string }
			if json.Unmarshal(ev.Payload, &p) == nil && p.Prompt != "" {
				prompts = append(prompts, "- "+truncate(p.Prompt, 160))
			}
		}
		content := fmt.Sprintf("Session with %d prompts.\nPrompts:\n%s", s.Prompts, strings.Join(prompts, "\n"))
		if len(files) > 0 {
			content += "\nFiles edited: " + strings.Join(files, ", ")
		}
		ep := &model.Node{Type: model.TypeEpisode, Title: "Session: " + truncate(s.FirstPrompt, 100),
			Content: content, Source: "session:" + session}
		if s.Project != "" {
			ep.Tags = []string{"project:" + s.Project}
		}
		if err := e.AddNodes([]*model.Node{ep}); err != nil {
			return nil, err
		}
		if rep.Links, err = e.linkToSession(session, ep.ID, "episode"); err != nil {
			return nil, err
		}
		rep.Episode = ep
	}
	episodeID := s.EpisodeID
	if rep.Episode != nil {
		episodeID = rep.Episode.ID
	}
	var ended int
	if err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.EndSession(session, episodeID, timestamp()); err != nil {
			return err
		}
		ended, err = tx.CountEndedSessions()
		return err
	}); err != nil {
		return rep, err
	}
	if rep.Episode != nil {
		e.Log(session, EventEpisode, map[string]any{"id": rep.Episode.ID, "links": rep.Links})
	}
	if ended%e.P.ConsolidateEverySessions == 0 {
		c, err := e.Consolidate(false)
		if err != nil {
			return rep, err
		}
		rep.Consolidated = c
		e.Log(session, EventConsolidate, c)
	}
	return rep, nil
}

// truncate cuts s to at most n runes, adding an ellipsis.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n-1]) + "…"
}
