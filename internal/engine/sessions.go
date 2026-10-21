package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
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
	// EventNotification is a task notification delivered as a prompt; it is
	// not counted as a prompt of the session.
	EventNotification = "notification"
	EventRecall       = "recall"
	EventUsed         = "used"
	EventAdd          = "add"
	EventNudge        = "nudge"
	EventEpisode      = "episode"
	EventConsolidate  = "consolidate"
	EventError        = "error"
	EventParam        = "param"    // a parameter or the embedder changed
	EventNote         = "note"     // the user's observation (plasticity note)
	EventMaintain     = "maintain" // a one-off repair of the graph (plasticity maintain)
)

// Note kinds.
var NoteKinds = []string{"miss", "noise", "helped", "other"}

// Note logs the user's observation about the memory: a miss (something
// stored was not recalled), noise (an irrelevant injection), a memory that
// helped, or anything else. ids optionally name the neurons concerned.
func (e *Engine) Note(session, kind, text string, ids []string) error {
	if !slices.Contains(NoteKinds, kind) {
		return fmt.Errorf("unknown note kind %q (%s)", kind, strings.Join(NoteKinds, "|"))
	}
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("the note needs a text")
	}
	if e.noLog {
		return fmt.Errorf("the event log is disabled")
	}
	payload := map[string]any{"kind": kind, "text": truncate(text, 2000)}
	if len(ids) > 0 {
		payload["ids"] = ids
	}
	raw, _ := json.Marshal(payload)
	return e.st.Write(func(tx *store.Tx) error { return tx.LogEvent(timestamp(), session, EventNote, raw) })
}

// Recall sources, logged with each recall event.
const (
	RecallSourcePrompt = "prompt" // automatic recall for a user prompt (hook)
	RecallSourceClaude = "claude" // recall Claude ran itself during a session
	// RecallSourceNotification: automatic recall for a failed background
	// task's notification (hook).
	RecallSourceNotification = "notification"
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
		"recall_id": r.RecallID, "trace_skipped": r.TraceSkipped, "suppressed": r.Suppressed,
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

// NodeRef names a neuron in listings.
type NodeRef struct {
	ID    string         `json:"id"`
	Type  model.NodeType `json:"type"`
	Title string         `json:"title"`
}

// SessionSummary is a past session with the neurons it used and created.
type SessionSummary struct {
	*store.Session
	Used    []NodeRef `json:"used"`
	Created []NodeRef `json:"created"`
}

// Sessions lists up to limit sessions, most recent first (optionally only
// one project's), with the neurons each used and created: the history of
// work on the graph, kept outside the graph itself.
func (e *Engine) Sessions(project string, limit int) ([]SessionSummary, error) {
	out := []SessionSummary{}
	err := e.st.Read(func(tx *store.Tx) error {
		ss, err := tx.ListSessions(project, limit)
		if err != nil {
			return err
		}
		refs := func(session, role string) ([]NodeRef, error) {
			ids, err := tx.SessionNodes(session, role)
			if err != nil {
				return nil, err
			}
			nodes, err := tx.GetNodes(ids)
			if err != nil {
				return nil, err
			}
			r := []NodeRef{}
			for _, id := range ids {
				if n := nodes[id]; n != nil {
					r = append(r, NodeRef{ID: n.ID, Type: n.Type, Title: n.Title})
				}
			}
			return r, nil
		}
		for _, s := range ss {
			sum := SessionSummary{Session: s}
			if sum.Used, err = refs(s.ID, RoleUsed); err != nil {
				return err
			}
			if sum.Created, err = refs(s.ID, RoleCreated); err != nil {
				return err
			}
			out = append(out, sum)
		}
		return nil
	})
	return out, err
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

// NoteNotification records a task notification delivered to a session as a
// prompt. Unlike NotePrompt it neither counts as a prompt nor becomes the
// session's first prompt.
func (e *Engine) NoteNotification(id, project, text string) error {
	if err := e.st.Write(func(tx *store.Tx) error { return tx.TouchSession(id, project, timestamp()) }); err != nil {
		return err
	}
	e.Log(id, EventNotification, map[string]string{"text": truncate(text, 2000)})
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
	// Supersedes are neurons the new one replaces; they are marked in the
	// same transaction as the insert.
	Supersedes []string
}

// SimilarNeuron is an existing neuron close to a new one, listed so that the
// agent can mark it superseded or update it.
type SimilarNeuron struct {
	ID         string  `json:"id"`
	Title      string  `json:"title"`
	Similarity float64 `json:"similarity"`
}

// similarCos is the lowest embedding cosine at which an existing neuron is
// listed as similar to a new one (up to DedupeCos, where it is a duplicate).
const similarCos = 0.6

// AddResult is the outcome of AddNodeChecked.
type AddResult struct {
	Node *model.Node `json:"node"`
	// Duplicate is set (and nothing was inserted) when an existing neuron is
	// a near-duplicate; update or link it instead, or add with Force.
	Duplicate    *model.Node `json:"duplicate_of,omitempty"`
	Similarity   float64     `json:"similarity,omitempty"`
	Reason       string      `json:"reason,omitempty"`
	ContextLinks int         `json:"context_links"`
	// Similar lists up to 3 existing neurons with cosine in [0.6, dedupe_cos)
	// to the new one; Hint says what to do if one of them is outdated.
	Similar []SimilarNeuron `json:"similar,omitempty"`
	Hint    string          `json:"hint,omitempty"`
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
	similar, err := e.similarNeurons(vec, 3)
	if err != nil {
		return nil, err
	}
	err = e.insertNodes([]*model.Node{n}, []*nodeVector{vec}, func(tx *store.Tx) error {
		for _, old := range opt.Supersedes {
			if err := supersede(tx, old, n.ID); err != nil {
				return fmt.Errorf("--supersedes %s: %w", old, err)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	for _, s := range similar {
		if !slices.Contains(opt.Supersedes, s.ID) {
			res.Similar = append(res.Similar, s)
		}
	}
	if len(res.Similar) > 0 {
		res.Hint = fmt.Sprintf("If a similar neuron is now wrong or replaced, mark it: update ID --superseded-by %s. If it is still partly true, update its content instead.", n.ID)
	}
	if opt.Session != "" {
		n, err := e.linkToSession(opt.Session, n.ID, RoleCreated, true)
		if err != nil {
			return res, err
		}
		res.ContextLinks = n
	}
	e.Log(opt.Session, EventAdd, map[string]any{"id": n.ID, "type": n.Type, "title": n.Title, "context_links": res.ContextLinks,
		"supersedes": opt.Supersedes, "similar": len(res.Similar)})
	return res, nil
}

// findDuplicate looks for an existing neuron equivalent to n.
// similarNeurons returns up to k existing neurons with embedding cosine in
// [similarCos, DedupeCos) to vec, most similar first.
func (e *Engine) similarNeurons(vec *nodeVector, k int) ([]SimilarNeuron, error) {
	if vec == nil || e.emb == nil {
		return nil, nil
	}
	v, err := embed.Decode(vec.blob)
	if err != nil {
		return nil, err
	}
	var out []SimilarNeuron
	err = e.st.Read(func(tx *store.Tx) error {
		hits, err := e.semanticSearch(tx, v, k+1, nil)
		if err != nil {
			return err
		}
		var ids []string
		for _, h := range hits {
			if h.cos >= similarCos && h.cos < e.P.DedupeCos {
				ids = append(ids, h.id)
			}
		}
		nodes, err := tx.GetNodes(ids)
		if err != nil {
			return err
		}
		for _, h := range hits {
			if n := nodes[h.id]; n != nil && len(out) < k {
				out = append(out, SimilarNeuron{ID: n.ID, Title: n.Title, Similarity: h.cos})
			}
		}
		return nil
	})
	return out, err
}

func (e *Engine) findDuplicate(n *model.Node, vec *nodeVector) (*model.Node, float64, string, error) {
	var dup *model.Node
	var sim float64
	var reason string
	err := e.st.Read(func(tx *store.Tx) error {
		same, err := tx.FindTitle(titlePattern(n.Title), func(title string) bool {
			return normTitle(title) == normTitle(n.Title)
		})
		if err != nil {
			return err
		}
		if same != nil {
			dup, sim, reason = same, 1, "same title"
			return nil
		}
		if vec == nil || e.emb == nil {
			return nil
		}
		v, err := embed.Decode(vec.blob)
		if err != nil {
			return err
		}
		best, err := e.semanticSearch(tx, v, 1, nil)
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

// titlePattern is a LIKE pattern matched by every title equal to title up
// to case and whitespace: its words in order, separated by %. SQLite's LIKE
// ignores case only for ASCII, so other characters match any character (_).
func titlePattern(title string) string {
	var b strings.Builder
	b.WriteByte('%')
	for _, w := range strings.Fields(strings.ToLower(title)) {
		for _, r := range w {
			switch {
			case r == '%' || r == '_' || r == '\\':
				b.WriteByte('\\')
				b.WriteRune(r)
			case r > 127:
				b.WriteByte('_')
			default:
				b.WriteRune(r)
			}
		}
		b.WriteByte('%')
	}
	return b.String()
}

// NoteNodeProject records that a neuron was created in project and/or used
// there uses more times (sessions normally record this themselves; this is
// for imports and evaluation).
func (e *Engine) NoteNodeProject(id, project string, created bool, uses int) error {
	return e.st.Write(func(tx *store.Tx) error { return tx.AddNodeProject(id, project, created, uses) })
}

// linkToSession links node to the session's most recent used/created
// neurons, records it under role, and returns the number of links made.
// With window, only neurons of the last ContextWindow prompts are linked (an
// episode summarizes the whole session, a new neuron belongs to its task).
// Existing links are kept as they are (their learned weights included).
func (e *Engine) linkToSession(session, node, role string, window bool) (int, error) {
	var links []LinkSpec
	err := e.st.Read(func(tx *store.Tx) error {
		since := 0
		if window && e.P.ContextWindow > 0 {
			s, err := tx.GetSession(session)
			if err != nil {
				return err
			}
			if s != nil {
				since = s.Prompts - e.P.ContextWindow
			}
		}
		ctxIDs, err := tx.SessionNodesSince(session, since, RoleUsed, RoleCreated)
		if err != nil {
			return err
		}
		for i := len(ctxIDs) - 1; i >= 0 && len(links) < e.P.ContextLinks; i-- {
			if ctxIDs[i] == node {
				continue
			}
			a, b := model.CanonicalPair(node, ctxIDs[i])
			old, err := tx.GetEdge(a, b, model.KindAssociative)
			if err != nil {
				return err
			}
			if old == nil {
				links = append(links, LinkSpec{Src: node, Dst: ctxIDs[i], Kind: model.KindAssociative, Weight: e.P.ContextLinkWeight})
			}
		}
		return nil
	})
	if err != nil {
		return 0, err
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
	Updated      bool               `json:"updated,omitempty"` // an existing episode was refreshed
	Links        int                `json:"links"`
	Consolidated *ConsolidateReport `json:"consolidated,omitempty"`
}

// errSessionClosing aborts creating an episode for a session that another
// process has given one in the meantime.
var errSessionClosing = errors.New("session is being closed by another process")

// EndSession closes a session and consolidates every ConsolidateEverySessions
// sessions. With episode set, it also creates the session's episode neuron
// (first prompt as title, prompts and edited files as content) and links it
// to the neurons the session used or created. Sessions without prompts get no
// episode. A session that was resumed after being closed gets its episode
// refreshed with the later prompts, files and neurons.
func (e *Engine) EndSession(session string, files []string, episode bool) (*EpisodeReport, error) {
	s, err := e.Session(session)
	if err != nil || s == nil {
		return nil, err
	}
	rep := &EpisodeReport{}
	if !s.EndedAt.IsZero() {
		return rep, nil // already closed and not resumed since
	}
	var ep *model.Node
	if s.EpisodeID != "" {
		if ep, err = e.node(s.EpisodeID); errors.Is(err, store.ErrNotFound) {
			ep = nil // deleted by the user: create a new one
		} else if err != nil {
			return nil, err
		}
	}
	if episode && s.Prompts > 0 {
		content, err := e.episodeContent(s, files)
		if err != nil {
			return nil, err
		}
		if ep == nil {
			ep = &model.Node{Type: model.TypeEpisode, Title: "Session: " + truncate(s.FirstPrompt, 100),
				Content: content, Source: "session:" + session}
			if s.Project != "" {
				ep.Tags = []string{"project:" + s.Project}
			}
			// Record the episode on the session in the same transaction:
			// if the hook is killed before the session is closed, the next
			// EndSession refreshes this episode instead of adding another.
			if err := prepareNodes([]*model.Node{ep}); err != nil {
				return nil, err
			}
			vec := e.embedNode(context.Background(), ep)
			err := e.insertNodes([]*model.Node{ep}, []*nodeVector{vec}, func(tx *store.Tx) error {
				cur, err := tx.GetSession(session)
				if err != nil {
					return err
				}
				if cur.EpisodeID != "" || !cur.EndedAt.IsZero() {
					return errSessionClosing
				}
				return tx.SetSessionEpisode(session, ep.ID)
			})
			if errors.Is(err, errSessionClosing) {
				return rep, nil // another process is closing it (concurrent session starts)
			}
			if err != nil {
				return nil, err
			}
			if e.afterEpisode != nil {
				if err := e.afterEpisode(); err != nil {
					return nil, err
				}
			}
		} else {
			rep.Updated = true
			if ep.Content != content {
				if ep, err = e.UpdateNode(ep.ID, NodePatch{Content: &content}); err != nil {
					return nil, err
				}
			}
		}
		if rep.Links, err = e.linkToSession(session, ep.ID, "episode", false); err != nil {
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
		e.Log(session, EventEpisode, map[string]any{"id": rep.Episode.ID, "links": rep.Links, "updated": rep.Updated})
	}
	if ended%e.P.ConsolidateEverySessions == 0 {
		c, err := e.Consolidate()
		if err != nil {
			return rep, err
		}
		rep.Consolidated = c
		e.Log(session, EventConsolidate, c)
	}
	return rep, nil
}

// node returns a stored node.
func (e *Engine) node(id string) (*model.Node, error) {
	var n *model.Node
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		n, err = tx.GetNode(id)
		return err
	})
	return n, err
}

// episodeContent renders a session's prompts and edited files.
func (e *Engine) episodeContent(s *store.Session, files []string) (string, error) {
	evs, err := e.Events(store.EventFilter{Session: s.ID, Kind: EventPrompt})
	if err != nil {
		return "", err
	}
	var prompts []string
	for _, ev := range evs {
		var p struct{ Prompt string }
		if json.Unmarshal(ev.Payload, &p) == nil && p.Prompt != "" {
			prompts = append(prompts, "- "+truncate(p.Prompt, 160))
		}
	}
	if len(prompts) > maxEpisodePrompts { // keep the beginning and the end
		n := len(prompts)
		prompts = append(append(prompts[:maxEpisodePrompts/2:maxEpisodePrompts/2], fmt.Sprintf("- … %d more prompts …", n-maxEpisodePrompts)),
			prompts[n-maxEpisodePrompts/2:]...)
	}
	content := fmt.Sprintf("Session with %d prompts.\nPrompts:\n%s", s.Prompts, strings.Join(prompts, "\n"))
	if len(files) > 0 {
		content += "\nFiles edited: " + strings.Join(files, ", ")
	}
	return content, nil
}

// maxEpisodePrompts bounds the prompts quoted in an episode.
const maxEpisodePrompts = 20

// truncate cuts s to at most n runes, adding an ellipsis.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n-1]) + "…"
}
