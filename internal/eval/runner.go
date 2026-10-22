package eval

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/hooks"
	"github.com/vhavlena/plasticity/internal/model"
)

// Learning modes: how the simulated user turns recalls into learning.
const (
	// LearnNone: recall only; the baseline without learning.
	LearnNone = "none"
	// LearnReinforce: each session ends with `reinforce --session`
	// (unsupervised: co-recalled, independently activated nodes are linked).
	LearnReinforce = "reinforce"
	// LearnFeedback: after each recall, `feedback` with the relevant nodes
	// that were actually recalled (what Claude could report as used).
	LearnFeedback = "feedback"
	// LearnExplicit: after each recall, `reinforce` with all relevant nodes,
	// as if Claude found the missing ones by other means (upper bound).
	LearnExplicit = "explicit"
	// LearnSession: as in real use: the session's recalls leave traces, and
	// only the queries' scripted `used` reports (Query.Used) give feedback,
	// against all recalls of the session since the previous report.
	LearnSession = "session"
)

// LearnModes lists the valid modes.
var LearnModes = []string{LearnNone, LearnReinforce, LearnFeedback, LearnExplicit, LearnSession}

// Config is one evaluated configuration.
type Config struct {
	Name     string            // label in the report
	Learn    string            // learning mode
	Embedder embed.Embedder    // nil: lexical seeding only
	Params   map[string]string // parameter overrides
	Passes   int               // how often the session list is replayed (default 1)
	K        int               // result limit per recall (default 10)
	// ConsolidateEvery runs `consolidate` after every n sessions (0: only
	// after each pass).
	ConsolidateEvery int
}

// QueryResult records one evaluated recall.
type QueryResult struct {
	Pass       int          `json:"pass"`
	Session    int          `json:"session"`
	Query      int          `json:"query"`
	Text       string       `json:"q"`
	Source     string       `json:"source,omitempty"`
	Skipped    bool         `json:"skipped,omitempty"` // filtered out by the prompt hook
	Hits       []string     `json:"hits"`              // node keys, best first
	Metrics    QueryMetrics `json:"metrics"`
	Confidence float64      `json:"confidence"`
	Traced     bool         `json:"traced"`
	LatencyMS  float64      `json:"latency_ms"`
}

// GraphHealth describes the graph after the run.
type GraphHealth struct {
	Nodes   int `json:"nodes"`
	Dormant int `json:"dormant"`
	Typed   int `json:"typed_edges"`
	// Preloaded are associative edges that came with the scenario; Learned
	// are associative edges created during the run.
	Preloaded int `json:"preloaded_edges"`
	Learned   int `json:"learned_edges"`
	// Spurious learned edges connect nodes that are never relevant to the
	// same query in the scenario.
	Spurious     int     `json:"spurious_edges"`
	SpuriousRate float64 `json:"spurious_rate"`
	MeanWeight   float64 `json:"mean_learned_weight"`
	MaxDegree    int     `json:"max_learned_degree"`
	// CrossProject learned edges connect nodes of different projects.
	CrossProject int `json:"cross_project_edges"`
	// Active learned edges are strong enough to carry activation from a
	// fully active neighbor (effective weight · hop decay ≥ firing
	// threshold); ActiveSpurious of them are spurious.
	ActiveLearned  int `json:"active_learned_edges"`
	ActiveSpurious int `json:"active_spurious_edges"`
	// PreloadedWeight is the mean effective weight of the preloaded
	// associative edges at the end (their scenario weight at the start).
	PreloadedWeight float64 `json:"mean_preloaded_weight"`
	PendingTraces   int     `json:"pending_traces"`
}

// Perf collects timings and sizes.
type Perf struct {
	LoadMS      float64 `json:"load_ms"`
	Recall      Latency `json:"recall"`
	Learn       Latency `json:"learn"`
	Consolidate Latency `json:"consolidate"`
	HeapInUseMB float64 `json:"heap_in_use_mb"`
	DBSizeMB    float64 `json:"db_size_mb"`
}

// Result is the outcome of one configuration.
type Result struct {
	Config string            `json:"config"`
	Learn  string            `json:"learn"`
	Embed  string            `json:"embed"`
	Params map[string]string `json:"params,omitempty"`
	Passes []Aggregate       `json:"passes"` // one aggregate per pass
	Graph  GraphHealth       `json:"graph"`
	// Warnings are the engine's non-fatal problems (e.g. embedding
	// failures that silently fell back to lexical seeding), deduplicated.
	Warnings []string      `json:"warnings,omitempty"`
	Perf     Perf          `json:"perf"`
	Queries  []QueryResult `json:"queries,omitempty"`
}

// Run replays the scenario under cfg in a fresh database inside dir.
func Run(s *Scenario, cfg Config, dir string) (*Result, error) {
	if cfg.Passes <= 0 {
		cfg.Passes = 1
	}
	if cfg.K <= 0 {
		cfg.K = 10
	}
	if cfg.Learn == "" {
		cfg.Learn = LearnNone
	}
	valid := false
	for _, m := range LearnModes {
		valid = valid || m == cfg.Learn
	}
	if !valid {
		return nil, fmt.Errorf("unknown learning mode %q", cfg.Learn)
	}
	res := &Result{Config: cfg.Name, Learn: cfg.Learn, Embed: "none", Params: cfg.Params}
	opts := []engine.Option{engine.WithoutEmbeddings()}
	if cfg.Embedder != nil {
		opts = []engine.Option{engine.WithEmbedder(cfg.Embedder)}
		res.Embed = cfg.Embedder.ID()
	}
	dbPath := filepath.Join(dir, "eval.db")
	e, err := engine.Open(dbPath, opts...)
	if err != nil {
		return nil, err
	}
	defer e.Close()
	for k, v := range cfg.Params {
		if err := e.SetParam(k, v); err != nil {
			return nil, err
		}
	}

	// Load the knowledge base.
	t0 := time.Now()
	idOf, keyOf, err := load(e, s)
	if err != nil {
		return nil, err
	}
	res.Perf.LoadMS = ms(time.Since(t0))

	nodeTerms := make(map[string]map[string]bool, len(s.Nodes))
	project := make(map[string]string, len(s.Nodes))
	byKey := make(map[string]ScenarioNode, len(s.Nodes))
	for _, n := range s.Nodes {
		nodeTerms[n.Key] = terms(n.Title + " " + n.Content + " " + n.Rationale + " " + strings.Join(n.Tags, " "))
		project[n.Key] = n.Project
		byKey[n.Key] = n
	}

	hookCfg := hooks.ConfigFromEnv() // e.g. PLASTICITY_NOTIFICATIONS=all replays the old behavior
	var recallT, learnT, consT []time.Duration
	sessionsRun := 0
	consolidate := func() error {
		t := time.Now()
		_, err := e.Consolidate()
		consT = append(consT, time.Since(t))
		return err
	}
	for pass := 1; pass <= cfg.Passes; pass++ {
		var passResults []QueryResult
		for si, sess := range s.Sessions {
			sessionID := fmt.Sprintf("p%d-s%d", pass, si)
			if err := e.StartSession(sessionID, sess.Project, "eval"); err != nil {
				return nil, err
			}
			for qi, q := range sess.Queries {
				// The session decides which traces learning sees: per session for
				// reinforce and session, per recall for feedback; none and
				// explicit store no traces (explicit learns from the labels).
				sid := ""
				switch cfg.Learn {
				case LearnReinforce, LearnSession:
					sid = sessionID
				case LearnFeedback:
					sid = fmt.Sprintf("%s-q%d", sessionID, qi)
				}
				// User prompts advance the session's prompt count (as in the
				// prompt hook), which bounds the context window of added neurons;
				// Claude's own recalls and notifications do not.
				if q.Source == "" || q.Source == SourcePrompt {
					if err := e.NotePrompt(sessionID, sess.Project, q.Q); err != nil {
						return nil, err
					}
				}
				qr := QueryResult{Pass: pass, Session: si, Query: qi, Text: q.Q, Source: q.Source, Hits: []string{}}
				// Prompts and notifications are planned like the prompt hook does.
				req := engine.RecallRequest{Query: q.Q, Session: sid, Project: sess.Project, Limit: cfg.K}
				if q.Source == SourceNotification && q.Planned {
					req.ExcludeSuperseded, req.Auto = true, true
					req.MinConfidence = hookCfg.NotificationMinConfidence
					if hookCfg.NotificationLimit > 0 {
						req.Limit = min(req.Limit, hookCfg.NotificationLimit)
					}
				} else if q.Source == SourcePrompt || q.Source == SourceNotification {
					req.ExcludeSuperseded, req.Auto = true, true // as the prompt hook
					plan := hookCfg.PlanRecall(q.Q, func(string) string { return q.OutputTail })
					qr.Skipped = plan.Skip
					req.Query, req.MinConfidence = plan.Query, plan.MinConfidence
					if plan.Limit > 0 {
						req.Limit = min(req.Limit, plan.Limit)
					}
				}
				if !qr.Skipped {
					t := time.Now()
					r, err := e.Recall(req)
					d := time.Since(t)
					if err != nil {
						return nil, fmt.Errorf("pass %d session %d query %d: %w", pass, si, qi, err)
					}
					recallT = append(recallT, d)
					for _, h := range r.Hits {
						qr.Hits = append(qr.Hits, keyOf[h.Node.ID])
					}
					qr.Confidence, qr.Traced, qr.LatencyMS = r.Confidence, r.RecallID != "", ms(d)
				}
				qr.Metrics = score(qr.Hits, q.Relevant, associative(q.Q, q.Relevant, nodeTerms))
				qr.Metrics.noise(qr.Hits, q, sess.Project, project)
				passResults = append(passResults, qr)

				t := time.Now()
				if err := learnAfterQuery(e, cfg.Learn, sid, q, qr.Hits, idOf); err != nil {
					return nil, err
				}
				if cfg.Learn == LearnFeedback || cfg.Learn == LearnExplicit || cfg.Learn == LearnSession {
					learnT = append(learnT, time.Since(t))
				}
				for _, k := range q.Add {
					if err := addDuring(e, byKey[k], sessionID, idOf, keyOf); err != nil {
						return nil, fmt.Errorf("pass %d session %d query %d: add %q: %w", pass, si, qi, k, err)
					}
					if err := applySupersession(e, s, idOf); err != nil {
						return nil, err
					}
				}
			}
			if cfg.Learn == LearnReinforce {
				t := time.Now()
				if _, err := e.ReinforceSession(sessionID); err != nil {
					return nil, err
				}
				learnT = append(learnT, time.Since(t))
			}
			sessionsRun++
			if cfg.ConsolidateEvery > 0 && sessionsRun%cfg.ConsolidateEvery == 0 {
				if err := consolidate(); err != nil {
					return nil, err
				}
			}
		}
		if cfg.ConsolidateEvery == 0 {
			if err := consolidate(); err != nil {
				return nil, err
			}
		}
		res.Passes = append(res.Passes, aggregate(passResults))
		res.Queries = append(res.Queries, passResults...)
	}

	res.Perf.Recall, res.Perf.Learn, res.Perf.Consolidate = latency(recallT), latency(learnT), latency(consT)
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	res.Perf.HeapInUseMB = float64(mem.HeapInuse) / (1 << 20)
	res.Perf.DBSizeMB = fileMB(dbPath) + fileMB(dbPath+"-wal")
	seen := map[string]bool{}
	for _, w := range e.Warnings {
		if !seen[w] {
			seen[w] = true
			res.Warnings = append(res.Warnings, w)
		}
	}
	res.Graph, err = graphHealth(e, s, keyOf, project)
	return res, err
}

func newNode(n ScenarioNode) *model.Node {
	return &model.Node{Type: n.Type, Title: n.Title, Content: n.Content, Rationale: n.Rationale,
		Tags: n.Tags, Importance: n.Importance, Source: "eval"}
}

// load inserts the scenario's nodes (except those added during sessions)
// and links, returning key↔ID maps.
func load(e *engine.Engine, s *Scenario) (idOf, keyOf map[string]string, err error) {
	deferred := s.deferred()
	var nodes []*model.Node
	var keys []string
	for _, n := range s.Nodes {
		if !deferred[n.Key] {
			nodes = append(nodes, newNode(n))
			keys = append(keys, n.Key)
		}
	}
	if err := e.AddNodes(nodes); err != nil {
		return nil, nil, err
	}
	idOf, keyOf = make(map[string]string, len(s.Nodes)), make(map[string]string, len(s.Nodes))
	for i, n := range nodes {
		idOf[keys[i]], keyOf[n.ID] = n.ID, keys[i]
	}
	// Project records, as sessions would have left them: created in the
	// node's project, used once in each of UsedIn.
	for _, n := range s.Nodes {
		id := idOf[n.Key]
		if id == "" {
			continue
		}
		if n.Project != "" {
			if err := e.NoteNodeProject(id, n.Project, true, 0); err != nil {
				return nil, nil, err
			}
		}
		for _, p := range n.UsedIn {
			if err := e.NoteNodeProject(id, p, false, 1); err != nil {
				return nil, nil, err
			}
		}
	}
	links := make([]engine.LinkSpec, len(s.Links))
	for i, l := range s.Links {
		links[i] = engine.LinkSpec{Src: idOf[l.Src], Dst: idOf[l.Dst], Kind: l.Kind, Weight: l.Weight}
	}
	if err := e.AddLinks(links); err != nil {
		return nil, nil, err
	}
	return idOf, keyOf, applySupersession(e, s, idOf)
}

// applySupersession marks every scenario supersession whose two neurons
// exist (again marking one is harmless).
func applySupersession(e *engine.Engine, s *Scenario, idOf map[string]string) error {
	for _, n := range s.Nodes {
		old, succ := idOf[n.Key], idOf[n.SupersededBy]
		if n.SupersededBy == "" || old == "" || succ == "" {
			continue
		}
		if _, err := e.UpdateNode(old, engine.NodePatch{SupersededBy: &succ}); err != nil {
			return fmt.Errorf("supersede %q by %q: %w", n.Key, n.SupersededBy, err)
		}
	}
	return nil
}

// addDuring stores a node during a session the way `plasticity add
// --session` does, so it gets the engine's add-time context links. On a
// later pass the node already exists and is left alone.
func addDuring(e *engine.Engine, n ScenarioNode, session string, idOf, keyOf map[string]string) error {
	if idOf[n.Key] != "" {
		return nil
	}
	res, err := e.AddNodeChecked(newNode(n), engine.AddOptions{Session: session, Force: true})
	if err != nil {
		return err
	}
	idOf[n.Key], keyOf[res.Node.ID] = res.Node.ID, n.Key
	return nil
}

// learnAfterQuery applies per-recall learning for the feedback, explicit
// and session modes.
func learnAfterQuery(e *engine.Engine, mode, session string, q Query, hits []string, idOf map[string]string) error {
	switch mode {
	case LearnFeedback:
		var used []string
		if len(q.Used) > 0 {
			used = existing(q.Used, idOf)
		} else {
			rel := map[string]bool{}
			for _, r := range q.Relevant {
				rel[r] = true
			}
			for _, h := range hits {
				if rel[h] {
					used = append(used, idOf[h])
				}
			}
		}
		if len(used) == 0 {
			return nil // nothing useful was recalled: nothing to report
		}
		_, err := e.Feedback(session, used)
		return err
	case LearnSession:
		used := existing(q.Used, idOf)
		if len(used) == 0 {
			return nil
		}
		_, err := e.Used(session, used)
		return err
	case LearnExplicit:
		if len(q.Relevant) < 2 {
			return nil
		}
		ids := make([]string, len(q.Relevant))
		for i, r := range q.Relevant {
			ids[i] = idOf[r]
		}
		_, err := e.Reinforce(ids)
		return err
	}
	return nil
}

// existing maps keys to IDs, leaving out nodes not yet added.
func existing(keys []string, idOf map[string]string) []string {
	var ids []string
	for _, k := range keys {
		if id := idOf[k]; id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// graphHealth inspects the final graph.
func graphHealth(e *engine.Engine, s *Scenario, keyOf, project map[string]string) (GraphHealth, error) {
	var g GraphHealth
	st, err := e.Stats()
	if err != nil {
		return g, err
	}
	g.Nodes, g.Dormant, g.PendingTraces = st.Nodes, st.DormantNodes, st.PendingTraces
	_, edges, err := e.Snapshot()
	if err != nil {
		return g, err
	}
	preloaded := map[[2]string]bool{}
	for _, l := range s.Links {
		if !l.Kind.Typed() {
			preloaded[[2]string{l.Src, l.Dst}], preloaded[[2]string{l.Dst, l.Src}] = true, true
		}
	}
	coRelevant := map[[2]string]bool{}
	for _, sess := range s.Sessions {
		for _, q := range sess.Queries {
			for _, a := range q.Relevant {
				for _, b := range q.Relevant {
					coRelevant[[2]string{a, b}] = true
				}
			}
		}
	}
	degree := map[string]int{}
	sum, preSum := 0.0, 0.0
	for _, ed := range edges {
		if ed.Kind.Typed() {
			g.Typed++
			continue
		}
		a, b := keyOf[ed.Src], keyOf[ed.Dst]
		if preloaded[[2]string{a, b}] {
			g.Preloaded++
			preSum += ed.EffectiveWeight
			continue
		}
		g.Learned++
		sum += ed.EffectiveWeight
		degree[ed.Src]++
		degree[ed.Dst]++
		spurious := !coRelevant[[2]string{a, b}]
		if spurious {
			g.Spurious++
		}
		if ed.EffectiveWeight*e.P.HopDecay >= e.P.FiringThreshold {
			g.ActiveLearned++
			if spurious {
				g.ActiveSpurious++
			}
		}
		if project[a] != "" && project[b] != "" && project[a] != project[b] {
			g.CrossProject++
		}
	}
	if g.Preloaded > 0 {
		g.PreloadedWeight = preSum / float64(g.Preloaded)
	}
	if g.Learned > 0 {
		g.MeanWeight = sum / float64(g.Learned)
		g.SpuriousRate = float64(g.Spurious) / float64(g.Learned)
	}
	for _, d := range degree {
		g.MaxDegree = max(g.MaxDegree, d)
	}
	return g, nil
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func fileMB(path string) float64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return float64(fi.Size()) / (1 << 20)
}
