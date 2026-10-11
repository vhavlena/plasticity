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
)

// LearnModes lists the valid modes.
var LearnModes = []string{LearnNone, LearnReinforce, LearnFeedback, LearnExplicit}

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
	Hits       []string     `json:"hits"` // node keys, best first
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
	Spurious      int     `json:"spurious_edges"`
	SpuriousRate  float64 `json:"spurious_rate"`
	MeanWeight    float64 `json:"mean_learned_weight"`
	MaxDegree     int     `json:"max_learned_degree"`
	PendingTraces int     `json:"pending_traces"`
}

// Perf collects timings and sizes.
type Perf struct {
	LoadMS        float64 `json:"load_ms"`
	Recall        Latency `json:"recall"`
	Learn         Latency `json:"learn"`
	Consolidate   Latency `json:"consolidate"`
	HeapInUseMB   float64 `json:"heap_in_use_mb"`
	DBSizeMB      float64 `json:"db_size_mb"`
	ConsolidatePr int     `json:"pruned_total"`
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
	for _, n := range s.Nodes {
		nodeTerms[n.Key] = terms(n.Title + " " + n.Content + " " + n.Rationale + " " + strings.Join(n.Tags, " "))
	}

	var recallT, learnT, consT []time.Duration
	sessionsRun := 0
	consolidate := func() error {
		t := time.Now()
		rep, err := e.Consolidate(false)
		consT = append(consT, time.Since(t))
		if err == nil {
			res.Perf.ConsolidatePr += rep.Pruned
		}
		return err
	}
	for pass := 1; pass <= cfg.Passes; pass++ {
		var passResults []QueryResult
		for si, sess := range s.Sessions {
			sessionID := fmt.Sprintf("p%d-s%d", pass, si)
			for qi, q := range sess.Queries {
				// The session decides which traces learning sees: per session for
				// reinforce, per recall for feedback; none and explicit store no
				// traces (explicit learns from the labels directly).
				sid := ""
				switch cfg.Learn {
				case LearnReinforce:
					sid = sessionID
				case LearnFeedback:
					sid = fmt.Sprintf("%s-q%d", sessionID, qi)
				}
				t := time.Now()
				r, err := e.Recall(engine.RecallRequest{Query: q.Q, Session: sid, Limit: cfg.K})
				d := time.Since(t)
				if err != nil {
					return nil, fmt.Errorf("pass %d session %d query %d: %w", pass, si, qi, err)
				}
				recallT = append(recallT, d)
				hits := make([]string, len(r.Hits))
				for i, h := range r.Hits {
					hits[i] = keyOf[h.Node.ID]
				}
				qr := QueryResult{
					Pass: pass, Session: si, Query: qi, Text: q.Q, Hits: hits,
					Metrics:    score(hits, q.Relevant, associative(q.Q, q.Relevant, nodeTerms)),
					Confidence: r.Confidence, Traced: r.RecallID != "", LatencyMS: ms(d),
				}
				passResults = append(passResults, qr)

				t = time.Now()
				if err := learnAfterQuery(e, cfg.Learn, sid, q, hits, idOf); err != nil {
					return nil, err
				}
				if cfg.Learn == LearnFeedback || cfg.Learn == LearnExplicit {
					learnT = append(learnT, time.Since(t))
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
	res.Graph, err = graphHealth(e, s, keyOf)
	return res, err
}

// load inserts the scenario's nodes and links, returning key↔ID maps.
func load(e *engine.Engine, s *Scenario) (idOf, keyOf map[string]string, err error) {
	nodes := make([]*model.Node, len(s.Nodes))
	for i, n := range s.Nodes {
		nodes[i] = &model.Node{Type: n.Type, Title: n.Title, Content: n.Content, Rationale: n.Rationale,
			Tags: n.Tags, Importance: n.Importance, Source: "eval"}
	}
	if err := e.AddNodes(nodes); err != nil {
		return nil, nil, err
	}
	idOf, keyOf = make(map[string]string, len(nodes)), make(map[string]string, len(nodes))
	for i, n := range nodes {
		idOf[s.Nodes[i].Key], keyOf[n.ID] = n.ID, s.Nodes[i].Key
	}
	links := make([]engine.LinkSpec, len(s.Links))
	for i, l := range s.Links {
		links[i] = engine.LinkSpec{Src: idOf[l.Src], Dst: idOf[l.Dst], Kind: l.Kind, Weight: l.Weight}
	}
	return idOf, keyOf, e.AddLinks(links)
}

// learnAfterQuery applies per-recall learning for the feedback and explicit
// modes.
func learnAfterQuery(e *engine.Engine, mode, session string, q Query, hits []string, idOf map[string]string) error {
	switch mode {
	case LearnFeedback:
		rel := map[string]bool{}
		for _, r := range q.Relevant {
			rel[r] = true
		}
		var used []string
		for _, h := range hits {
			if rel[h] {
				used = append(used, idOf[h])
			}
		}
		if len(used) == 0 {
			return nil // nothing useful was recalled: nothing to report
		}
		_, err := e.Feedback(session, used)
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

// graphHealth inspects the final graph.
func graphHealth(e *engine.Engine, s *Scenario, keyOf map[string]string) (GraphHealth, error) {
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
	sum := 0.0
	for _, ed := range edges {
		if ed.Kind.Typed() {
			g.Typed++
			continue
		}
		if preloaded[[2]string{keyOf[ed.Src], keyOf[ed.Dst]}] {
			g.Preloaded++
			continue
		}
		g.Learned++
		sum += ed.EffectiveWeight
		degree[ed.Src]++
		degree[ed.Dst]++
		if !coRelevant[[2]string{keyOf[ed.Src], keyOf[ed.Dst]}] {
			g.Spurious++
		}
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
