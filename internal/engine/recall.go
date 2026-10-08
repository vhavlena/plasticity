package engine

import (
	"math"
	"sort"

	"github.com/oklog/ulid/v2"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/retrieval"
	"github.com/vhavlena/plasticity/internal/store"
)

func (e *Engine) bm25Weights() [4]float64 {
	return [4]float64{e.P.BM25Title, e.P.BM25Content, e.P.BM25Rationale, e.P.BM25Tags}
}

// Search runs lexical BM25 search only (no spreading, no learning).
func (e *Engine) Search(query string, limit int) ([]store.SearchHit, error) {
	var hits []store.SearchHit
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		hits, err = tx.Search(MatchExpr(query), e.bm25Weights(), limit)
		return err
	})
	if hits == nil {
		hits = []store.SearchHit{}
	}
	return hits, err
}

// txGraph exposes the stored graph, with decayed weights, to retrieval.
type txGraph struct {
	tx *store.Tx
	p  model.Params
}

func (g txGraph) Neighbors(id string) ([]retrieval.Neighbor, int, error) {
	edges, err := g.tx.EdgesOf(id, g.p.EdgesPerNode)
	if err != nil {
		return nil, 0, err
	}
	deg := len(edges)
	if deg == g.p.EdgesPerNode {
		if deg, err = g.tx.Degree(id); err != nil {
			return nil, 0, err
		}
	}
	out := make([]retrieval.Neighbor, 0, len(edges))
	for _, ed := range edges {
		if w := plasticity.EffectiveWeight(ed, g.p); w > 0 {
			out = append(out, retrieval.Neighbor{ID: ed.Other(id), Weight: w, Kind: string(ed.Kind)})
		}
	}
	return out, deg, nil
}

// RecallRequest parameterizes a recall. Zero values select the defaults.
type RecallRequest struct {
	Query     string
	Session   string // if set, a trace is stored for later reinforcement
	Limit     int
	Threshold float64
	HopDecay  float64
	Rerank    string // "" or "ppr"
	NoRecord  bool   // do not record accesses or traces
}

// RecallHit is one recalled neuron.
type RecallHit struct {
	Node       *model.Node `json:"node"`
	Score      float64     `json:"score"`
	Activation float64     `json:"activation"`
	BaseLevel  *float64    `json:"base_level,omitempty"`
	IsSeed     bool        `json:"is_seed"`
	Hops       int         `json:"hops"`
	Path       []string    `json:"path"`
	PathTitles []string    `json:"path_titles"`
}

// RecallResult is the answer to a recall.
type RecallResult struct {
	// RecallID is set when a learning trace was stored.
	RecallID string `json:"recall_id,omitempty"`
	// TraceSkipped explains why no trace was stored despite a session.
	TraceSkipped string `json:"trace_skipped,omitempty"`
	Query        string `json:"query"`
	// Confidence is the IDF-weighted share of the query explained by the
	// lexical matches (see computeSeeds).
	Confidence float64     `json:"confidence"`
	Terms      []TermInfo  `json:"terms"`
	Seeds      int         `json:"seeds"`
	Reached    int         `json:"reached"`
	Expanded   int         `json:"expanded"`
	Hits       []RecallHit `json:"hits"`
}

// Recall finds calibrated BM25 seeds, spreads activation through the
// weighted graph, ranks by λ1·graph + λ2·σ(B) + λ3·importance, and (unless
// NoRecord) records accesses and, if the query confidence is high enough, a
// trace for Hebbian learning.
func (e *Engine) Recall(req RecallRequest) (*RecallResult, error) {
	p := e.P
	if req.Limit <= 0 {
		req.Limit = p.ResultLimit
	}
	cfg := retrieval.SpreadConfig{
		HopDecay: p.HopDecay, Threshold: p.FiringThreshold,
		FanK0: p.FanK0, FanBeta: p.FanBeta, MaxNodes: p.MaxNodes, MaxHops: p.MaxHops,
	}
	if req.Threshold > 0 {
		cfg.Threshold = req.Threshold
	}
	if req.HopDecay > 0 {
		cfg.HopDecay = math.Min(1, req.HopDecay)
	}
	now := timestamp()
	res := &RecallResult{Query: req.Query, Terms: []TermInfo{}, Hits: []RecallHit{}}

	err := e.st.Read(func(tx *store.Tx) error {
		terms := queryTerms(req.Query)
		hits, err := tx.Search(MatchExpr(req.Query), e.bm25Weights(), p.SeedLimit)
		if err != nil {
			return err
		}
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		sd, err := e.computeSeeds(tx, terms, hits)
		if err != nil {
			return err
		}
		seeds := sd.seeds
		res.Confidence, res.Terms, res.Seeds = sd.confidence, sd.terms, len(seeds)
		if len(seeds) == 0 {
			return nil
		}

		spread, err := retrieval.Spread(txGraph{tx, p}, seeds, cfg)
		if err != nil {
			return err
		}
		res.Reached, res.Expanded = len(spread.Activations), spread.Expanded

		graphScore := make(map[string]float64, len(spread.Activations))
		ids := make([]string, 0, len(spread.Activations))
		for id, a := range spread.Activations {
			ids = append(ids, id)
			graphScore[id] = a.Score
		}
		sort.Strings(ids)
		if req.Rerank == "ppr" {
			pr := retrieval.PersonalizedPageRank(ids, spread.Edges, seeds, p.PPRRestart, 1e-8, 200)
			top := 0.0
			for _, v := range pr {
				top = math.Max(top, v)
			}
			for id, v := range pr {
				graphScore[id] = v / top
			}
		}

		nodes, err := tx.GetNodes(ids)
		if err != nil {
			return err
		}
		acc, err := tx.Accesses(ids)
		if err != nil {
			return err
		}
		for _, id := range ids {
			n, a := nodes[id], spread.Activations[id]
			if n == nil {
				continue
			}
			b := plasticity.BaseLevel(acc[id], tick, p.ActrDecay)
			h := RecallHit{
				Node: n, Activation: a.Score, IsSeed: a.IsSeed, Hops: a.Hops, Path: a.Path,
				Score: p.ScoreGraph*graphScore[id] + p.ScoreBaseLevel*plasticity.Sigmoid(b) + p.ScoreImportance*n.Importance,
			}
			if len(acc[id]) > 0 {
				h.BaseLevel = &b
			}
			for _, pid := range a.Path {
				if pn := nodes[pid]; pn != nil {
					h.PathTitles = append(h.PathTitles, pn.Title)
				}
			}
			res.Hits = append(res.Hits, h)
		}
		sort.Slice(res.Hits, func(i, j int) bool {
			if res.Hits[i].Score != res.Hits[j].Score {
				return res.Hits[i].Score > res.Hits[j].Score
			}
			return res.Hits[i].Node.ID < res.Hits[j].Node.ID
		})
		if len(res.Hits) > req.Limit {
			res.Hits = res.Hits[:req.Limit]
		}
		return nil
	})
	if err != nil || req.NoRecord || len(res.Hits) == 0 {
		return res, err
	}

	// Learning gate: only confident recalls, and only strongly activated
	// nodes, may later rewire the graph.
	tr := &store.Trace{Session: req.Session}
	if req.Session != "" {
		if res.Confidence < p.LearnMinConfidence {
			res.TraceSkipped = "low query confidence"
		} else {
			for _, h := range res.Hits {
				if h.Activation >= p.LearnMinActivation {
					tr.Entries = append(tr.Entries, store.TraceEntry{NodeID: h.Node.ID, Activation: h.Activation, Path: h.Path})
				}
			}
			if len(tr.Entries) == 0 {
				res.TraceSkipped = "no strongly activated nodes"
			} else {
				tr.RecallID = ulid.MustNew(ulid.Timestamp(now), ulid.DefaultEntropy()).String()
				res.RecallID = tr.RecallID
			}
		}
	}
	// A recorded recall advances the global clock and fires the recalled
	// neurons, which ages their synapses to neurons that did not fire.
	err = e.st.Write(func(tx *store.Tx) error {
		tick, err := tx.AdvanceTick()
		if err != nil {
			return err
		}
		tr.Tick = tick
		ids := make([]string, len(res.Hits))
		for i, h := range res.Hits {
			ids[i] = h.Node.ID
		}
		if err := tx.Fire(ids); err != nil {
			return err
		}
		for _, h := range res.Hits {
			h.Node.Fires++
			if err := tx.RecordAccess(h.Node.ID, tick, p.MaxAccessRecords); err != nil {
				return err
			}
			if h.Node.Dormant { // recalling a dormant memory reactivates it
				if err := tx.SetDormant(h.Node.ID, false); err != nil {
					return err
				}
				h.Node.Dormant = false
			}
		}
		if tr.RecallID == "" {
			return nil
		}
		return tx.InsertTrace(tr)
	})
	return res, err
}
